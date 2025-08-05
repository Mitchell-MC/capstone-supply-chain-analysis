import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SupplyChainRiskScorer:
    """
    Comprehensive risk scoring system for international supply chains
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.risk_models = {}
        self.cluster_model = None
        self.anomaly_detector = None
        
    def calculate_geographic_risk(self, df):
        """
        Calculate geographic concentration risk using clustering and distance analysis
        """
        # Extract geographic features
        geo_features = df[['fr_orig', 'fr_dest', 'dms_origst', 'dms_destst']].copy()
        
        # Create distance-based risk metrics
        geo_features['origin_diversity'] = df.groupby('fr_orig')['value_2023'].transform('count')
        geo_features['destination_diversity'] = df.groupby('fr_dest')['value_2023'].transform('count')
        
        # Calculate concentration risk (Herfindahl-Hirschman Index)
        origin_concentration = df.groupby('fr_orig')['value_2023'].sum().pow(2).sum() / df['value_2023'].sum().pow(2)
        dest_concentration = df.groupby('fr_dest')['value_2023'].sum().pow(2).sum() / df['value_2023'].sum().pow(2)
        
        geo_features['origin_concentration_risk'] = origin_concentration
        geo_features['destination_concentration_risk'] = dest_concentration
        
        # Normalize risk scores
        geo_risk = (geo_features['origin_concentration_risk'] + 
                   geo_features['destination_concentration_risk']) / 2
        
        return geo_risk
    
    def calculate_mode_risk(self, df):
        """
        Calculate transportation mode dependency risk using ML clustering
        """
        # Extract mode-related features
        mode_features = df[['trade_type', 'tons_2023', 'tmiles_2023']].copy()
        
        # Create mode efficiency metrics
        mode_features['tons_per_mile'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        mode_features['value_per_mile'] = df['value_2023'] / (df['tmiles_2023'] + 1)
        
        # Use K-means clustering to identify mode risk clusters
        if len(mode_features) > 10:  # Need sufficient data for clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            mode_features['mode_cluster'] = kmeans.fit_predict(mode_features[['tons_per_mile', 'value_per_mile']])
            
            # Calculate cluster-based risk (higher risk for less efficient clusters)
            cluster_risk = mode_features.groupby('mode_cluster')['tons_per_mile'].mean()
            mode_features['mode_efficiency_risk'] = mode_features['mode_cluster'].map(cluster_risk)
        else:
            mode_features['mode_efficiency_risk'] = 0.5  # Default risk
            
        return mode_features['mode_efficiency_risk']
    
    def calculate_volatility_risk(self, df):
        """
        Calculate economic volatility risk using time series analysis
        """
        # Calculate volatility metrics across years
        year_columns = [col for col in df.columns if 'tons_' in col and col != 'tons_2023']
        
        if len(year_columns) > 1:
            # Calculate coefficient of variation for tons
            tons_data = df[year_columns]
            volatility = tons_data.std(axis=1) / (tons_data.mean(axis=1) + 1e-6)
            
            # Normalize volatility to 0-1 scale
            volatility_risk = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-6)
        else:
            volatility_risk = pd.Series(0.5, index=df.index)
            
        return volatility_risk
    
    def calculate_infrastructure_risk(self, df):
        """
        Calculate infrastructure vulnerability risk using anomaly detection
        """
        # Create infrastructure-related features
        infra_features = df[['tons_2023', 'tmiles_2023', 'value_2023']].copy()
        
        # Add derived features
        infra_features['tons_per_mile'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        infra_features['value_density'] = df['value_2023'] / (df['tons_2023'] + 1)
        
        # Use Isolation Forest for anomaly detection
        if len(infra_features) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(infra_features)
            
            # Convert to risk scores (anomalies = higher risk)
            infra_risk = (anomaly_scores == -1).astype(float)
        else:
            infra_risk = pd.Series(0.1, index=df.index)
            
        return infra_risk
    
    def calculate_comprehensive_risk_score(self, df):
        """
        Calculate comprehensive risk score using weighted combination of all risk factors
        """
        # Calculate individual risk components
        geo_risk = self.calculate_geographic_risk(df)
        mode_risk = self.calculate_mode_risk(df)
        volatility_risk = self.calculate_volatility_risk(df)
        infra_risk = self.calculate_infrastructure_risk(df)
        
        # Weight the risk components (can be adjusted based on business priorities)
        weights = {
            'geographic': 0.3,
            'mode': 0.25,
            'volatility': 0.25,
            'infrastructure': 0.2
        }
        
        # Calculate weighted risk score
        comprehensive_risk = (
            geo_risk * weights['geographic'] +
            mode_risk * weights['mode'] +
            volatility_risk * weights['volatility'] +
            infra_risk * weights['infrastructure']
        )
        
        return comprehensive_risk
    
    def build_risk_prediction_model(self, df):
        """
        Build ML model to predict future risk scores
        """
        # Prepare features for risk prediction
        features = df[['tons_2023', 'value_2023', 'tmiles_2023', 'trade_type']].copy()
        
        # Add engineered features
        features['tons_per_mile'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        features['value_per_mile'] = df['value_2023'] / (df['tmiles_2023'] + 1)
        features['efficiency_ratio'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        
        # Calculate current risk as target
        current_risk = self.calculate_comprehensive_risk_score(df)
        
        # Prepare data for ML
        X = features.fillna(0)
        y = current_risk
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Risk Prediction Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Store model
        self.risk_models['prediction'] = rf_model
        
        return rf_model, mse, r2
    
    def identify_high_risk_corridors(self, df, risk_threshold=0.7):
        """
        Identify high-risk supply chain corridors
        """
        risk_scores = self.calculate_comprehensive_risk_score(df)
        
        # Identify high-risk corridors
        high_risk_mask = risk_scores > risk_threshold
        high_risk_corridors = df[high_risk_mask].copy()
        high_risk_corridors['risk_score'] = risk_scores[high_risk_mask]
        
        return high_risk_corridors.sort_values('risk_score', ascending=False)
    
    def generate_risk_report(self, df):
        """
        Generate comprehensive risk analysis report
        """
        risk_scores = self.calculate_comprehensive_risk_score(df)
        
        report = {
            'total_corridors': len(df),
            'high_risk_corridors': len(risk_scores[risk_scores > 0.7]),
            'medium_risk_corridors': len(risk_scores[(risk_scores > 0.4) & (risk_scores <= 0.7)]),
            'low_risk_corridors': len(risk_scores[risk_scores <= 0.4]),
            'average_risk_score': risk_scores.mean(),
            'risk_score_std': risk_scores.std(),
            'max_risk_score': risk_scores.max(),
            'min_risk_score': risk_scores.min()
        }
        
        return report

# Usage example:
def implement_risk_scoring_in_notebook(df):
    """
    Function to integrate risk scoring into the main notebook
    """
    print("🔍 IMPLEMENTING COMPREHENSIVE RISK SCORING SYSTEM")
    print("=" * 60)
    
    # Initialize risk scorer
    risk_scorer = SupplyChainRiskScorer()
    
    # Calculate comprehensive risk scores
    print("📊 Calculating risk scores...")
    risk_scores = risk_scorer.calculate_comprehensive_risk_score(df)
    df['comprehensive_risk_score'] = risk_scores
    
    # Build prediction model
    print("🤖 Building risk prediction model...")
    model, mse, r2 = risk_scorer.build_risk_prediction_model(df)
    
    # Identify high-risk corridors
    print("⚠️ Identifying high-risk corridors...")
    high_risk_corridors = risk_scorer.identify_high_risk_corridors(df)
    
    # Generate risk report
    print("📋 Generating risk analysis report...")
    risk_report = risk_scorer.generate_risk_report(df)
    
    print("\n📈 RISK ANALYSIS SUMMARY:")
    print(f"   • Total Corridors Analyzed: {risk_report['total_corridors']:,}")
    print(f"   • High-Risk Corridors: {risk_report['high_risk_corridors']:,}")
    print(f"   • Medium-Risk Corridors: {risk_report['medium_risk_corridors']:,}")
    print(f"   • Low-Risk Corridors: {risk_report['low_risk_corridors']:,}")
    print(f"   • Average Risk Score: {risk_report['average_risk_score']:.3f}")
    print(f"   • Risk Prediction Model R²: {r2:.3f}")
    
    return df, risk_scorer, high_risk_corridors, risk_report 