import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SupplyChainPredictiveAnalytics:
    """
    Comprehensive predictive analytics system for supply chain disruption forecasting
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def prepare_features_for_prediction(self, df):
        """
        Prepare comprehensive feature set for ML models
        """
        print("🔧 Preparing features for predictive analytics...")
        
        # Base features
        features = df[['tons_2023', 'value_2023', 'tmiles_2023', 'trade_type']].copy()
        
        # Engineered features
        features['tons_per_mile'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        features['value_per_mile'] = df['value_2023'] / (df['tmiles_2023'] + 1)
        features['value_per_ton'] = df['value_2023'] / (df['tons_2023'] + 1)
        features['efficiency_ratio'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        
        # Geographic features
        features['origin_region'] = df['fr_orig']
        features['destination_region'] = df['fr_dest']
        features['origin_state'] = df['dms_origst']
        features['destination_state'] = df['dms_destst']
        
        # Volatility features (if available)
        year_columns = [col for col in df.columns if 'tons_' in col and col != 'tons_2023']
        if len(year_columns) > 1:
            tons_data = df[year_columns]
            features['tons_volatility'] = tons_data.std(axis=1)
            features['tons_trend'] = tons_data.iloc[:, -1] - tons_data.iloc[:, 0]
        
        # Categorical encoding
        categorical_cols = ['origin_region', 'destination_region', 'origin_state', 'destination_state']
        for col in categorical_cols:
            if col in features.columns:
                le = LabelEncoder()
                features[col + '_encoded'] = le.fit_transform(features[col].astype(str))
                self.encoders[col] = le
        
        # Remove original categorical columns
        features = features.drop(columns=[col for col in categorical_cols if col in features.columns])
        
        # Handle missing values
        features = features.fillna(0)
        
        print(f"   • Created {features.shape[1]} features for prediction")
        return features
    
    def build_disruption_prediction_model(self, df):
        """
        Build ML model to predict supply chain disruptions
        """
        print("🚨 Building disruption prediction model...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(df)
        
        # Create disruption target (example: high volatility = potential disruption)
        if 'tons_volatility' in X.columns:
            # Define disruption as high volatility
            disruption_threshold = X['tons_volatility'].quantile(0.8)
            y_disruption = (X['tons_volatility'] > disruption_threshold).astype(int)
        else:
            # Alternative: use efficiency ratio as proxy
            efficiency_threshold = X['efficiency_ratio'].quantile(0.2)
            y_disruption = (X['efficiency_ratio'] < efficiency_threshold).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_disruption, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = model.score(X_test, y_test)
                else:
                    # Regression model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                
                print(f"   • {name} Score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"   • {name} failed: {str(e)}")
        
        # Store best model
        self.models['disruption_prediction'] = best_model
        
        # Feature importance for Random Forest
        if isinstance(best_model, RandomForestClassifier):
            self.feature_importance['disruption'] = dict(zip(X.columns, best_model.feature_importances_))
        
        print(f"   • Best Model: {type(best_model).__name__} (Score: {best_score:.3f})")
        
        return best_model, best_score
    
    def build_cost_prediction_model(self, df):
        """
        Build ML model to predict transportation costs
        """
        print("💰 Building cost prediction model...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(df)
        
        # Target: value per ton (proxy for cost)
        y_cost = X['value_per_ton']
        X_cost = X.drop(columns=['value_per_ton'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf'),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                
                print(f"   • {name} R² Score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"   • {name} failed: {str(e)}")
        
        # Store best model
        self.models['cost_prediction'] = best_model
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance['cost'] = dict(zip(X_cost.columns, best_model.feature_importances_))
        
        print(f"   • Best Cost Model: {type(best_model).__name__} (R²: {best_score:.3f})")
        
        return best_model, best_score
    
    def build_capacity_prediction_model(self, df):
        """
        Build ML model to predict capacity constraints
        """
        print("📦 Building capacity prediction model...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(df)
        
        # Target: tons per mile (capacity utilization proxy)
        y_capacity = X['tons_per_mile']
        X_capacity = X.drop(columns=['tons_per_mile'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_capacity, y_capacity, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf'),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                
                print(f"   • {name} R² Score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"   • {name} failed: {str(e)}")
        
        # Store best model
        self.models['capacity_prediction'] = best_model
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance['capacity'] = dict(zip(X_capacity.columns, best_model.feature_importances_))
        
        print(f"   • Best Capacity Model: {type(best_model).__name__} (R²: {best_score:.3f})")
        
        return best_model, best_score
    
    def build_delivery_delay_prediction_model(self, df):
        """
        Build ML model to predict delivery delays
        """
        print("⏰ Building delivery delay prediction model...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(df)
        
        # Create delay target (example: low efficiency = potential delay)
        efficiency_threshold = X['efficiency_ratio'].quantile(0.3)
        y_delay = (X['efficiency_ratio'] < efficiency_threshold).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
        
        # Train classification models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                print(f"   • {name} Accuracy: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"   • {name} failed: {str(e)}")
        
        # Store best model
        self.models['delay_prediction'] = best_model
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance['delay'] = dict(zip(X.columns, best_model.feature_importances_))
        
        print(f"   • Best Delay Model: {type(best_model).__name__} (Accuracy: {best_score:.3f})")
        
        return best_model, best_score
    
    def cluster_supply_chains(self, df, n_clusters=5):
        """
        Cluster supply chains to identify patterns and segments
        """
        print("🎯 Clustering supply chains...")
        
        # Prepare features for clustering
        X = self.prepare_features_for_prediction(df)
        
        # Use PCA for dimensionality reduction
        if X.shape[1] > 10:
            pca = PCA(n_components=min(10, X.shape[1]))
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X.values
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['supply_chain_cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['supply_chain_cluster'] == cluster_id]
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'avg_tons': cluster_data['tons_2023'].mean(),
                'avg_value': cluster_data['value_2023'].mean(),
                'avg_distance': cluster_data['tmiles_2023'].mean(),
                'avg_efficiency': (cluster_data['tons_2023'] / (cluster_data['tmiles_2023'] + 1)).mean()
            }
        
        print(f"   • Identified {n_clusters} supply chain clusters")
        for cluster_id, analysis in cluster_analysis.items():
            print(f"   • Cluster {cluster_id}: {analysis['size']} corridors, "
                  f"Avg Efficiency: {analysis['avg_efficiency']:.3f}")
        
        return df_clustered, cluster_analysis
    
    def generate_predictions(self, df):
        """
        Generate predictions for all models
        """
        print("🔮 Generating predictions...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(df)
        
        predictions = {}
        
        # Generate predictions for each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    pred_proba = model.predict_proba(X)
                    predictions[f'{model_name}_probability'] = pred_proba[:, 1]  # Probability of positive class
                    predictions[f'{model_name}_prediction'] = model.predict(X)
                else:
                    # Regression model
                    predictions[f'{model_name}_prediction'] = model.predict(X)
                    
            except Exception as e:
                print(f"   • Warning: {model_name} prediction failed: {str(e)}")
        
        # Add predictions to dataframe
        df_with_predictions = df.copy()
        for pred_name, pred_values in predictions.items():
            df_with_predictions[pred_name] = pred_values
        
        return df_with_predictions, predictions
    
    def generate_feature_importance_report(self):
        """
        Generate feature importance report for all models
        """
        print("📊 Generating feature importance report...")
        
        importance_report = {}
        
        for model_type, importance_dict in self.feature_importance.items():
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            importance_report[model_type] = {
                'top_features': sorted_features[:10],  # Top 10 features
                'total_features': len(sorted_features),
                'avg_importance': np.mean(list(importance_dict.values())),
                'std_importance': np.std(list(importance_dict.values()))
            }
            
            print(f"   • {model_type.title()} Model:")
            print(f"     - Top Feature: {sorted_features[0][0]} ({sorted_features[0][1]:.3f})")
            print(f"     - Average Importance: {importance_report[model_type]['avg_importance']:.3f}")
        
        return importance_report
    
    def comprehensive_prediction_analysis(self, df):
        """
        Run comprehensive predictive analytics
        """
        print("🤖 COMPREHENSIVE PREDICTIVE ANALYTICS")
        print("=" * 60)
        
        # Build all prediction models
        disruption_model, disruption_score = self.build_disruption_prediction_model(df)
        cost_model, cost_score = self.build_cost_prediction_model(df)
        capacity_model, capacity_score = self.build_capacity_prediction_model(df)
        delay_model, delay_score = self.build_delivery_delay_prediction_model(df)
        
        # Cluster supply chains
        df_clustered, cluster_analysis = self.cluster_supply_chains(df)
        
        # Generate predictions
        df_with_predictions, predictions = self.generate_predictions(df)
        
        # Generate feature importance report
        importance_report = self.generate_feature_importance_report()
        
        # Summary report
        summary = {
            'models_trained': len(self.models),
            'disruption_prediction_score': disruption_score,
            'cost_prediction_score': cost_score,
            'capacity_prediction_score': capacity_score,
            'delay_prediction_score': delay_score,
            'clusters_identified': len(cluster_analysis),
            'predictions_generated': len(predictions)
        }
        
        print("\n📈 PREDICTIVE ANALYTICS SUMMARY:")
        print(f"   • Models Trained: {summary['models_trained']}")
        print(f"   • Disruption Prediction Score: {summary['disruption_prediction_score']:.3f}")
        print(f"   • Cost Prediction R²: {summary['cost_prediction_score']:.3f}")
        print(f"   • Capacity Prediction R²: {summary['capacity_prediction_score']:.3f}")
        print(f"   • Delay Prediction Accuracy: {summary['delay_prediction_score']:.3f}")
        print(f"   • Supply Chain Clusters: {summary['clusters_identified']}")
        print(f"   • Prediction Types Generated: {summary['predictions_generated']}")
        
        return df_with_predictions, self.models, cluster_analysis, importance_report, summary

# Usage function for notebook integration
def implement_predictive_analytics_in_notebook(df):
    """
    Function to integrate predictive analytics into the main notebook
    """
    print("🤖 IMPLEMENTING PREDICTIVE ANALYTICS FOR DISRUPTION FORECASTING")
    print("=" * 70)
    
    # Initialize predictive analytics
    predictor = SupplyChainPredictiveAnalytics()
    
    # Run comprehensive analysis
    df_with_predictions, models, cluster_analysis, importance_report, summary = predictor.comprehensive_prediction_analysis(df)
    
    return predictor, df_with_predictions, models, cluster_analysis, importance_report, summary 