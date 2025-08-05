import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

class SupplyChainNetworkAnalyzer:
    """
    Network analysis system for identifying critical chokepoints and vulnerabilities
    """
    
    def __init__(self):
        self.network_graph = None
        self.centrality_metrics = {}
        self.chokepoints = {}
        self.vulnerability_scores = {}
        
    def build_supply_chain_network(self, df):
        """
        Build network graph from supply chain data
        """
        print("🌐 Building supply chain network...")
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes (origins and destinations)
        for _, row in df.iterrows():
            origin = f"O_{row['fr_orig']}_{row['dms_origst']}"
            destination = f"D_{row['fr_dest']}_{row['dms_destst']}"
            
            # Add nodes with attributes
            G.add_node(origin, type='origin', region=row['fr_orig'], state=row['dms_origst'])
            G.add_node(destination, type='destination', region=row['fr_dest'], state=row['dms_destst'])
            
            # Add edge with flow attributes
            G.add_edge(origin, destination, 
                      tons=row['tons_2023'],
                      value=row['value_2023'],
                      distance=row['tmiles_2023'],
                      trade_type=row['trade_type'])
        
        self.network_graph = G
        print(f"   • Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def calculate_network_centrality(self, G):
        """
        Calculate various centrality metrics to identify critical nodes
        """
        print("📊 Calculating network centrality metrics...")
        
        centrality_metrics = {}
        
        # Degree centrality
        centrality_metrics['degree'] = nx.degree_centrality(G)
        
        # Betweenness centrality (identifies bridges/chokepoints)
        centrality_metrics['betweenness'] = nx.betweenness_centrality(G)
        
        # Closeness centrality
        centrality_metrics['closeness'] = nx.closeness_centrality(G)
        
        # Eigenvector centrality (importance based on connections to important nodes)
        centrality_metrics['eigenvector'] = nx.eigenvector_centrality_numpy(G)
        
        # Flow centrality (custom metric based on tons/value flow)
        flow_centrality = {}
        for node in G.nodes():
            in_flow = sum([G[predecessor][node]['tons'] for predecessor in G.predecessors(node)])
            out_flow = sum([G[node][successor]['tons'] for successor in G.successors(node)])
            flow_centrality[node] = (in_flow + out_flow) / (sum([G[u][v]['tons'] for u, v in G.edges()]) + 1e-6)
        
        centrality_metrics['flow'] = flow_centrality
        
        self.centrality_metrics = centrality_metrics
        return centrality_metrics
    
    def identify_critical_chokepoints(self, G, centrality_threshold=0.8):
        """
        Identify critical chokepoints using centrality analysis
        """
        print("⚠️ Identifying critical chokepoints...")
        
        if not self.centrality_metrics:
            self.calculate_network_centrality(G)
        
        chokepoints = {}
        
        # Identify nodes with high betweenness centrality (bridges)
        betweenness_threshold = np.percentile(list(self.centrality_metrics['betweenness'].values()), 
                                           centrality_threshold * 100)
        
        bridge_nodes = [node for node, centrality in self.centrality_metrics['betweenness'].items() 
                       if centrality > betweenness_threshold]
        
        # Identify nodes with high flow centrality
        flow_threshold = np.percentile(list(self.centrality_metrics['flow'].values()), 
                                     centrality_threshold * 100)
        
        flow_nodes = [node for node, centrality in self.centrality_metrics['flow'].items() 
                     if centrality > flow_threshold]
        
        # Combine critical nodes
        critical_nodes = list(set(bridge_nodes + flow_nodes))
        
        chokepoints['critical_nodes'] = critical_nodes
        chokepoints['bridge_nodes'] = bridge_nodes
        chokepoints['flow_nodes'] = flow_nodes
        
        print(f"   • Identified {len(critical_nodes)} critical chokepoints")
        print(f"   • Bridge nodes: {len(bridge_nodes)}")
        print(f"   • High-flow nodes: {len(flow_nodes)}")
        
        self.chokepoints = chokepoints
        return chokepoints
    
    def calculate_vulnerability_scores(self, G):
        """
        Calculate vulnerability scores for each node using ML
        """
        print("🔍 Calculating vulnerability scores...")
        
        vulnerability_scores = {}
        
        # Prepare features for vulnerability analysis
        node_features = []
        node_names = []
        
        for node in G.nodes():
            features = {
                'degree': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
                'betweenness': self.centrality_metrics['betweenness'].get(node, 0),
                'closeness': self.centrality_metrics['closeness'].get(node, 0),
                'flow_centrality': self.centrality_metrics['flow'].get(node, 0),
                'avg_tons': np.mean([G[u][v]['tons'] for u, v in G.edges() if u == node or v == node]),
                'avg_value': np.mean([G[u][v]['value'] for u, v in G.edges() if u == node or v == node])
            }
            
            node_features.append(list(features.values()))
            node_names.append(node)
        
        # Convert to numpy array
        X = np.array(node_features)
        
        # Use Isolation Forest for anomaly detection (vulnerable nodes = anomalies)
        if len(X) > 10:
            iso_forest = IsolationForest(contamination=0.15, random_state=42)
            anomaly_scores = iso_forest.fit_predict(X)
            
            # Convert to vulnerability scores
            for i, node in enumerate(node_names):
                vulnerability_scores[node] = 1.0 if anomaly_scores[i] == -1 else 0.0
        else:
            # Default vulnerability scores
            for node in node_names:
                vulnerability_scores[node] = 0.5
        
        self.vulnerability_scores = vulnerability_scores
        return vulnerability_scores
    
    def cluster_network_nodes(self, G, n_clusters=5):
        """
        Cluster network nodes to identify similar regions/modes
        """
        print("🎯 Clustering network nodes...")
        
        # Prepare features for clustering
        node_features = []
        node_names = []
        
        for node in G.nodes():
            features = [
                self.centrality_metrics['degree'].get(node, 0),
                self.centrality_metrics['betweenness'].get(node, 0),
                self.centrality_metrics['flow'].get(node, 0),
                G.degree(node),
                np.mean([G[u][v]['tons'] for u, v in G.edges() if u == node or v == node])
            ]
            
            node_features.append(features)
            node_names.append(node)
        
        X = np.array(node_features)
        
        # Use DBSCAN for clustering
        if len(X) > 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            clusters = dbscan.fit_predict(X_scaled)
            
            # Create cluster mapping
            cluster_mapping = {}
            for i, node in enumerate(node_names):
                cluster_mapping[node] = clusters[i]
            
            print(f"   • Identified {len(set(clusters))} node clusters")
            return cluster_mapping
        else:
            print("   • Insufficient data for clustering")
            return {node: 0 for node in node_names}
    
    def analyze_network_resilience(self, G):
        """
        Analyze overall network resilience
        """
        print("🛡️ Analyzing network resilience...")
        
        resilience_metrics = {}
        
        # Network density
        resilience_metrics['density'] = nx.density(G)
        
        # Network diameter
        try:
            resilience_metrics['diameter'] = nx.diameter(G)
        except:
            resilience_metrics['diameter'] = float('inf')
        
        # Average shortest path length
        try:
            resilience_metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        except:
            resilience_metrics['avg_path_length'] = float('inf')
        
        # Clustering coefficient
        resilience_metrics['clustering_coefficient'] = nx.average_clustering(G)
        
        # Network efficiency
        resilience_metrics['efficiency'] = nx.global_efficiency(G)
        
        # Redundancy (number of alternative paths)
        total_paths = 0
        for source in list(G.nodes())[:10]:  # Sample for performance
            for target in list(G.nodes())[:10]:
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=3))
                        total_paths += len(paths)
                    except:
                        pass
        
        resilience_metrics['path_redundancy'] = total_paths / 100  # Normalized
        
        print(f"   • Network Density: {resilience_metrics['density']:.3f}")
        print(f"   • Network Diameter: {resilience_metrics['diameter']}")
        print(f"   • Clustering Coefficient: {resilience_metrics['clustering_coefficient']:.3f}")
        print(f"   • Network Efficiency: {resilience_metrics['efficiency']:.3f}")
        
        return resilience_metrics
    
    def generate_network_report(self, df):
        """
        Generate comprehensive network analysis report
        """
        print("📋 Generating network analysis report...")
        
        # Build network
        G = self.build_supply_chain_network(df)
        
        # Calculate centrality
        centrality = self.calculate_network_centrality(G)
        
        # Identify chokepoints
        chokepoints = self.identify_critical_chokepoints(G)
        
        # Calculate vulnerability scores
        vulnerability = self.calculate_vulnerability_scores(G)
        
        # Cluster nodes
        clusters = self.cluster_network_nodes(G)
        
        # Analyze resilience
        resilience = self.analyze_network_resilience(G)
        
        # Generate report
        report = {
            'network_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': resilience['density'],
                'diameter': resilience['diameter'],
                'clustering_coefficient': resilience['clustering_coefficient']
            },
            'chokepoints': {
                'critical_nodes': len(chokepoints['critical_nodes']),
                'bridge_nodes': len(chokepoints['bridge_nodes']),
                'flow_nodes': len(chokepoints['flow_nodes'])
            },
            'vulnerability': {
                'high_vulnerability_nodes': len([v for v in vulnerability.values() if v > 0.8]),
                'avg_vulnerability_score': np.mean(list(vulnerability.values())),
                'vulnerability_std': np.std(list(vulnerability.values()))
            },
            'clusters': {
                'num_clusters': len(set(clusters.values())),
                'cluster_distribution': pd.Series(clusters.values()).value_counts().to_dict()
            }
        }
        
        return report, G, centrality, chokepoints, vulnerability, clusters, resilience

# Usage function for notebook integration
def implement_network_analysis_in_notebook(df):
    """
    Function to integrate network analysis into the main notebook
    """
    print("🌐 IMPLEMENTING NETWORK ANALYSIS & CHOKEPOINT IDENTIFICATION")
    print("=" * 70)
    
    # Initialize network analyzer
    analyzer = SupplyChainNetworkAnalyzer()
    
    # Generate comprehensive network analysis
    report, G, centrality, chokepoints, vulnerability, clusters, resilience = analyzer.generate_network_report(df)
    
    print("\n📊 NETWORK ANALYSIS SUMMARY:")
    print(f"   • Network Nodes: {report['network_stats']['nodes']:,}")
    print(f"   • Network Edges: {report['network_stats']['edges']:,}")
    print(f"   • Critical Chokepoints: {report['chokepoints']['critical_nodes']:,}")
    print(f"   • High Vulnerability Nodes: {report['vulnerability']['high_vulnerability_nodes']:,}")
    print(f"   • Network Clusters: {report['clusters']['num_clusters']}")
    print(f"   • Network Density: {report['network_stats']['density']:.3f}")
    print(f"   • Average Vulnerability Score: {report['vulnerability']['avg_vulnerability_score']:.3f}")
    
    return analyzer, report, G, centrality, chokepoints, vulnerability, clusters, resilience 