"""
Advanced Pipeline 2.0 Result Visualization
==========================================

Generates 5 types of visualizations:
1. Network graph of channels with cluster coloring
2. UMAP/t-SNE 2D cluster distribution
3. Temporal heatmap of cluster activity
4. Top-10 channels per cluster ranking
5. Community profile summaries
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib for better output
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultVisualizer:
    def __init__(self, root_dir="c:\\Users\\xuzijian\\Desktop\\5002"):
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, "advanced_outputs")
        self.viz_dir = os.path.join(root_dir, "visualization_results")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load necessary data
        self.load_data()
    
    def load_data(self):
        """Load all required data files."""
        print("Loading data...")
        
        # Graph data
        self.nodes_df = pd.read_csv(os.path.join(self.output_dir, "channel_graph_nodes.csv"))
        self.edges_df = pd.read_csv(os.path.join(self.output_dir, "channel_graph_edges.csv"))
        
        # Clustering data
        self.clusters_df = pd.read_csv(os.path.join(self.output_dir, "channel_clusters_communities.csv"))
        
        # Temporal data
        self.temporal_df = pd.read_csv(os.path.join(self.output_dir, "cluster_temporal_stats.csv"))
        
        # Fused features for UMAP
        self.fused_features = np.load(os.path.join(self.output_dir, "channel_fused_features.npy"))
        
        # Original video data for channel info
        self.video_df = pd.read_csv(os.path.join(self.root_dir, "youtube_video.csv"))
        
        # Merge channel info with clusters
        self.channel_info = self.video_df.groupby("channel_id").agg({
            "view_count": "sum",
            "like_count": "sum",
            "comment_count": "sum",
            "channel_name": "first"
        }).reset_index()
        
        # Map channel_id to cluster_id and community_id
        id_map = self.clusters_df.set_index("channel_id_x")[["cluster_id", "community_id"]].to_dict()
        self.channel_info["cluster_id"] = self.channel_info["channel_id"].map(id_map["cluster_id"])
        self.channel_info["community_id"] = self.channel_info["channel_id"].map(id_map["community_id"])
        
        print(f"✓ Loaded {len(self.channel_info)} channels, {len(self.nodes_df)} graph nodes")
        print(f"✓ Loaded {len(self.edges_df)} edges, {len(self.clusters_df)} cluster assignments")
    
    def visualize_network_graph(self, top_k_edges=None):
        """
        Visualize channel similarity network using networkx.
        Nodes are colored by cluster, sized by view count.
        """
        print("\n[1/5] Generating network graph visualization...")
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for _, row in self.nodes_df.iterrows():
            channel_id = row["channel_id_x"]
            if channel_id in self.channel_info["channel_id"].values:
                channel_data = self.channel_info[self.channel_info["channel_id"] == channel_id].iloc[0]
                cluster_id = channel_data.get("cluster_id", -1)
                view_count = max(channel_data.get("view_count", 0), 1)
                label = channel_data.get("channel_name", channel_id)[:20]  # Truncate long names
                
                G.add_node(channel_id, 
                          label=label,
                          cluster=cluster_id,
                          views=view_count)
        
        # Add edges (use top-k if specified)
        edges_to_add = self.edges_df
        if top_k_edges:
            edges_to_add = self.edges_df.nlargest(top_k_edges, "weight")
        
        for _, row in edges_to_add.iterrows():
            if row["source"] in G and row["target"] in G:
                G.add_edge(row["source"], row["target"], weight=row["weight"])
        
        # Create visualization with matplotlib
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Use spring layout
        print("  Computing graph layout (this may take a minute)...")
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Color nodes by cluster
        cluster_colors = {}
        colors = plt.cm.tab20(np.linspace(0, 1, self.clusters_df["cluster_id"].nunique()))
        for i, cluster_id in enumerate(sorted(self.clusters_df["cluster_id"].unique())):
            cluster_colors[cluster_id] = colors[i]
        
        node_colors = [cluster_colors.get(G.nodes[node].get("cluster", -1), [0.5, 0.5, 0.5, 1]) for node in G.nodes()]
        node_sizes = [max(G.nodes[node].get("views", 0) / 1e6, 20) for node in G.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax)
        
        ax.set_title("Channel Similarity Network\n(Nodes: Channels, Colors: Clusters, Size: View Count)", 
                    fontsize=14, fontweight="bold")
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "network_graph.png"), dpi=150, bbox_inches="tight")
        print(f"✓ Network graph saved: network_graph.png")
        plt.close()
    
    def visualize_cluster_distribution(self, method="tsne"):
        """
        Visualize cluster distribution using t-SNE dimensionality reduction.
        """
        print(f"\n[2/5] Computing t-SNE 2D projection...")
        
        print(f"  Reducing {self.fused_features.shape[0]} channels × {self.fused_features.shape[1]} dims...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, metric="cosine", random_state=42, n_jobs=-1)
        embedding_2d = tsne.fit_transform(self.fused_features)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Colored by cluster_id
        scatter1 = ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                              c=self.clusters_df["cluster_id"].values,
                              s=50, alpha=0.6, cmap="tab20")
        ax1.set_title("Channel Clusters (t-SNE, Colored by Cluster ID)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("t-SNE Dimension 1")
        ax1.set_ylabel("t-SNE Dimension 2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster ID")
        
        # Plot 2: Colored by community_id
        scatter2 = ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                              c=self.clusters_df["community_id"].values,
                              s=50, alpha=0.6, cmap="tab20")
        ax2.set_title("Channel Communities (t-SNE, Colored by Community ID)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        plt.colorbar(scatter2, ax=ax2, label="Community ID")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f"cluster_distribution_tsne.png"), dpi=300, bbox_inches="tight")
        print(f"✓ Cluster distribution saved: cluster_distribution_tsne.png")
        plt.close()
    
    def visualize_temporal_heatmap(self):
        """
        Create heatmap showing cluster activity across years.
        """
        print("\n[3/5] Generating temporal activity heatmap...")
        
        # Pivot temporal data: rows=cluster, cols=year
        heatmap_data = self.temporal_df.pivot_table(
            index="cluster_id",
            columns="year",
            values="video_count",
            aggfunc="sum",
            fill_value=0
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt="g", cbar_kws={"label": "Video Count"},
                   ax=ax, linewidths=0.5)
        ax.set_title("Cluster Activity Over Years (Video Count Heatmap)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cluster ID")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "temporal_heatmap.png"), dpi=300, bbox_inches="tight")
        print(f"✓ Temporal heatmap saved: temporal_heatmap.png")
        plt.close()
    
    def rank_channels_by_cluster(self, top_n=10):
        """
        Output top-N channels per cluster by view count.
        """
        print(f"\n[4/5] Ranking top-{top_n} channels per cluster...")
        
        output_data = []
        
        for cluster_id in sorted(self.channel_info["cluster_id"].dropna().unique()):
            cluster_channels = self.channel_info[self.channel_info["cluster_id"] == cluster_id].nlargest(top_n, "view_count")
            
            for rank, (_, channel) in enumerate(cluster_channels.iterrows(), 1):
                output_data.append({
                    "cluster_id": int(cluster_id),
                    "rank": rank,
                    "channel_id": channel["channel_id"],
                    "channel_name": channel["channel_name"],
                    "view_count": int(channel["view_count"]),
                    "like_count": int(channel["like_count"]),
                    "comment_count": int(channel["comment_count"]),
                    "engagement_rate": (channel["like_count"] + channel["comment_count"]) / max(channel["view_count"], 1)
                })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join(self.viz_dir, "top_channels_per_cluster.csv"), index=False)
        print(f"✓ Top channels ranking saved: top_channels_per_cluster.csv")
        
        # Print summary
        for cluster_id in sorted(output_df["cluster_id"].unique()):
            cluster_data = output_df[output_df["cluster_id"] == cluster_id]
            print(f"\n  Cluster {int(cluster_id)} Top Channels:")
            for _, ch in cluster_data.head(3).iterrows():
                print(f"    {ch['rank']}. {ch['channel_name'][:40]:40s} - Views: {ch['view_count']:>12,} - Engagement: {ch['engagement_rate']:.4f}")
    
    def generate_community_profiles(self):
        """
        Generate summary profiles for each community.
        """
        print("\n[5/5] Generating community profiles...")
        
        profiles = []
        
        for community_id in sorted(self.clusters_df["community_id"].unique()):
            community_channels = self.clusters_df[self.clusters_df["community_id"] == community_id]["channel_id_x"].unique()
            
            community_channel_data = self.channel_info[self.channel_info["channel_id"].isin(community_channels)]
            
            # Calculate aggregated statistics
            total_views = community_channel_data["view_count"].sum()
            total_videos = len(self.video_df[self.video_df["channel_id"].isin(community_channels)])
            avg_engagement = (community_channel_data["like_count"].sum() + community_channel_data["comment_count"].sum()) / max(total_views, 1)
            num_channels = len(community_channels)
            
            # Get dominant clusters in this community
            dominant_clusters = self.clusters_df[
                self.clusters_df["community_id"] == community_id
            ]["cluster_id"].value_counts().head(3).to_dict()
            
            profiles.append({
                "community_id": int(community_id),
                "num_channels": num_channels,
                "total_views": int(total_views),
                "total_videos": total_videos,
                "avg_views_per_channel": int(total_views / max(num_channels, 1)),
                "avg_engagement_rate": avg_engagement,
                "dominant_clusters": str(list(dominant_clusters.keys())),
                "top_channel": community_channel_data.nlargest(1, "view_count")["channel_name"].values[0] if len(community_channel_data) > 0 else "N/A"
            })
        
        profiles_df = pd.DataFrame(profiles)
        profiles_df.to_csv(os.path.join(self.viz_dir, "community_profiles.csv"), index=False)
        print(f"✓ Community profiles saved: community_profiles.csv")
        
        # Print summary
        print("\n  Community Profiles Summary:")
        print("-" * 120)
        print(f"{'Community':>10} | {'#Channels':>10} | {'Total Views':>15} | {'Avg Engagement':>15} | {'Top Channel':>40}")
        print("-" * 120)
        for _, profile in profiles_df.iterrows():
            print(f"{int(profile['community_id']):>10} | {profile['num_channels']:>10} | {profile['total_views']:>15,} | "
                  f"{profile['avg_engagement_rate']:>15.6f} | {profile['top_channel'][:40]:>40}")
        print("-" * 120)
    
    def run_all(self):
        """Run all visualizations."""
        print("\n" + "="*80)
        print("Advanced Pipeline 2.0 - Result Visualization")
        print("="*80)
        
        try:
            self.visualize_network_graph(top_k_edges=50000)
            self.visualize_cluster_distribution(method="umap")
            self.visualize_temporal_heatmap()
            self.rank_channels_by_cluster(top_n=10)
            self.generate_community_profiles()
            
            print("\n" + "="*80)
            print(f"✓ All visualizations completed successfully!")
            print(f"✓ Results saved to: {self.viz_dir}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n✗ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    visualizer = ResultVisualizer()
    visualizer.run_all()
