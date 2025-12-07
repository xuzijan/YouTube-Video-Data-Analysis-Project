import os

import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans

from .config import CONFIG


class CommunityDetector:
	"""Run clustering in embedding space and community detection on the graph."""

	def __init__(self) -> None:
		self.cfg = CONFIG.clustering
		self.paths = CONFIG.data_paths

	def load_channel_fused_features(self) -> pd.DataFrame:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		index_path = os.path.join(out_dir, "channel_fused_features_index.csv")
		return pd.read_csv(index_path)

	def load_channel_fused_matrix(self) -> pd.DataFrame:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		feat_path = os.path.join(out_dir, "channel_fused_features.npy")
		import numpy as np

		return np.load(feat_path)

	def load_graph_edges(self) -> pd.DataFrame:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		edges_path = os.path.join(out_dir, "channel_graph_edges.csv")
		return pd.read_csv(edges_path)

	def cluster_in_embedding_space(self, df_idx: pd.DataFrame, feats) -> pd.DataFrame:
		model = KMeans(n_clusters=self.cfg.channel_n_clusters, random_state=self.cfg.random_state)
		labels = model.fit_predict(feats)

		df_out = df_idx.copy()
		df_out["cluster_id"] = labels
		return df_out

	def detect_graph_communities(self, df_idx: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
		# Align node identifier with channel_fused_features_index.csv header.
		# channel_fused_features_index.csv starts with: channel_id_x,fusion_embedding_index
		if "channel_id_x" in df_idx.columns:
			node_col = "channel_id_x"
		elif "channel_id" in df_idx.columns:
			node_col = "channel_id"
		elif "channelId" in df_idx.columns:
			node_col = "channelId"
		elif "channel_name_x" in df_idx.columns:
			node_col = "channel_name_x"
		elif "channel_name" in df_idx.columns:
			node_col = "channel_name"
		else:
			raise KeyError(
				"No channel identifier column (channel_id_x/channel_id/channelId/channel_name_x/channel_name) "
				"found in channel_fused_features_index dataframe"
			)

		G = nx.Graph()
		for _, row in df_idx.iterrows():
			G.add_node(row[node_col])
		for _, row in edges_df.iterrows():
			G.add_edge(row["source"], row["target"], weight=row["weight"])

		try:
			import community as community_louvain

			partition = community_louvain.best_partition(G, weight="weight")
		except ImportError:
			# fallback: connected components id as community
			partition = {}
			for cid, comp in enumerate(nx.connected_components(G)):
				for node in comp:
					partition[node] = cid

		df_comm = df_idx.copy()
		df_comm["community_id"] = df_comm[node_col].map(partition).fillna(-1).astype(int)
		return df_comm

	def save_results(self, df_clusters: pd.DataFrame) -> None:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		os.makedirs(out_dir, exist_ok=True)
		df_clusters.to_csv(os.path.join(out_dir, "channel_clusters_communities.csv"), index=False)

	def run(self) -> pd.DataFrame:
		print("Step 4/5: Running embedding-space clustering and graph community detection...")
		df_idx = self.load_channel_fused_features()
		feats = self.load_channel_fused_matrix()
		edges = self.load_graph_edges()

		df_embed_clusters = self.cluster_in_embedding_space(df_idx, feats)
		df_communities = self.detect_graph_communities(df_embed_clusters, edges)
		self.save_results(df_communities)
		return df_communities

