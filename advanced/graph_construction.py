import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import CONFIG


class ChannelGraphBuilder:
	"""Build a similarity graph over channels based on fused features.

	Nodes: channels
	Edges: similarity (cosine over fused features), sparsified by top-k / threshold.
	"""

	def __init__(self) -> None:
		self.cfg = CONFIG.graph
		self.paths = CONFIG.data_paths

	def load_channel_fused_features(self) -> Tuple[pd.DataFrame, np.ndarray]:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		index_path = os.path.join(out_dir, "channel_fused_features_index.csv")
		feat_path = os.path.join(out_dir, "channel_fused_features.npy")

		df_idx = pd.read_csv(index_path)
		feats = np.load(feat_path)
		return df_idx, feats

	def compute_similarity_matrix(self, feats: np.ndarray) -> np.ndarray:
		sim = cosine_similarity(feats)
		np.fill_diagonal(sim, 0.0)
		return sim

	def build_edge_list(self,
						df_idx: pd.DataFrame,
						sim: np.ndarray) -> pd.DataFrame:
		n = sim.shape[0]
		k = self.cfg.similarity_top_k
		thr = self.cfg.similarity_threshold

		# Align node identifier with channel-level fused index file.
		# For channel_fused_features_index.csv header starts with e.g.:
		# channel_id_x or channel_id / channel_name_x / channel_name
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
				"found in channel_fused_features_index.csv"
			)

		sources = []
		targets = []
		weights = []

		for i in range(n):
			row = sim[i]
			idx_sorted = np.argsort(row)[::-1]
			count = 0
			for j in idx_sorted:
				if i == j:
					continue
				w = row[j]
				if w < thr:
					break
				sources.append(df_idx.loc[i, node_col])
				targets.append(df_idx.loc[j, node_col])
				weights.append(float(w))
				count += 1
				if count >= k:
					break

		edges = pd.DataFrame({
			"source": sources,
			"target": targets,
			"weight": weights,
			"view_type": "fused_semantic",
		})

		return edges

	def save_graph(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		os.makedirs(out_dir, exist_ok=True)
		nodes_df.to_csv(os.path.join(out_dir, "channel_graph_nodes.csv"), index=False)
		edges_df.to_csv(os.path.join(out_dir, "channel_graph_edges.csv"), index=False)

	def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		print("Step 3/5: Building channel similarity graph from fused features...")
		df_idx, feats = self.load_channel_fused_features()
		sim = self.compute_similarity_matrix(feats)
		edges = self.build_edge_list(df_idx, sim)
		self.save_graph(df_idx, edges)
		return df_idx, edges

