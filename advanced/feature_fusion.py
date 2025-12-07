import os
from typing import Tuple

import numpy as np
import pandas as pd

from .config import CONFIG


class FeatureFusion:
	"""Fuse engineered numeric features with LLM-based embeddings.

	Pipeline:
	- load engineered_features_scaled.csv (video-level)
	- load video_embedding_index.csv + video_embeddings_deepseek.npy
	- align by video id / index
	- early fusion by concatenation
	- aggregate to channel-level fused features
	"""

	def __init__(self) -> None:
		self.paths = CONFIG.data_paths

	def load_engineered_features(self) -> pd.DataFrame:
		csv_path = os.path.join(self.paths.root_dir, self.paths.engineered_features_scaled)
		return pd.read_csv(csv_path)

	def load_video_embeddings_index(self) -> pd.DataFrame:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		index_path = os.path.join(out_dir, "video_embedding_index.csv")
		return pd.read_csv(index_path)

	def load_video_embedding_matrix(self) -> np.ndarray:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		emb_path = os.path.join(out_dir, "video_embeddings_deepseek.npy")
		return np.load(emb_path)

	def early_fusion_video(self) -> pd.DataFrame:
		"""Perform early fusion at video level.

		Returns a DataFrame that contains:
		- identifiers (e.g., video_id, channelId, publish time, etc. if present)
		- numeric engineered features
		- a column `fusion_embedding_index` pointing into saved fused matrix
		Side effect: saves `video_fused_features.npy` and `video_fused_features_index.csv`.
		"""

		df_feat = self.load_engineered_features()
		df_idx = self.load_video_embeddings_index()
		emb_mat = self.load_video_embedding_matrix()

		# Assume there is a common key `video_id` or fall back to positional join
		join_key = None
		for cand in ["video_id", "videoId", "id"]:
			if cand in df_feat.columns and cand in df_idx.columns:
				join_key = cand
				break

		if join_key is not None:
			df_merged = df_feat.merge(df_idx, on=join_key, how="inner")
		else:
			# fallback: assume same order / length
			df_merged = df_feat.copy()
			df_merged["embedding_index"] = df_idx["embedding_index"]

		# Separate numeric engineered features: drop all non-numeric / obvious id/meta columns.
		id_like_cols = {
			"video_id",
			"videoId",
			"id",
			"channel_id",
			"channelId",
			"channel_name",
			"title",
			"published_date",
			"thumbnail",
			"view_count",
			"like_count",
			"comment_count",
		}
		non_feature_cols = {c for c in df_feat.columns if c in id_like_cols}
		# Only keep columns that (1) are not in id/meta, and (2) exist in the merged df
		feature_cols = [c for c in df_feat.columns if c not in non_feature_cols and c in df_merged.columns]

		engineered_arr = df_merged[feature_cols].to_numpy(dtype=np.float32)
		emb_arr = emb_mat[df_merged["embedding_index"].to_numpy(dtype=int)]

		fused_arr = np.concatenate([engineered_arr, emb_arr], axis=1)

		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		os.makedirs(out_dir, exist_ok=True)
		np.save(os.path.join(out_dir, "video_fused_features.npy"), fused_arr)

		df_index = df_merged.copy()
		df_index["fusion_embedding_index"] = range(len(df_index))
		df_index.to_csv(os.path.join(out_dir, "video_fused_features_index.csv"), index=False)

		return df_index

	def aggregate_channel_fused_features(self) -> pd.DataFrame:
		"""Aggregate video-level fused features to channel-level by mean.

		Side effect: saves `channel_fused_features.npy` and `channel_fused_features_index.csv`.
		"""

		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		index_path = os.path.join(out_dir, "video_fused_features_index.csv")
		fused_path = os.path.join(out_dir, "video_fused_features.npy")

		df_idx = pd.read_csv(index_path)
		fused_mat = np.load(fused_path)

		# Align with actual columns from video_fused_features_index.csv
		# header starts with: video_id,title_x,channel_name_x,channel_id_x,...,title_y,channel_name_y,channel_id_y,...
		if "channel_id_x" in df_idx.columns:
			group_key = "channel_id_x"
		elif "channel_id" in df_idx.columns:
			group_key = "channel_id"
		elif "channelId" in df_idx.columns:
			group_key = "channelId"
		elif "channel_name_x" in df_idx.columns:
			group_key = "channel_name_x"
		elif "channel_name" in df_idx.columns:
			group_key = "channel_name"
		else:
			raise KeyError(
				"No channel identifier column (channel_id_x/channel_id/channelId/channel_name_x/channel_name) "
				"found in video_fused_features_index.csv"
			)

		grouped = df_idx.groupby(group_key)["fusion_embedding_index"].apply(list).reset_index()

		channel_ids = []
		channel_vecs = []
		for _, row in grouped.iterrows():
			idx_list = row["fusion_embedding_index"]
			vecs = fused_mat[idx_list]
			mean_vec = vecs.mean(axis=0)
			channel_ids.append(row[group_key])
			channel_vecs.append(mean_vec)

		channel_fused_arr = np.vstack(channel_vecs)
		np.save(os.path.join(out_dir, "channel_fused_features.npy"), channel_fused_arr)

		df_channels = pd.DataFrame({group_key: channel_ids, "fusion_embedding_index": range(len(channel_ids))})
		df_channels.to_csv(os.path.join(out_dir, "channel_fused_features_index.csv"), index=False)

		return df_channels

	def run(self) -> None:
		print("Step 2/5: Running feature fusion (video-level + channel-level)...")
		self.early_fusion_video()
		self.aggregate_channel_fused_features()

