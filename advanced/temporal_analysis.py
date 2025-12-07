import os
from typing import Dict

import numpy as np
import pandas as pd

from .config import CONFIG


class TemporalAnalyzer:
	"""Analyze temporal evolution of clusters and communities."""

	def __init__(self) -> None:
		self.cfg = CONFIG.temporal
		self.paths = CONFIG.data_paths

	def load_video_with_clusters(self) -> pd.DataFrame:
		"""Load video-level data and map channel clusters/communities onto it.

		Requires:
		- original youtube_video.csv
		- channel_clusters_communities.csv
		"""

		root = self.paths.root_dir
		video_df = pd.read_csv(os.path.join(root, self.paths.raw_video_csv))

		out_dir = os.path.join(root, self.paths.output_dir)
		channel_clusters = pd.read_csv(os.path.join(out_dir, "channel_clusters_communities.csv"))

		# 我们在整个 advanced 管线中，频道标识统一为 channel_id_x
		# youtube_video.csv 中对应的是 channel_id
		if "channel_id" in video_df.columns and "channel_id_x" in channel_clusters.columns:
			merged = video_df.merge(channel_clusters, left_on="channel_id", right_on="channel_id_x", how="left")
		else:
			# 如果将来表头有变化，这里给出清晰错误信息
			raise KeyError(f"Expected columns 'channel_id' in youtube_video.csv and 'channel_id_x' in channel_clusters_communities.csv, got {video_df.columns.tolist()} and {channel_clusters.columns.tolist()}")

		return merged

	def _extract_time_bucket(self, df: pd.DataFrame) -> pd.Series:
		"""Extract a year-based time bucket from publish time columns."""

		if "publish_year" in df.columns:
			years = df["publish_year"]
		elif "publishedAt" in df.columns:
			years = pd.to_datetime(df["publishedAt"], errors="coerce").dt.year
		elif "published_date" in df.columns:
			years = pd.to_datetime(df["published_date"], errors="coerce").dt.year
		else:
			years = pd.Series(np.nan, index=df.index)

		return years

	def compute_cluster_temporal_stats(self, df: pd.DataFrame) -> pd.DataFrame:
		years = self._extract_time_bucket(df)
		df = df.copy()
		df["year"] = years

		metrics = {
			"video_count": ("video_id" if "video_id" in df.columns else None),
		}

		agg_dict = {
			"video_id": "count" if "video_id" in df.columns else ("title", "count"),
		}

		# Simpler explicit aggregation
		df["video_count"] = 1
		if "view_count" in df.columns:
			df["view_count"] = df["view_count"].fillna(0)

		group_cols = ["year", "cluster_id"] if "cluster_id" in df.columns else ["year"]

		agg_spec = {"video_count": "sum"}
		if "view_count" in df.columns:
			agg_spec["view_count"] = "sum"
		if "engagement" in df.columns:
			df["engagement"] = df["engagement"].fillna(0)
			agg_spec["engagement"] = "mean"

		agg = df.groupby(group_cols).agg(agg_spec).reset_index()

		return agg

	def save_temporal_stats(self, cluster_stats: pd.DataFrame) -> None:
		out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
		os.makedirs(out_dir, exist_ok=True)
		cluster_stats.to_csv(os.path.join(out_dir, "cluster_temporal_stats.csv"), index=False)

	def run(self) -> Dict[str, pd.DataFrame]:
		print("Step 5/5: Computing temporal evolution of clusters...")
		df = self.load_video_with_clusters()
		cluster_stats = self.compute_cluster_temporal_stats(df)
		self.save_temporal_stats(cluster_stats)
		return {"cluster_stats": cluster_stats}

