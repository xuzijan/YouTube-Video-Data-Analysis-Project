import os
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import CONFIG


class LocalEmbeddingClient:
    """Use a local sentence-transformers model to generate embeddings.

    This replaces the previous DeepSeek embedding client to avoid
    relying on a remote embeddings API.
    """

    def __init__(self) -> None:
        # 你可以按需要换成更大的模型，比如 paraphrase-multilingual-MiniLM-L12-v2
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
        self.model = SentenceTransformer(model_name)
        self.batch_size = 64

    def get_embedding(self, text: str) -> List[float]:
        emb = self.model.encode([text], show_progress_bar=False)
        return emb[0].tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # Disable internal progress bar; we manage a single outer tqdm per phase.
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return emb.tolist()


class VideoTextBuilder:
    def __init__(self) -> None:
        self.cfg = CONFIG.embedding_text

    def build_text_view_a(self, row: pd.Series) -> str:
        title = str(row.get("title", ""))
        category = str(row.get("categoryId", ""))
        channel = str(row.get("channelTitle", ""))
        return f"Title: {title}\nCategory: {category}\nChannel: {channel}"

    def build_text_view_b(self, row: pd.Series) -> str:
        views = row.get("view_count", row.get("views", ""))
        likes = row.get("like_count", row.get("likes", ""))
        comments = row.get("comment_count", row.get("comments", ""))
        year = row.get("publish_year", "")
        month = row.get("publish_month", "")
        stats = f"Stats: {views} views, {likes} likes, {comments} comments."
        time_str = f"Published: {year}-{month}"
        return f"{time_str}\n{stats}"

    def build_final_text(self, row: pd.Series) -> str:
        parts: List[str] = []
        if self.cfg.use_view_a:
            parts.append(self.build_text_view_a(row))
        if self.cfg.use_view_b:
            parts.append(self.build_text_view_b(row))
        return "\n".join(p for p in parts if p)


class VideoEmbeddingGenerator:
    def __init__(self) -> None:
        self.paths = CONFIG.data_paths
        self.client = LocalEmbeddingClient()
        self.text_builder = VideoTextBuilder()

    def load_raw_videos(self) -> pd.DataFrame:
        csv_path = os.path.join(self.paths.root_dir, self.paths.raw_video_csv)
        return pd.read_csv(csv_path)

    def compute_video_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [self.text_builder.build_final_text(row) for _, row in df.iterrows()]
        embeddings: List[List[float]] = []
        batch_size = self.client.batch_size

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Step 1/5: Computing video embeddings",
            ncols=100,
        ):
            batch = texts[i : i + batch_size]
            batch_emb = self.client.get_embeddings_batch(batch)
            embeddings.extend(batch_emb)
        emb_arr = np.array(embeddings, dtype=np.float32)
        df_out = df.copy()
        df_out["embedding_index"] = range(len(df_out))
        out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "video_embeddings_deepseek.npy"), emb_arr)
        return df_out

    def save_video_embeddings_index(self, df_with_index: pd.DataFrame) -> None:
        out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        df_with_index.to_csv(os.path.join(out_dir, "video_embedding_index.csv"), index=False)

    def run(self) -> None:
        print("Step 1/5: Computing video & channel embeddings...")
        video_df = self.load_raw_videos()
        df_with_index = self.compute_video_embeddings(video_df)
        self.save_video_embeddings_index(df_with_index)


class ChannelEmbeddingAggregator:
    def __init__(self) -> None:
        self.paths = CONFIG.data_paths

    def load_video_embeddings_index(self) -> pd.DataFrame:
        out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
        index_path = os.path.join(out_dir, "video_embedding_index.csv")
        return pd.read_csv(index_path)

    def load_video_embedding_matrix(self) -> np.ndarray:
        out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
        emb_path = os.path.join(out_dir, "video_embeddings_deepseek.npy")
        return np.load(emb_path)

    def aggregate_to_channels(self) -> pd.DataFrame:
        df_idx = self.load_video_embeddings_index()
        emb_mat = self.load_video_embedding_matrix()

        # Use actual columns from video_embedding_index.csv
        # header: video_id,title,channel_name,channel_id,view_count,like_count,comment_count,published_date,thumbnail,embedding_index
        if "channel_id" in df_idx.columns:
            group_key = "channel_id"
        elif "channelId" in df_idx.columns:
            group_key = "channelId"
        elif "channel_name" in df_idx.columns:
            group_key = "channel_name"
        else:
            raise KeyError("No channel identifier column (channel_id/channelId/channel_name) found in video_embedding_index.csv")
        grouped = df_idx.groupby(group_key)["embedding_index"].apply(list).reset_index()
        channel_ids: List[str] = []
        channel_embs: List[np.ndarray] = []
        for _, row in grouped.iterrows():
            idx_list = row["embedding_index"]
            vecs = emb_mat[idx_list]
            mean_vec = vecs.mean(axis=0)
            channel_ids.append(row[group_key])
            channel_embs.append(mean_vec)
        channel_emb_arr = np.vstack(channel_embs)
        out_dir = os.path.join(self.paths.root_dir, self.paths.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "channel_embeddings_deepseek.npy"), channel_emb_arr)
        df_channels = pd.DataFrame({group_key: channel_ids, "embedding_index": range(len(channel_ids))})
        df_channels.to_csv(os.path.join(out_dir, "channel_embedding_index.csv"), index=False)
        return df_channels

    def run(self) -> None:
        self.aggregate_to_channels()
