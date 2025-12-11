from dataclasses import dataclass
from typing import List


@dataclass
class DataPaths:
    root_dir: str = r"c:\\Users\\xuzijian\\Desktop\\5002"
    raw_video_csv: str = "youtube_video.csv"
    engineered_features_scaled: str = "engineered_features_scaled.csv"
    output_dir: str = "advanced_outputs"


@dataclass
class DeepSeekConfig:
    api_key: str = ""  # keep empty, use DEEPSEEK_API_KEY env if needed for chat
    base_url: str = "https://api.deepseek.com"
    embedding_model: str = ""
    enabled: bool = False
    batch_size: int = 32
    request_timeout: int = 60


@dataclass
class EmbeddingTextTemplate:
    use_view_a: bool = True
    use_view_b: bool = True


@dataclass
class GraphConfig:
    similarity_top_k: int = 20
    similarity_threshold: float = 0.4
    use_multi_view_graph: bool = False


@dataclass
class ClusteringConfig:
    channel_n_clusters: int = 10
    video_n_clusters: int = 20
    random_state: int = 42


@dataclass
class TemporalConfig:
    time_col: str = "publishedAt"
    time_granularity: str = "year"  # or "month" / "window"
    window_size_years: int = 3


@dataclass
class WebExportConfig:
    export_graph: bool = True
    export_umap: bool = True
    export_temporal: bool = True


@dataclass
class AdvancedConfig:
    data_paths: DataPaths = None
    deepseek: DeepSeekConfig = None
    embedding_text: EmbeddingTextTemplate = None
    graph: GraphConfig = None
    clustering: ClusteringConfig = None
    temporal: TemporalConfig = None
    web_export: WebExportConfig = None


CONFIG = AdvancedConfig(
    data_paths=DataPaths(),
    deepseek=DeepSeekConfig(),
    embedding_text=EmbeddingTextTemplate(),
    graph=GraphConfig(),
    clustering=ClusteringConfig(),
    temporal=TemporalConfig(),
    web_export=WebExportConfig(),
)
