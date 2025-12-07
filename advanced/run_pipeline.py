from .llm_embeddings import VideoEmbeddingGenerator, ChannelEmbeddingAggregator
from .feature_fusion import FeatureFusion
from .graph_construction import ChannelGraphBuilder
from .community_detection import CommunityDetector
from .temporal_analysis import TemporalAnalyzer


def run_llm_embedding_pipeline() -> None:
    video_emb_gen = VideoEmbeddingGenerator()
    video_emb_gen.run()

    channel_agg = ChannelEmbeddingAggregator()
    channel_agg.run()


def run_full_advanced_pipeline() -> None:
    """Run the entire advanced pipeline end-to-end.

    Steps:
    1) LLM embeddings (video & channel)
    2) Feature fusion (video & channel)
    3) Channel graph construction
    4) Community detection / clustering
    5) Temporal analysis
    """

    run_llm_embedding_pipeline()

    fusion = FeatureFusion()
    fusion.run()

    graph_builder = ChannelGraphBuilder()
    graph_builder.run()

    community_detector = CommunityDetector()
    community_detector.run()

    temporal_analyzer = TemporalAnalyzer()
    temporal_analyzer.run()


if __name__ == "__main__":
    run_full_advanced_pipeline()

