"""
Dataset loaders for VideoRAG pipeline.

Provides unified interfaces for loading various video Q&A datasets:
- SurveillanceVQA-589K: Large-scale surveillance video Q&A dataset
- UCF-Crime: Crime activity detection dataset (legacy support)
"""

from .surveillance_vqa import (
    SurveillanceVQADataset,
    SurveillanceVQAParser,
    VideoSample,
    QAPair,
    FlattenedQASample,
    QuestionCategory,
    load_surveillance_vqa,
    create_surveillance_vqa_benchmark
)

from .surveillance_vqa_pipeline import (
    SurveillanceVQAPipeline,
    SurveillanceVQAConfig,
    run_full_pipeline
)

from .evaluation import (
    SurveillanceVQAEvaluator,
    EvaluationResult,
    RetrievalMetrics,
    GenerationMetrics,
    run_benchmark
)

__all__ = [
    # Main dataset class
    "SurveillanceVQADataset",
    
    # Parser and data structures
    "SurveillanceVQAParser",
    "VideoSample",
    "QAPair",
    "FlattenedQASample",
    "QuestionCategory",
    
    # Pipeline integration
    "SurveillanceVQAPipeline",
    "SurveillanceVQAConfig",
    "run_full_pipeline",
    
    # Evaluation
    "SurveillanceVQAEvaluator",
    "EvaluationResult",
    "RetrievalMetrics",
    "GenerationMetrics",
    "run_benchmark",
    
    # Convenience functions
    "load_surveillance_vqa",
    "create_surveillance_vqa_benchmark",
]
