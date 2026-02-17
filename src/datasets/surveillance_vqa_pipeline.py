"""
SurveillanceVQA Pipeline Integration for VideoRAG.

This module provides the integration layer between SurveillanceVQA dataset
and the existing VideoRAG preprocessing, feature extraction, and indexing pipeline.

Workflow:
1. Load SurveillanceVQA dataset
2. Download/locate video files
3. Run preprocessing (chunking, keyframe extraction)
4. Extract features (CLIP embeddings, YOLO detection)
5. Build FAISS index
6. Create evaluation benchmark
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import requests

# Import from existing VideoRAG modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, 
    VIDEO_EXTENSIONS, CHUNK_DURATION
)
from datasets.surveillance_vqa import (
    SurveillanceVQADataset,
    VideoSample,
    QuestionCategory,
    load_surveillance_vqa,
    create_surveillance_vqa_benchmark
)


class SurveillanceVQAConfig:
    """Configuration for SurveillanceVQA pipeline integration."""
    
    # Dataset paths
    DATASET_NAME = "fei213/SurveillanceVQA-589K"
    VIDEO_DIR = PROJECT_ROOT / "surveillance_vqa_videos"
    ANNOTATIONS_DIR = PROJECT_ROOT / "surveillance_vqa_annotations"
    OUTPUT_DIR = PROJECT_ROOT / "outputs_surveillance_vqa"
    BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "surveillance_vqa"
    
    # Processing settings
    MAX_VIDEOS = None  # None for all, or int for subset
    SPLITS = ["train", "test"]
    
    # Question categories to include for benchmarking
    BENCHMARK_CATEGORIES = [
        QuestionCategory.EVENT_DESCRIPTION,
        QuestionCategory.ANOMALY_TYPE,
        QuestionCategory.SUBJECT_IDENTIFICATION,
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.VIDEO_DIR, cls.ANNOTATIONS_DIR, 
                        cls.OUTPUT_DIR, cls.BENCHMARK_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


class SurveillanceVQAPipeline:
    """
    Pipeline for processing SurveillanceVQA dataset with VideoRAG.
    
    This class handles:
    1. Dataset loading from HuggingFace
    2. Video file management
    3. Integration with existing preprocessing pipeline
    4. Benchmark creation for evaluation
    """
    
    def __init__(
        self,
        video_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[SurveillanceVQAConfig] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            video_dir: Directory containing/for video files
            output_dir: Directory for processed outputs
            config: Optional custom configuration
        """
        self.config = config or SurveillanceVQAConfig()
        self.config.ensure_directories()
        
        self.video_dir = Path(video_dir) if video_dir else self.config.VIDEO_DIR
        self.output_dir = Path(output_dir) if output_dir else self.config.OUTPUT_DIR
        
        self.dataset: Optional[SurveillanceVQADataset] = None
        self.video_list: List[Dict] = []
    
    def load_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        from_cache: bool = True
    ) -> SurveillanceVQADataset:
        """
        Load the SurveillanceVQA dataset.
        
        Args:
            split: Dataset split ("train" or "test")
            max_samples: Maximum number of samples to load
            from_cache: Whether to use cached data if available
        
        Returns:
            SurveillanceVQADataset instance
        """
        cache_path = self.config.ANNOTATIONS_DIR / f"surveillance_vqa_{split}_cache.json"
        
        # Try to load from cache
        if from_cache and cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            self.dataset = SurveillanceVQADataset.from_json(
                json_path=str(cache_path),
                video_dir=str(self.video_dir),
                flatten=True
            )
        else:
            # Load from HuggingFace
            self.dataset = SurveillanceVQADataset.from_huggingface(
                video_dir=str(self.video_dir),
                split=split,
                max_samples=max_samples,
                flatten=True
            )
            
            # Cache for future use
            self.dataset.to_json(str(cache_path))
            print(f"Cached dataset to: {cache_path}")
        
        # Build video list
        self._build_video_list()
        
        return self.dataset
    
    def _build_video_list(self) -> None:
        """Build list of unique videos with their metadata."""
        if not self.dataset:
            return
        
        self.video_list = []
        for sample in self.dataset.samples:
            self.video_list.append({
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "anomaly_type": sample.anomaly_type,
                "num_questions": len(sample.qa_pairs),
                "exists": Path(sample.video_path).exists()
            })
    
    def check_video_availability(self) -> Dict[str, Any]:
        """
        Check which videos are available locally.
        
        Returns:
            Dict with availability statistics
        """
        if not self.video_list:
            return {"error": "No videos loaded. Call load_dataset() first."}
        
        available = sum(1 for v in self.video_list if v["exists"])
        missing = len(self.video_list) - available
        
        return {
            "total": len(self.video_list),
            "available": available,
            "missing": missing,
            "availability_rate": available / len(self.video_list) if self.video_list else 0,
            "missing_videos": [v["video_id"] for v in self.video_list if not v["exists"]]
        }
    
    def prepare_for_preprocessing(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Prepare video list for the existing preprocessing pipeline.
        
        Creates a JSON file compatible with the existing preprocess.py script.
        
        Args:
            output_path: Path to save the video list
        
        Returns:
            Path to the prepared video list
        """
        if not self.video_list:
            raise ValueError("No videos loaded. Call load_dataset() first.")
        
        output_path = output_path or str(self.output_dir / "video_list_for_preprocessing.json")
        
        # Filter to only available videos
        available_videos = [v for v in self.video_list if v["exists"]]
        
        if not available_videos:
            print("Warning: No videos are available locally!")
            print("Please download videos first or update video_dir path.")
        
        with open(output_path, 'w') as f:
            json.dump(available_videos, f, indent=2)
        
        print(f"Prepared {len(available_videos)} videos for preprocessing")
        print(f"Video list saved to: {output_path}")
        
        return output_path
    
    def create_benchmark(
        self,
        output_dir: Optional[str] = None,
        categories: Optional[List[QuestionCategory]] = None
    ) -> Dict[str, str]:
        """
        Create benchmark files for evaluation.
        
        Args:
            output_dir: Directory to save benchmark files
            categories: Question categories to include
        
        Returns:
            Dict with paths to created files
        """
        if not self.dataset:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        output_dir = output_dir or str(self.config.BENCHMARK_DIR)
        categories = categories or self.config.BENCHMARK_CATEGORIES
        
        # Filter dataset by categories if specified
        if categories:
            filtered_dataset = self.dataset.filter_by_category(categories)
        else:
            filtered_dataset = self.dataset
        
        return create_surveillance_vqa_benchmark(
            dataset=filtered_dataset,
            output_dir=output_dir
        )
    
    def get_queries_for_evaluation(
        self,
        num_queries: Optional[int] = None,
        category: Optional[QuestionCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of queries for evaluation.
        
        Args:
            num_queries: Maximum number of queries to return
            category: Filter by question category
        
        Returns:
            List of query dicts with video_id, question, answer
        """
        if not self.dataset:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        queries = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            
            # Filter by category if specified
            if category and sample["question_category"] != category.value:
                continue
            
            queries.append({
                "query_id": sample["sample_id"],
                "video_id": sample["video_id"],
                "video_path": sample["video_path"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "category": sample["question_category"],
                "anomaly_type": sample.get("anomaly_type")
            })
            
            if num_queries and len(queries) >= num_queries:
                break
        
        return queries
    
    def export_for_videorag(self, output_path: Optional[str] = None) -> str:
        """
        Export dataset in a format ready for VideoRAG pipeline.
        
        Creates:
        1. Video directory structure
        2. Annotations JSON
        3. Query benchmark file
        
        Args:
            output_path: Base path for exports
        
        Returns:
            Path to the main export directory
        """
        if not self.dataset:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        export_dir = Path(output_path) if output_path else self.output_dir / "videorag_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export annotations
        annotations_path = export_dir / "annotations.json"
        self.dataset.to_json(str(annotations_path))
        
        # Export VideoRAG format
        videorag_path = export_dir / "queries.json"
        self.dataset.to_videorag_format(str(videorag_path))
        
        # Export video info
        video_info_path = export_dir / "videos.json"
        with open(video_info_path, 'w') as f:
            json.dump(self.video_list, f, indent=2)
        
        # Create summary
        summary = {
            "dataset": "SurveillanceVQA-589K",
            "export_date": datetime.now().isoformat(),
            "num_videos": len(self.video_list),
            "num_qa_pairs": len(self.dataset),
            "available_videos": sum(1 for v in self.video_list if v["exists"]),
            "anomaly_types": self.dataset.get_unique_anomaly_types(),
            "category_distribution": self.dataset.get_category_distribution(),
            "files": {
                "annotations": str(annotations_path),
                "queries": str(videorag_path),
                "videos": str(video_info_path)
            }
        }
        
        summary_path = export_dir / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExport Summary:")
        print(f"  Videos: {summary['num_videos']}")
        print(f"  Q&A Pairs: {summary['num_qa_pairs']}")
        print(f"  Available Videos: {summary['available_videos']}")
        print(f"  Export Directory: {export_dir}")
        
        return str(export_dir)


def run_full_pipeline(
    video_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    skip_preprocessing: bool = False
) -> Dict[str, Any]:
    """
    Run the complete SurveillanceVQA → VideoRAG pipeline.
    
    Steps:
    1. Load dataset from HuggingFace
    2. Check video availability
    3. Prepare for preprocessing
    4. Run preprocessing (if videos available)
    5. Extract features
    6. Build index
    7. Create benchmark
    
    Args:
        video_dir: Directory containing video files
        split: Dataset split to use
        max_samples: Maximum samples to process
        skip_preprocessing: If True, assume preprocessing is done
    
    Returns:
        Dict with pipeline results and paths
    """
    from preprocess import process_all_videos
    from extract_features import process_all_chunks
    from index import VideoIndex
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "video_dir": video_dir,
            "split": split,
            "max_samples": max_samples
        }
    }
    
    # Initialize pipeline
    pipeline = SurveillanceVQAPipeline(video_dir=video_dir)
    
    # Step 1: Load dataset
    print("\n" + "="*50)
    print("Step 1: Loading SurveillanceVQA Dataset")
    print("="*50)
    
    dataset = pipeline.load_dataset(split=split, max_samples=max_samples)
    results["dataset"] = {
        "num_videos": len(pipeline.video_list),
        "num_qa_pairs": len(dataset)
    }
    
    # Step 2: Check video availability
    print("\n" + "="*50)
    print("Step 2: Checking Video Availability")
    print("="*50)
    
    availability = pipeline.check_video_availability()
    print(f"Available: {availability['available']}/{availability['total']}")
    results["video_availability"] = availability
    
    if availability["available"] == 0:
        print("\nERROR: No videos available. Please download videos first.")
        results["status"] = "failed_no_videos"
        return results
    
    # Step 3: Prepare for preprocessing
    print("\n" + "="*50)
    print("Step 3: Preparing for Preprocessing")
    print("="*50)
    
    video_list_path = pipeline.prepare_for_preprocessing()
    results["video_list_path"] = video_list_path
    
    if not skip_preprocessing:
        # Step 4: Run preprocessing
        print("\n" + "="*50)
        print("Step 4: Running Preprocessing (Chunking + Keyframes)")
        print("="*50)
        
        chunks_metadata = process_all_videos(video_dir)
        results["preprocessing"] = {
            "num_chunks": len(chunks_metadata) if chunks_metadata else 0
        }
        
        # Step 5: Extract features
        print("\n" + "="*50)
        print("Step 5: Extracting Features (CLIP + YOLO)")
        print("="*50)
        
        features_path = process_all_chunks()
        results["features_path"] = str(features_path)
        
        # Step 6: Build index
        print("\n" + "="*50)
        print("Step 6: Building FAISS Index")
        print("="*50)
        
        index = VideoIndex()
        index.build_index(str(features_path))
        index.save_index()
        results["index_built"] = True
    
    # Step 7: Create benchmark
    print("\n" + "="*50)
    print("Step 7: Creating Evaluation Benchmark")
    print("="*50)
    
    benchmark_paths = pipeline.create_benchmark()
    results["benchmark_paths"] = benchmark_paths
    
    # Export summary
    results["status"] = "completed"
    export_dir = pipeline.export_for_videorag()
    results["export_dir"] = export_dir
    
    print("\n" + "="*50)
    print("Pipeline Completed Successfully!")
    print("="*50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SurveillanceVQA Pipeline for VideoRAG"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(SurveillanceVQAConfig.VIDEO_DIR),
        help="Directory containing video files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing (assume already done)"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export dataset, don't run full pipeline"
    )
    
    args = parser.parse_args()
    
    if args.export_only:
        # Just load and export
        pipeline = SurveillanceVQAPipeline(video_dir=args.video_dir)
        pipeline.load_dataset(split=args.split, max_samples=args.max_samples)
        pipeline.export_for_videorag()
    else:
        # Run full pipeline
        results = run_full_pipeline(
            video_dir=args.video_dir,
            split=args.split,
            max_samples=args.max_samples,
            skip_preprocessing=args.skip_preprocessing
        )
        
        # Save results
        results_path = SurveillanceVQAConfig.OUTPUT_DIR / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_path}")
