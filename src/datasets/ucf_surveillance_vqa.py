"""
UCF-SurveillanceVQA Dataset Builder

This module creates a local dataset by matching UCF-Crime videos from sampledata/
with Q&A annotations from the SurveillanceVQA-589K dataset.

Usage:
    from src.datasets.ucf_surveillance_vqa import UCFSurveillanceVQADataset
    
    dataset = UCFSurveillanceVQADataset.build(
        video_dir="sampledata",
        output_dir="data/ucf_surveillance_vqa"
    )
    
    # Iterate over samples
    for sample in dataset:
        print(sample.video_path, sample.question, sample.answer)
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum
import logging

from huggingface_hub import hf_hub_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions in SurveillanceVQA dataset."""
    SUMMARY = "summary"                    # Full video description
    ANOMALY_DETECTION = "anomaly_detection"  # Does video contain anomaly?
    ANOMALY_TYPE = "anomaly_type"          # What type of anomaly?
    SUBJECT = "subject"                    # Who is the main subject?
    ACTION = "action"                      # What actions are happening?
    TEMPORAL = "temporal"                  # Time-related questions
    CAUSAL = "causal"                      # Why/cause questions
    ENVIRONMENT = "environment"            # Scene/setting questions
    UNKNOWN = "unknown"


@dataclass
class VideoSegment:
    """Represents a video segment with timestamps."""
    video_id: str
    segment_id: int
    start_time: float
    end_time: float
    category: str  # normal or anomaly type (Abuse, Arrest, etc.)
    caption: str


@dataclass  
class QAPair:
    """A single question-answer pair."""
    question: str
    answer: str
    question_type: QuestionType
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type.value
        }


@dataclass
class VideoQASample:
    """
    A sample linking a video segment to its Q&A pairs.
    
    This is the primary data structure for evaluation.
    """
    sample_id: str
    video_id: str              # e.g., "Abuse002_x264"
    video_path: str            # Full path to video file
    segment_id: int            # Segment number (1, 2, 3, ...)
    start_time: float          # Segment start in seconds
    end_time: float            # Segment end in seconds
    anomaly_type: str          # Category: "Abuse", "normal", etc.
    question: str
    answer: str
    question_type: str
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UCFSurveillanceVQADataset:
    """
    Dataset that matches UCF-Crime videos with SurveillanceVQA annotations.
    
    The SurveillanceVQA dataset provides:
    - Video segmentation with timestamps
    - Captions for each segment
    - Multiple Q&A pairs per segment
    
    This class downloads the annotations, matches them with local videos,
    and provides an iterable dataset for evaluation.
    """
    
    REPO_ID = "fei213/SurveillanceVQA-589K"
    QA_FILE = "github/7_qwen_max_caption_to_qa/qa_pairs/UCF_QA.zip"
    CATEGORY_FILE = "github/6_find_normal_abnormal/output/UCF_qwen_category.json"
    
    def __init__(
        self,
        samples: List[VideoQASample],
        video_dir: Path,
        metadata: Dict[str, Any] = None
    ):
        self.samples = samples
        self.video_dir = video_dir
        self.metadata = metadata or {}
        
        # Build indices for fast lookup
        self._by_video: Dict[str, List[VideoQASample]] = {}
        self._by_question_type: Dict[str, List[VideoQASample]] = {}
        
        for sample in samples:
            # Index by video
            if sample.video_id not in self._by_video:
                self._by_video[sample.video_id] = []
            self._by_video[sample.video_id].append(sample)
            
            # Index by question type
            if sample.question_type not in self._by_question_type:
                self._by_question_type[sample.question_type] = []
            self._by_question_type[sample.question_type].append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VideoQASample:
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[VideoQASample]:
        return iter(self.samples)
    
    @property
    def video_ids(self) -> List[str]:
        """Get all unique video IDs."""
        return list(self._by_video.keys())
    
    @property
    def question_types(self) -> List[str]:
        """Get all question types in dataset."""
        return list(self._by_question_type.keys())
    
    def get_by_video(self, video_id: str) -> List[VideoQASample]:
        """Get all samples for a specific video."""
        return self._by_video.get(video_id, [])
    
    def get_by_question_type(self, qtype: str) -> List[VideoQASample]:
        """Get all samples of a specific question type."""
        return self._by_question_type.get(qtype, [])
    
    def filter(
        self,
        video_ids: List[str] = None,
        question_types: List[str] = None,
        anomaly_types: List[str] = None
    ) -> 'UCFSurveillanceVQADataset':
        """Create a filtered subset of the dataset."""
        filtered = self.samples
        
        if video_ids:
            filtered = [s for s in filtered if s.video_id in video_ids]
        if question_types:
            filtered = [s for s in filtered if s.question_type in question_types]
        if anomaly_types:
            filtered = [s for s in filtered if s.anomaly_type in anomaly_types]
        
        return UCFSurveillanceVQADataset(filtered, self.video_dir, self.metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        anomaly_counts = {}
        qtype_counts = {}
        
        for sample in self.samples:
            # Count by anomaly type
            atype = sample.anomaly_type
            anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
            
            # Count by question type
            qtype = sample.question_type
            qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1
        
        return {
            "total_samples": len(self.samples),
            "unique_videos": len(self._by_video),
            "question_types": qtype_counts,
            "anomaly_types": anomaly_counts
        }
    
    def to_json(self, output_path: str) -> None:
        """Export dataset to JSON file."""
        data = {
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
            "samples": [s.to_dict() for s in self.samples]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.samples)} samples to {output_path}")
    
    @classmethod
    def from_json(cls, json_path: str, video_dir: str = None) -> 'UCFSurveillanceVQADataset':
        """Load dataset from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for s in data['samples']:
            samples.append(VideoQASample(
                sample_id=s['sample_id'],
                video_id=s['video_id'],
                video_path=s['video_path'],
                segment_id=s['segment_id'],
                start_time=s['start_time'],
                end_time=s['end_time'],
                anomaly_type=s['anomaly_type'],
                question=s['question'],
                answer=s['answer'],
                question_type=s['question_type'],
                caption=s.get('caption'),
                metadata=s.get('metadata', {})
            ))
        
        video_dir = Path(video_dir) if video_dir else Path(samples[0].video_path).parent
        return cls(samples, video_dir, data.get('metadata', {}))
    
    @classmethod
    def build(
        cls,
        video_dir: str = "sampledata",
        output_dir: str = "data/ucf_surveillance_vqa",
        cache_dir: str = None,
        force_download: bool = False
    ) -> 'UCFSurveillanceVQADataset':
        """
        Build the dataset by downloading annotations and matching with local videos.
        
        Args:
            video_dir: Directory containing UCF-Crime videos
            output_dir: Directory to store processed dataset
            cache_dir: Directory to cache downloaded files
            force_download: Force re-download of annotations
            
        Returns:
            UCFSurveillanceVQADataset instance
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cache_dir = Path(cache_dir) if cache_dir else output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing processed dataset
        dataset_json = output_dir / "dataset.json"
        if dataset_json.exists() and not force_download:
            logger.info(f"Loading existing dataset from {dataset_json}")
            return cls.from_json(str(dataset_json), str(video_dir))
        
        # Step 1: Get list of available videos
        logger.info(f"Scanning videos in {video_dir}")
        available_videos = cls._scan_videos(video_dir)
        logger.info(f"Found {len(available_videos)} videos")
        
        if not available_videos:
            raise ValueError(f"No videos found in {video_dir}")
        
        # Step 2: Download annotations
        logger.info("Downloading SurveillanceVQA annotations...")
        category_data, qa_data = cls._download_annotations(cache_dir, force_download)
        
        # Step 3: Match videos with annotations
        logger.info("Matching videos with Q&A annotations...")
        samples = cls._build_samples(available_videos, category_data, qa_data, video_dir)
        
        # Step 4: Create dataset
        metadata = {
            "source": "SurveillanceVQA-589K (UCF subset)",
            "video_dir": str(video_dir),
            "num_videos": len(available_videos),
            "build_version": "1.0"
        }
        
        dataset = cls(samples, video_dir, metadata)
        
        # Step 5: Save to disk
        dataset.to_json(str(dataset_json))
        
        return dataset
    
    @staticmethod
    def _scan_videos(video_dir: Path) -> Dict[str, Path]:
        """Scan directory for video files and return mapping of video_id -> path."""
        videos = {}
        
        for ext in ['*.mp4', '*.avi', '*.mkv']:
            for video_path in video_dir.glob(ext):
                # Extract video ID (remove extension)
                video_id = video_path.stem  # e.g., "Abuse002_x264"
                videos[video_id] = video_path
        
        return videos
    
    @classmethod
    def _download_annotations(
        cls, 
        cache_dir: Path, 
        force_download: bool
    ) -> Tuple[Dict, Dict]:
        """Download and parse annotation files."""
        
        # Download category file (has timestamps and captions)
        category_cache = cache_dir / "UCF_qwen_category.json"
        if not category_cache.exists() or force_download:
            logger.info("Downloading category annotations...")
            cat_path = hf_hub_download(
                cls.REPO_ID, 
                cls.CATEGORY_FILE, 
                repo_type="dataset"
            )
            shutil.copy(cat_path, category_cache)
        
        with open(category_cache, 'r', encoding='utf-8') as f:
            category_data = json.load(f)
        
        # Download and extract Q&A pairs
        qa_cache = cache_dir / "UCF_QA"
        if not qa_cache.exists() or force_download:
            logger.info("Downloading Q&A annotations...")
            qa_zip_path = hf_hub_download(
                cls.REPO_ID,
                cls.QA_FILE,
                repo_type="dataset"
            )
            
            # Extract ZIP
            with zipfile.ZipFile(qa_zip_path, 'r') as zf:
                zf.extractall(cache_dir)
        
        # Parse all Q&A files
        qa_data = {}
        qa_folder = cache_dir / "UCF_QA"
        
        for qa_file in qa_folder.glob("*.json"):
            # Parse filename: Abuse002_x264_1.json -> video_id=Abuse002_x264, segment=1
            parts = qa_file.stem.rsplit('_', 1)
            if len(parts) == 2:
                video_id, segment_str = parts
                try:
                    segment_id = int(segment_str)
                except ValueError:
                    continue
                
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_content = json.load(f)
                
                key = f"{video_id}_{segment_id}"
                qa_data[key] = {
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "content": qa_content
                }
        
        logger.info(f"Loaded {len(category_data)} video categories, {len(qa_data)} Q&A segments")
        return category_data, qa_data
    
    @classmethod
    def _classify_question(cls, question: str) -> QuestionType:
        """Classify a question into a type based on its content."""
        q_lower = question.lower()
        
        if any(kw in q_lower for kw in ['entire video', 'start to finish', 'walkthrough', 'throughout']):
            return QuestionType.SUMMARY
        elif any(kw in q_lower for kw in ['abnormal', 'anomaly', 'unusual', 'criminal', 'violent']):
            if 'type' in q_lower or 'what kind' in q_lower:
                return QuestionType.ANOMALY_TYPE
            return QuestionType.ANOMALY_DETECTION
        elif any(kw in q_lower for kw in ['who', 'person', 'individual', 'subject', 'main character']):
            return QuestionType.SUBJECT
        elif any(kw in q_lower for kw in ['what action', 'doing', 'activity', 'happen']):
            return QuestionType.ACTION
        elif any(kw in q_lower for kw in ['when', 'how long', 'time', 'before', 'after', 'duration']):
            return QuestionType.TEMPORAL
        elif any(kw in q_lower for kw in ['why', 'cause', 'reason', 'because', 'result']):
            return QuestionType.CAUSAL
        elif any(kw in q_lower for kw in ['environment', 'setting', 'location', 'scene', 'background']):
            return QuestionType.ENVIRONMENT
        else:
            return QuestionType.UNKNOWN
    
    @classmethod
    def _build_samples(
        cls,
        available_videos: Dict[str, Path],
        category_data: Dict,
        qa_data: Dict,
        video_dir: Path
    ) -> List[VideoQASample]:
        """Build VideoQASample objects by matching videos with annotations."""
        samples = []
        sample_counter = 0
        matched_videos = 0
        
        for video_id, video_path in tqdm(available_videos.items(), desc="Building samples"):
            # Check if we have category data for this video
            if video_id not in category_data:
                logger.debug(f"No category data for {video_id}")
                continue
            
            cat_info = category_data[video_id]
            timestamps = cat_info.get('timestamps', [])
            captions = cat_info.get('sentences', [])
            categories = cat_info.get('category', [])
            
            matched_videos += 1
            
            # Process each segment
            for seg_idx, (timestamp, caption, category) in enumerate(
                zip(timestamps, captions, categories), start=1
            ):
                start_time, end_time = timestamp
                
                # Look up Q&A pairs for this segment
                qa_key = f"{video_id}_{seg_idx}"
                
                if qa_key not in qa_data:
                    continue
                
                qa_content = qa_data[qa_key]["content"]
                
                # Extract Q&A pairs from different categories
                all_qa_pairs = []
                
                # Summary Q&A pairs
                if "summary_qa_pairs" in qa_content:
                    for qa in qa_content["summary_qa_pairs"]:
                        all_qa_pairs.append((qa["Q"], qa["A"], "summary"))
                
                # Anomaly Q&A pairs
                if "abnormal_qa_pairs" in qa_content:
                    for qa in qa_content["abnormal_qa_pairs"]:
                        all_qa_pairs.append((qa["Q"], qa["A"], "anomaly"))
                
                # Normal Q&A pairs  
                if "normal_qa_pairs" in qa_content:
                    for qa in qa_content["normal_qa_pairs"]:
                        all_qa_pairs.append((qa["Q"], qa["A"], "normal"))
                
                # Create samples for each Q&A pair
                for question, answer, qa_source in all_qa_pairs:
                    qtype = cls._classify_question(question)
                    
                    sample = VideoQASample(
                        sample_id=f"ucf_svqa_{sample_counter:06d}",
                        video_id=video_id,
                        video_path=str(video_path.absolute()),
                        segment_id=seg_idx,
                        start_time=start_time,
                        end_time=end_time,
                        anomaly_type=category,
                        question=question,
                        answer=answer,
                        question_type=qtype.value,
                        caption=caption,
                        metadata={"qa_source": qa_source}
                    )
                    samples.append(sample)
                    sample_counter += 1
        
        logger.info(f"Matched {matched_videos}/{len(available_videos)} videos")
        logger.info(f"Created {len(samples)} Q&A samples")
        
        return samples


def build_ucf_surveillance_vqa_dataset(
    video_dir: str = "sampledata",
    output_dir: str = "data/ucf_surveillance_vqa",
    force_rebuild: bool = False
) -> UCFSurveillanceVQADataset:
    """
    Convenience function to build the UCF-SurveillanceVQA dataset.
    
    Args:
        video_dir: Directory containing UCF-Crime videos
        output_dir: Directory to store processed dataset
        force_rebuild: Force rebuild even if cached dataset exists
        
    Returns:
        UCFSurveillanceVQADataset instance ready for evaluation
    """
    return UCFSurveillanceVQADataset.build(
        video_dir=video_dir,
        output_dir=output_dir,
        force_download=force_rebuild
    )


# Quick test when run directly
if __name__ == "__main__":
    print("Building UCF-SurveillanceVQA Dataset...")
    print("=" * 60)
    
    dataset = build_ucf_surveillance_vqa_dataset(
        video_dir="sampledata",
        output_dir="data/ucf_surveillance_vqa"
    )
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    stats = dataset.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"\nQuestion types:")
    for qtype, count in sorted(stats['question_types'].items()):
        print(f"  {qtype}: {count}")
    print(f"\nAnomaly types:")
    for atype, count in sorted(stats['anomaly_types'].items()):
        print(f"  {atype}: {count}")
    
    print("\n" + "=" * 60)
    print("Sample entries (first 3):")
    print("=" * 60)
    for i, sample in enumerate(dataset[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Video: {sample.video_id}")
        print(f"Segment: {sample.segment_id} ({sample.start_time:.1f}s - {sample.end_time:.1f}s)")
        print(f"Anomaly: {sample.anomaly_type}")
        print(f"Question: {sample.question[:100]}...")
        print(f"Answer: {sample.answer[:100]}...")
