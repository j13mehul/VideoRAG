"""
SurveillanceVQA-589K Dataset Loader for VideoRAG Pipeline.

This module provides a PyTorch-style Dataset class for loading and parsing
the SurveillanceVQA-589K dataset from Hugging Face, designed to be compatible
with the existing VideoRAG pipeline that processes surveillance/CCTV videos.

Dataset: https://huggingface.co/datasets/fei213/SurveillanceVQA-589K

Features:
- Parses multi-category Q&A pairs (anomaly detection, temporal, causal, etc.)
- Maintains compatibility with existing UCF-Crime loader interface
- Supports both inference and evaluation workflows
- Provides flattened Q&A format for VideoRAG retrieval pipeline
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class QuestionCategory(Enum):
    """Categories of questions in SurveillanceVQA dataset."""
    ANOMALY_DETECTION = "anomaly_detection"     # "Does this video contain..."
    ANOMALY_TYPE = "anomaly_type"               # "What type of abnormal event..."
    SUBJECT_IDENTIFICATION = "subject_id"       # "Who/what plays a key role..."
    EVENT_DESCRIPTION = "event_description"     # "What is happening in..."
    TEMPORAL = "temporal"                       # Time-related questions
    CAUSAL = "causal"                          # Why/cause questions
    ACTION = "action"                          # Action recognition questions
    OBJECT = "object"                          # Object identification
    UNKNOWN = "unknown"


@dataclass
class QAPair:
    """Represents a single Question-Answer pair."""
    question: str
    answer: str
    category: QuestionCategory
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category.value
        }


@dataclass
class VideoSample:
    """
    Represents a single video sample with all associated Q&A pairs.
    
    Attributes:
        video_id: Unique identifier for the video
        video_path: Path to the video file
        anomaly_type: Type of anomaly detected (e.g., "Assault", "Burglary")
        qa_pairs: List of all Q&A pairs for this video
        metadata: Additional metadata (source dataset, split, etc.)
    """
    video_id: str
    video_path: str
    anomaly_type: Optional[str] = None
    qa_pairs: List[QAPair] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "anomaly_type": self.anomaly_type,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata
        }
    
    def get_questions_by_category(self, category: QuestionCategory) -> List[QAPair]:
        """Get all Q&A pairs of a specific category."""
        return [qa for qa in self.qa_pairs if qa.category == category]
    
    def get_all_questions(self) -> List[str]:
        """Get all questions as a list of strings."""
        return [qa.question for qa in self.qa_pairs]
    
    def get_primary_question(self) -> Optional[str]:
        """Get the most descriptive question (event description if available)."""
        desc_pairs = self.get_questions_by_category(QuestionCategory.EVENT_DESCRIPTION)
        if desc_pairs:
            return desc_pairs[0].question
        return self.qa_pairs[0].question if self.qa_pairs else None


@dataclass
class FlattenedQASample:
    """
    Flattened representation for VideoRAG pipeline compatibility.
    Each sample represents one question about one video.
    
    This is the format expected by the VideoRAG pipeline for:
    - Query processing
    - Retrieval evaluation
    - Benchmark computation
    """
    sample_id: str
    video_id: str
    video_path: str
    question: str
    answer: str
    question_category: str
    anomaly_type: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "question": self.question,
            "answer": self.answer,
            "question_category": self.question_category,
            "anomaly_type": self.anomaly_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata
        }


class SurveillanceVQAParser:
    """
    Parser for SurveillanceVQA-589K dataset format.
    
    The dataset has columns with Q&A pairs in JSON format:
    - Column 1: Anomaly detection questions [{"Q": "...", "A": "Yes/No"}]
    - Column 2: Anomaly type questions [{"Q": "...", "A": "Assault/Burglary/..."}]
    - Column 3: Subject identification [{"Q": "...", "A": "..."}]
    - Column 4: Event description [{"Q": "...", "A": "detailed description..."}]
    - (Additional columns may exist)
    """
    
    # Keywords for categorizing questions
    CATEGORY_KEYWORDS = {
        QuestionCategory.ANOMALY_DETECTION: [
            "contain", "violent", "criminal", "abnormal", "unusual"
        ],
        QuestionCategory.ANOMALY_TYPE: [
            "type of abnormal", "type of anomaly", "what type"
        ],
        QuestionCategory.SUBJECT_IDENTIFICATION: [
            "who is", "who or what", "main person", "main subject", 
            "key role", "central to", "involved"
        ],
        QuestionCategory.EVENT_DESCRIPTION: [
            "what is happening", "describe", "can you describe", 
            "environment", "actions taking place"
        ],
        QuestionCategory.TEMPORAL: [
            "when", "how long", "duration", "before", "after", "time"
        ],
        QuestionCategory.CAUSAL: [
            "why", "cause", "reason", "led to", "result"
        ],
        QuestionCategory.ACTION: [
            "doing", "action", "activity", "behavior"
        ],
        QuestionCategory.OBJECT: [
            "what object", "which object", "identify the object"
        ]
    }
    
    @classmethod
    def categorize_question(cls, question: str) -> QuestionCategory:
        """Determine the category of a question based on keywords."""
        question_lower = question.lower()
        
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return category
        
        return QuestionCategory.UNKNOWN
    
    @classmethod
    def parse_qa_column(cls, qa_data: Any) -> List[QAPair]:
        """
        Parse a Q&A column which can be:
        - A list of dicts: [{"Q": "...", "A": "..."}]
        - A single dict: {"Q": "...", "A": "..."}
        - A JSON string
        - None
        """
        if qa_data is None:
            return []
        
        # Handle string input (JSON)
        if isinstance(qa_data, str):
            try:
                qa_data = json.loads(qa_data)
            except json.JSONDecodeError:
                return []
        
        # Ensure it's a list
        if isinstance(qa_data, dict):
            qa_data = [qa_data]
        
        qa_pairs = []
        for item in qa_data:
            if isinstance(item, dict) and "Q" in item and "A" in item:
                question = item["Q"]
                answer = item["A"]
                
                # Handle answer that might be a list
                if isinstance(answer, list):
                    answer = ", ".join(str(a) for a in answer)
                else:
                    answer = str(answer)
                
                category = cls.categorize_question(question)
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    category=category
                ))
        
        return qa_pairs
    
    @classmethod
    def extract_anomaly_type(cls, qa_pairs: List[QAPair]) -> Optional[str]:
        """Extract the anomaly type from Q&A pairs if available."""
        for qa in qa_pairs:
            if qa.category == QuestionCategory.ANOMALY_TYPE:
                return qa.answer
        return None


class SurveillanceVQADataset(Dataset):
    """
    PyTorch Dataset for SurveillanceVQA-589K.
    
    Supports two modes:
    1. Grouped mode (flatten=False): Returns VideoSample with all Q&A pairs
    2. Flattened mode (flatten=True): Returns individual FlattenedQASample
    
    Usage:
        # Load from Hugging Face
        dataset = SurveillanceVQADataset.from_huggingface(
            video_dir="./videos",
            split="train"
        )
        
        # Load from local JSON
        dataset = SurveillanceVQADataset.from_json(
            json_path="./annotations.json",
            video_dir="./videos"
        )
        
        # Access samples
        sample = dataset[0]  # FlattenedQASample (default)
    """
    
    def __init__(
        self,
        samples: List[VideoSample],
        video_dir: str,
        flatten: bool = True,
        filter_categories: Optional[List[QuestionCategory]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            samples: List of VideoSample objects
            video_dir: Root directory containing video files
            flatten: If True, returns individual Q&A pairs; else grouped by video
            filter_categories: Only include questions of these categories
            transform: Optional transform to apply to samples
        """
        self.video_dir = Path(video_dir)
        self.flatten = flatten
        self.filter_categories = filter_categories
        self.transform = transform
        self.samples = samples
        
        # Build flattened index if needed
        self._flattened_samples: List[FlattenedQASample] = []
        if flatten:
            self._build_flattened_index()
    
    def _build_flattened_index(self) -> None:
        """Create flattened Q&A samples for efficient indexing."""
        sample_idx = 0
        for video_sample in self.samples:
            for qa in video_sample.qa_pairs:
                # Apply category filter if specified
                if self.filter_categories and qa.category not in self.filter_categories:
                    continue
                
                flat_sample = FlattenedQASample(
                    sample_id=f"{video_sample.video_id}_{sample_idx}",
                    video_id=video_sample.video_id,
                    video_path=video_sample.video_path,
                    question=qa.question,
                    answer=qa.answer,
                    question_category=qa.category.value,
                    anomaly_type=video_sample.anomaly_type,
                    start_time=video_sample.start_time,
                    end_time=video_sample.end_time,
                    metadata=video_sample.metadata
                )
                self._flattened_samples.append(flat_sample)
                sample_idx += 1
    
    def __len__(self) -> int:
        if self.flatten:
            return len(self._flattened_samples)
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Returns dict with keys compatible with VideoRAG pipeline:
        - video_id, video_path, question, answer, question_category
        - anomaly_type, start_time, end_time (if available)
        """
        if self.flatten:
            sample = self._flattened_samples[idx]
            result = sample.to_dict()
        else:
            sample = self.samples[idx]
            result = sample.to_dict()
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_video_sample(self, video_id: str) -> Optional[VideoSample]:
        """Get the full VideoSample for a given video_id."""
        for sample in self.samples:
            if sample.video_id == video_id:
                return sample
        return None
    
    def get_unique_videos(self) -> List[str]:
        """Get list of unique video IDs."""
        return [sample.video_id for sample in self.samples]
    
    def get_unique_anomaly_types(self) -> List[str]:
        """Get list of unique anomaly types in the dataset."""
        types = set()
        for sample in self.samples:
            if sample.anomaly_type:
                types.add(sample.anomaly_type)
        return sorted(list(types))
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of question categories."""
        distribution = {}
        for sample in self._flattened_samples if self.flatten else self.samples:
            if self.flatten:
                cat = sample.question_category
            else:
                for qa in sample.qa_pairs:
                    cat = qa.category.value
                    distribution[cat] = distribution.get(cat, 0) + 1
                continue
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution
    
    def filter_by_anomaly_type(self, anomaly_types: List[str]) -> 'SurveillanceVQADataset':
        """Create a new dataset filtered by anomaly types."""
        filtered_samples = [
            s for s in self.samples 
            if s.anomaly_type in anomaly_types
        ]
        return SurveillanceVQADataset(
            samples=filtered_samples,
            video_dir=str(self.video_dir),
            flatten=self.flatten,
            filter_categories=self.filter_categories,
            transform=self.transform
        )
    
    def filter_by_category(self, categories: List[QuestionCategory]) -> 'SurveillanceVQADataset':
        """Create a new dataset filtered by question categories."""
        return SurveillanceVQADataset(
            samples=self.samples,
            video_dir=str(self.video_dir),
            flatten=self.flatten,
            filter_categories=categories,
            transform=self.transform
        )
    
    @classmethod
    def from_huggingface(
        cls,
        video_dir: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        flatten: bool = True,
        **kwargs
    ) -> 'SurveillanceVQADataset':
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            video_dir: Local directory containing video files
            split: Dataset split ("train" or "test")
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of video samples to load
            flatten: Whether to flatten Q&A pairs
            **kwargs: Additional arguments passed to Dataset constructor
        
        Returns:
            SurveillanceVQADataset instance
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the 'datasets' library: pip install datasets"
            )
        
        print(f"Loading SurveillanceVQA-589K dataset from Hugging Face (split: {split})...")
        
        hf_dataset = load_dataset(
            "fei213/SurveillanceVQA-589K",
            split=split,
            cache_dir=cache_dir
        )
        
        samples = cls._parse_hf_dataset(hf_dataset, video_dir, max_samples)
        
        return cls(
            samples=samples,
            video_dir=video_dir,
            flatten=flatten,
            **kwargs
        )
    
    @classmethod
    def _parse_hf_dataset(
        cls,
        hf_dataset,
        video_dir: str,
        max_samples: Optional[int] = None
    ) -> List[VideoSample]:
        """Parse Hugging Face dataset into VideoSample objects."""
        samples = []
        video_dir = Path(video_dir)
        
        # Get column names
        columns = hf_dataset.column_names
        print(f"Dataset columns: {columns}")
        
        # Identify video column and Q&A columns
        video_col = None
        qa_columns = []
        
        for col in columns:
            col_lower = col.lower()
            if 'video' in col_lower and ('id' in col_lower or 'path' in col_lower or 'name' in col_lower):
                video_col = col
            elif col_lower not in ['video', 'video_id', 'video_path']:
                # Assume other columns contain Q&A data
                qa_columns.append(col)
        
        # If no explicit video column, try to infer from data structure
        if not video_col and 'video' in columns:
            video_col = 'video'
        
        total = min(len(hf_dataset), max_samples) if max_samples else len(hf_dataset)
        
        for idx in tqdm(range(total), desc="Parsing dataset"):
            row = hf_dataset[idx]
            
            # Extract video ID/path
            if video_col and row.get(video_col):
                video_id = str(row[video_col])
            else:
                video_id = f"video_{idx:06d}"
            
            # Build video path
            video_path = cls._resolve_video_path(video_id, video_dir)
            
            # Parse all Q&A columns
            all_qa_pairs = []
            for col in qa_columns:
                if col in row and row[col] is not None:
                    qa_pairs = SurveillanceVQAParser.parse_qa_column(row[col])
                    all_qa_pairs.extend(qa_pairs)
            
            # Skip samples with no Q&A pairs
            if not all_qa_pairs:
                continue
            
            # Extract anomaly type
            anomaly_type = SurveillanceVQAParser.extract_anomaly_type(all_qa_pairs)
            
            sample = VideoSample(
                video_id=video_id,
                video_path=str(video_path),
                anomaly_type=anomaly_type,
                qa_pairs=all_qa_pairs,
                metadata={
                    "source": "SurveillanceVQA-589K",
                    "split": hf_dataset.split if hasattr(hf_dataset, 'split') else "unknown",
                    "index": idx
                }
            )
            samples.append(sample)
        
        print(f"Parsed {len(samples)} video samples with {sum(len(s.qa_pairs) for s in samples)} Q&A pairs")
        return samples
    
    @classmethod
    def _resolve_video_path(cls, video_id: str, video_dir: Path) -> Path:
        """Resolve video path, handling various naming conventions."""
        # Try common extensions
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # Clean video_id (remove extension if present)
        video_id_clean = video_id
        for ext in extensions:
            if video_id.lower().endswith(ext):
                video_id_clean = video_id[:-len(ext)]
                break
        
        # Try different path patterns
        patterns = [
            video_dir / f"{video_id_clean}.mp4",
            video_dir / f"{video_id}",
            video_dir / video_id_clean / f"{video_id_clean}.mp4",
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        # Default: return the expected path (may not exist yet)
        return video_dir / f"{video_id_clean}.mp4"
    
    @classmethod
    def from_json(
        cls,
        json_path: str,
        video_dir: str,
        flatten: bool = True,
        **kwargs
    ) -> 'SurveillanceVQADataset':
        """
        Load dataset from local JSON file.
        
        Expected JSON format:
        [
            {
                "video_id": "video_001",
                "video_path": "path/to/video.mp4",  # optional
                "anomaly_type": "Assault",  # optional
                "qa_pairs": [
                    {"Q": "Question?", "A": "Answer"},
                    ...
                ],
                "start_time": 0.0,  # optional
                "end_time": 10.0    # optional
            },
            ...
        ]
        """
        print(f"Loading dataset from {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        video_dir_path = Path(video_dir)
        
        for item in tqdm(data, desc="Parsing JSON"):
            video_id = item.get("video_id", "")
            video_path = item.get("video_path", "")
            
            if not video_path:
                video_path = str(cls._resolve_video_path(video_id, video_dir_path))
            
            # Parse Q&A pairs
            qa_pairs = SurveillanceVQAParser.parse_qa_column(
                item.get("qa_pairs", [])
            )
            
            anomaly_type = item.get("anomaly_type") or \
                          SurveillanceVQAParser.extract_anomaly_type(qa_pairs)
            
            sample = VideoSample(
                video_id=video_id,
                video_path=video_path,
                anomaly_type=anomaly_type,
                qa_pairs=qa_pairs,
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                metadata=item.get("metadata", {})
            )
            samples.append(sample)
        
        return cls(
            samples=samples,
            video_dir=video_dir,
            flatten=flatten,
            **kwargs
        )
    
    def to_json(self, output_path: str) -> None:
        """Export dataset to JSON format."""
        data = [sample.to_dict() for sample in self.samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(data)} samples to {output_path}")
    
    def to_videorag_format(self, output_path: str) -> None:
        """
        Export to VideoRAG pipeline compatible format.
        
        Creates a JSON file with flattened Q&A samples suitable for:
        - Query-based evaluation
        - Retrieval benchmarking
        """
        if not self.flatten:
            # Temporarily flatten for export
            temp_dataset = SurveillanceVQADataset(
                samples=self.samples,
                video_dir=str(self.video_dir),
                flatten=True,
                filter_categories=self.filter_categories
            )
            flattened = temp_dataset._flattened_samples
        else:
            flattened = self._flattened_samples
        
        data = [sample.to_dict() for sample in flattened]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(data)} Q&A samples to {output_path}")
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, List]:
        """Custom collate function for batching."""
        if not batch:
            return {}
        
        keys = batch[0].keys()
        return {key: [item[key] for item in batch] for key in keys}


def create_surveillance_vqa_benchmark(
    dataset: SurveillanceVQADataset,
    output_dir: str,
    include_categories: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create benchmark files for VideoRAG evaluation.
    
    Args:
        dataset: SurveillanceVQADataset instance
        output_dir: Directory to save benchmark files
        include_categories: Question categories to include
    
    Returns:
        Dict with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export full dataset
    full_path = output_dir / "surveillance_vqa_full.json"
    dataset.to_videorag_format(str(full_path))
    
    # Export by category
    category_paths = {}
    for category in QuestionCategory:
        filtered = dataset.filter_by_category([category])
        if len(filtered) > 0:
            cat_path = output_dir / f"surveillance_vqa_{category.value}.json"
            filtered.to_videorag_format(str(cat_path))
            category_paths[category.value] = str(cat_path)
    
    # Export video list
    video_list_path = output_dir / "video_list.json"
    video_info = []
    for sample in dataset.samples:
        video_info.append({
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "anomaly_type": sample.anomaly_type,
            "num_questions": len(sample.qa_pairs)
        })
    
    with open(video_list_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=2)
    
    return {
        "full": str(full_path),
        "video_list": str(video_list_path),
        **category_paths
    }


# Convenience function for quick loading
def load_surveillance_vqa(
    video_dir: str,
    source: str = "huggingface",
    split: str = "train",
    json_path: Optional[str] = None,
    **kwargs
) -> SurveillanceVQADataset:
    """
    Convenience function to load SurveillanceVQA dataset.
    
    Args:
        video_dir: Directory containing video files
        source: "huggingface" or "json"
        split: Dataset split (for HuggingFace)
        json_path: Path to JSON file (for local loading)
        **kwargs: Additional arguments
    
    Returns:
        SurveillanceVQADataset instance
    """
    if source == "huggingface":
        return SurveillanceVQADataset.from_huggingface(
            video_dir=video_dir,
            split=split,
            **kwargs
        )
    elif source == "json":
        if not json_path:
            raise ValueError("json_path required when source='json'")
        return SurveillanceVQADataset.from_json(
            json_path=json_path,
            video_dir=video_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'huggingface' or 'json'")


if __name__ == "__main__":
    # Example usage and testing
    print("SurveillanceVQA Dataset Loader")
    print("=" * 50)
    
    # Test with sample data
    sample_data = [
        {
            "video_id": "test_video_001",
            "qa_pairs": [
                {"Q": "Does this video contain any violent activities?", "A": "Yes"},
                {"Q": "What type of abnormal event is present?", "A": "Assault"},
                {"Q": "Who is involved in the event?", "A": "Two people fighting"},
                {"Q": "What is happening in the video?", "A": "A detailed description..."}
            ],
            "anomaly_type": "Assault"
        }
    ]
    
    # Create temporary JSON for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name
    
    # Test loading
    dataset = SurveillanceVQADataset.from_json(
        json_path=temp_path,
        video_dir="./videos"
    )
    
    print(f"\nLoaded {len(dataset)} samples")
    print(f"First sample: {dataset[0]}")
    print(f"\nCategory distribution: {dataset.get_category_distribution()}")
    
    # Cleanup
    os.unlink(temp_path)
