"""
Fast UCF-SurveillanceVQA Dataset Builder

Matches UCF-Crime videos from sampledata/ with Q&A annotations 
from the already extracted SurveillanceVQA data.

Usage:
    python -m src.datasets.build_ucf_vqa_dataset
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict


@dataclass
class VQASample:
    """A video Q&A sample for evaluation."""
    sample_id: str
    video_id: str
    video_path: str
    segment_id: int
    start_time: float
    end_time: float  
    anomaly_type: str
    question: str
    answer: str
    question_type: str


# Original QA category labels from UCF_QA files
# Normal segments use: summary_qa_pairs, generic_qa_pairs, temporal_qa_pairs, 
#                      spatial_qa_pairs, reasoning_qa_pairs, short_temporal_qa_pairs
# Anomaly segments use: detection_qa_pairs, classification_qa_pairs, subject_qa_pairs,
#                       description_qa_pairs, cause_qa_pairs, result_qa_pairs


def load_category_timestamps(category_file: str) -> Dict:
    """Load video timestamps and categories."""
    if not os.path.exists(category_file):
        return {}
    with open(category_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_matching_qa_files(video_id: str, qa_dir: str) -> List[str]:
    """Find all QA files matching a video ID."""
    if not os.path.exists(qa_dir):
        return []
    
    # Pattern: Abuse002_x264_1.json, Abuse002_x264_2.json, etc.
    matching = []
    for f in os.listdir(qa_dir):
        if f.startswith(video_id + "_") and f.endswith(".json"):
            matching.append(os.path.join(qa_dir, f))
    return sorted(matching)


def parse_qa_file(qa_file: str) -> List[Dict]:
    """Parse a QA JSON file and extract question-answer pairs."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    # Extract from different QA categories in the file
    for key in data:
        if isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict) and "Q" in item and "A" in item:
                    qa_pairs.append({
                        "question": item["Q"],
                        "answer": item["A"],
                        "category": key
                    })
    
    return qa_pairs


def build_dataset(
    video_dir: str,
    qa_dir: str = "temp_ucf_qa/UCF_QA",
    category_file: str = None,
    output_file: str = "data/ucf_surveillance_vqa_dataset.json"
) -> List[VQASample]:
    """
    Build the UCF-SurveillanceVQA dataset.
    
    Args:
        video_dir: Directory containing UCF-Crime videos (e.g., sampledata/)
        qa_dir: Directory containing extracted QA JSON files
        category_file: Optional path to UCF_qwen_category.json for timestamps
        output_file: Where to save the built dataset
    
    Returns:
        List of VQASample objects
    """
    # Find all videos
    video_files = {}
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith('.mp4'):
                video_id = f.replace('.mp4', '')
                video_files[video_id] = os.path.join(root, f)
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    # Load timestamps if available
    timestamps_data = {}
    if category_file and os.path.exists(category_file):
        timestamps_data = load_category_timestamps(category_file)
        print(f"Loaded timestamps for {len(timestamps_data)} videos")
    
    # Build dataset
    samples = []
    matched_videos = 0
    total_qa_pairs = 0
    
    for video_id, video_path in sorted(video_files.items()):
        qa_files = find_matching_qa_files(video_id, qa_dir)
        
        if not qa_files:
            continue
        
        matched_videos += 1
        
        # Get timestamps for this video
        video_timestamps = timestamps_data.get(video_id, {})
        timestamps = video_timestamps.get("timestamps", [])
        categories = video_timestamps.get("category", [])
        
        for qa_file in qa_files:
            # Extract segment ID from filename (e.g., "Abuse002_x264_1.json" -> 1)
            filename = os.path.basename(qa_file)
            segment_id = int(filename.replace(video_id + "_", "").replace(".json", ""))
            
            # Get timestamp for this segment
            start_time = 0.0
            end_time = 0.0
            anomaly_type = "unknown"
            
            if timestamps and segment_id <= len(timestamps):
                ts = timestamps[segment_id - 1]
                start_time = ts[0]
                end_time = ts[1]
            
            if categories and segment_id <= len(categories):
                anomaly_type = categories[segment_id - 1]
            else:
                # Infer from video name
                for cat in ["Abuse", "Arrest", "Arson", "Assault", "Burglary", 
                           "Explosion", "Fighting", "RoadAccidents", "Robbery", 
                           "Shooting", "Shoplifting", "Stealing", "Vandalism"]:
                    if video_id.startswith(cat):
                        anomaly_type = cat
                        break
                else:
                    if "Normal" in video_id:
                        anomaly_type = "normal"
            
            # Parse QA pairs - category comes from the original JSON keys
            qa_pairs = parse_qa_file(qa_file)
            
            for i, qa in enumerate(qa_pairs):
                sample = VQASample(
                    sample_id=f"{video_id}_seg{segment_id}_q{i+1}",
                    video_id=video_id,
                    video_path=os.path.abspath(video_path),
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    anomaly_type=anomaly_type,
                    question=qa["question"],
                    answer=qa["answer"],
                    question_type=qa["category"]  # Use original label from UCF_QA
                )
                samples.append(sample)
                total_qa_pairs += 1
    
    print(f"\nDataset Statistics:")
    print(f"  Videos matched: {matched_videos}/{len(video_files)}")
    print(f"  Total QA pairs: {total_qa_pairs}")
    print(f"  Avg QA per video: {total_qa_pairs/matched_videos:.1f}" if matched_videos > 0 else "")
    
    # Count by anomaly type
    by_type = defaultdict(int)
    for s in samples:
        by_type[s.anomaly_type] += 1
    print(f"\n  By anomaly type:")
    for t, count in sorted(by_type.items()):
        print(f"    {t}: {count}")
    
    # Count by question type
    by_qtype = defaultdict(int)
    for s in samples:
        by_qtype[s.question_type] += 1
    print(f"\n  By question type:")
    for t, count in sorted(by_qtype.items()):
        print(f"    {t}: {count}")
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in samples], f, indent=2)
    print(f"\nDataset saved to: {output_file}")
    
    return samples


class UCFSurveillanceVQADataset:
    """
    PyTorch-style dataset for UCF-SurveillanceVQA.
    
    Usage:
        dataset = UCFSurveillanceVQADataset.load("data/ucf_surveillance_vqa_dataset.json")
        for sample in dataset:
            print(sample.question)
    """
    
    def __init__(self, samples: List[VQASample]):
        self.samples = samples
    
    @classmethod
    def load(cls, path: str) -> "UCFSurveillanceVQADataset":
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = [VQASample(**s) for s in data]
        return cls(samples)
    
    @classmethod
    def build(
        cls,
        video_dir: str = "sampledata",
        qa_dir: str = "temp_ucf_qa/UCF_QA",
        output_file: str = "data/ucf_surveillance_vqa_dataset.json"
    ) -> "UCFSurveillanceVQADataset":
        """Build and return dataset."""
        samples = build_dataset(video_dir, qa_dir, output_file=output_file)
        return cls(samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VQASample:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)
    
    def filter_by_anomaly(self, anomaly_type: str) -> "UCFSurveillanceVQADataset":
        """Filter samples by anomaly type."""
        filtered = [s for s in self.samples if s.anomaly_type == anomaly_type]
        return UCFSurveillanceVQADataset(filtered)
    
    def filter_by_question_type(self, q_type: str) -> "UCFSurveillanceVQADataset":
        """Filter samples by question type."""
        filtered = [s for s in self.samples if s.question_type == q_type]
        return UCFSurveillanceVQADataset(filtered)
    
    def filter_by_video(self, video_id: str) -> "UCFSurveillanceVQADataset":
        """Get all samples for a specific video."""
        filtered = [s for s in self.samples if s.video_id == video_id]
        return UCFSurveillanceVQADataset(filtered)
    
    def get_unique_videos(self) -> List[str]:
        """Get list of unique video IDs."""
        return list(set(s.video_id for s in self.samples))
    
    def get_video_segments(self, video_id: str) -> List[Dict]:
        """Get all segments for a video with their timestamps."""
        segments = {}
        for s in self.samples:
            if s.video_id == video_id and s.segment_id not in segments:
                segments[s.segment_id] = {
                    "segment_id": s.segment_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "anomaly_type": s.anomaly_type
                }
        return sorted(segments.values(), key=lambda x: x["segment_id"])
    
    def to_evaluation_format(self) -> List[Dict]:
        """Convert to format suitable for VQA evaluation."""
        return [
            {
                "id": s.sample_id,
                "video_path": s.video_path,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "question": s.question,
                "ground_truth": s.answer,
                "anomaly_type": s.anomaly_type,
                "question_type": s.question_type
            }
            for s in self.samples
        ]


if __name__ == "__main__":
    # Build the dataset
    dataset = UCFSurveillanceVQADataset.build(
        video_dir="sampledata",
        qa_dir="temp_ucf_qa/UCF_QA",
        output_file="data/ucf_surveillance_vqa_dataset.json"
    )
    
    # Show sample
    print("\n" + "="*60)
    print("Sample entries:")
    print("="*60)
    for i, sample in enumerate(dataset[:3]):
        print(f"\n[{i+1}] {sample.video_id} - Segment {sample.segment_id}")
        print(f"    Time: {sample.start_time:.1f}s - {sample.end_time:.1f}s")
        print(f"    Type: {sample.anomaly_type}")
        print(f"    Q: {sample.question[:100]}...")
        print(f"    A: {sample.answer[:100]}...")
