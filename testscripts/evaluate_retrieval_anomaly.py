"""
VideoRAG Retrieval Evaluation - Anomaly Detection Focused

Uses samples from anomaly_detection question type (which have proper anomaly timestamps)
but generates retrieval-focused queries like "Find the traffic accident".

This combines:
- Correct ground truth timestamps from anomaly_detection samples
- Retrieval-appropriate queries (not generic VQA questions)

Usage:
    python testscripts/evaluate_retrieval_anomaly.py
    python testscripts/evaluate_retrieval_anomaly.py --max-samples 50 --top-k 5
"""

import os
import sys
import json
import csv
import random
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT / "src")

from tqdm import tqdm

# Helper to load modules with numeric prefixes
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load query module
_query_module = _load_module("query", PROJECT_ROOT / "src" / "05_query.py")
VideoRAGQueryProcessor = _query_module.VideoRAGQueryProcessor


# Retrieval-focused query templates
ANOMALY_QUERY_TEMPLATES = {
    "Abuse": [
        "Find the abuse incident",
        "Show me where abuse happens",
        "Locate the abusive behavior",
    ],
    "Arrest": [
        "Find the arrest scene",
        "Show me the police arrest",
        "Locate where someone gets arrested",
    ],
    "Arson": [
        "Find the arson attack",
        "Show me the fire being set",
        "Locate the arson incident",
    ],
    "Assault": [
        "Find the assault",
        "Show me the violent attack",
        "Locate the fighting scene",
        "Find where people are fighting",
    ],
    "Burglary": [
        "Find the burglary",
        "Show me the break-in",
        "Locate the theft scene",
        "Find where someone breaks in",
    ],
    "Explosion": [
        "Find the explosion",
        "Show me the blast",
        "Locate the explosion scene",
    ],
    "Fighting": [
        "Find the fight",
        "Show me people fighting",
        "Locate the violent confrontation",
    ],
    "RoadAccidents": [
        "Find the accident",
        "Show me the car crash",
        "Locate the traffic accident",
        "Find the collision",
    ],
    "Robbery": [
        "Find the robbery",
        "Show me the theft",
        "Locate the robbery scene",
        "Find where someone is robbed",
    ],
    "Shooting": [
        "Find the shooting",
        "Show me the gunfire",
        "Locate the shooting incident",
    ],
    "Shoplifting": [
        "Find the shoplifting",
        "Show me the theft from store",
        "Locate the shoplifter",
    ],
    "Stealing": [
        "Find the stealing",
        "Show me the theft",
        "Locate where something is stolen",
    ],
    "Vandalism": [
        "Find the vandalism",
        "Show me property damage",
        "Locate the destruction",
    ],
    "Traffic Accident": [
        "Find the traffic accident",
        "Show me the crash",
        "Locate the collision",
    ],
    "People Falling": [
        "Find where someone falls",
        "Show me the fall incident",
        "Locate the person falling",
    ],
    "Fire": [
        "Find the fire",
        "Show me the flames",
        "Locate the fire incident",
    ],
}


@dataclass
class RetrievalResult:
    """Result of a single retrieval evaluation."""
    sample_id: str
    video_id: str
    segment_id: int
    generated_query: str
    original_question: str
    original_answer: str
    question_type: str
    query_type: str
    gt_start: float
    gt_end: float
    gt_anomaly_type: str
    # Without reranker (CLIP only)
    no_rerank_retrieved_timestamps: str
    no_rerank_hit_at_1: bool
    no_rerank_hit_at_3: bool
    no_rerank_hit_at_5: bool
    no_rerank_temporal_iou: float
    no_rerank_best_overlap_pct: float
    no_rerank_mrr: float
    # With reranker
    rerank_retrieved_timestamps: str
    rerank_hit_at_1: bool
    rerank_hit_at_3: bool
    rerank_hit_at_5: bool
    rerank_temporal_iou: float
    rerank_best_overlap_pct: float
    rerank_mrr: float


def compute_temporal_iou(gt_start: float, gt_end: float, ret_start: float, ret_end: float) -> float:
    """Compute Intersection over Union for temporal segments."""
    if gt_end <= gt_start or ret_end <= ret_start:
        return 0.0
    
    inter_start = max(gt_start, ret_start)
    inter_end = min(gt_end, ret_end)
    
    if inter_start >= inter_end:
        return 0.0
    
    intersection = inter_end - inter_start
    union = (gt_end - gt_start) + (ret_end - ret_start) - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_overlap_pct(gt_start: float, gt_end: float, ret_start: float, ret_end: float) -> float:
    """Compute what percentage of ground truth is covered."""
    if gt_end <= gt_start:
        return 100.0 if ret_start <= gt_start <= ret_end else 0.0
    
    inter_start = max(gt_start, ret_start)
    inter_end = min(gt_end, ret_end)
    
    if inter_start >= inter_end:
        return 0.0
    
    return min(100.0, ((inter_end - inter_start) / (gt_end - gt_start)) * 100)


def has_overlap(gt_start: float, gt_end: float, ret_start: float, ret_end: float, tolerance: float = 5.0) -> bool:
    """Check if there's meaningful overlap with ±tolerance seconds.
    
    Args:
        gt_start, gt_end: Ground truth time window
        ret_start, ret_end: Retrieved chunk time window
        tolerance: Acceptable gap in seconds (default ±5 seconds)
    """
    # Expand ground truth window by tolerance
    gt_start_expanded = gt_start - tolerance
    gt_end_expanded = gt_end + tolerance
    
    # Check if retrieved chunk overlaps with expanded window
    if gt_end <= gt_start:
        # Point annotation - check if within tolerance
        return ret_start <= (gt_start + tolerance) and ret_end >= (gt_start - tolerance)
    
    # Check overlap with expanded ground truth
    inter_start = max(gt_start_expanded, ret_start)
    inter_end = min(gt_end_expanded, ret_end)
    
    if inter_start < inter_end:
        return True  # Any overlap with expanded window counts
    
    # Also check if retrieved is within tolerance distance from GT
    # Gap between chunks
    gap = max(0, max(ret_start - gt_end, gt_start - ret_end))
    return gap <= tolerance


def get_query_for_anomaly(anomaly_type: str) -> str:
    """Generate a retrieval query from anomaly type."""
    primary_type = anomaly_type.split(",")[0].strip()
    
    if primary_type in ANOMALY_QUERY_TEMPLATES:
        return random.choice(ANOMALY_QUERY_TEMPLATES[primary_type])
    
    # Fallback
    return f"Find the {primary_type.lower()} incident"


class RetrievalEvaluator:
    """Evaluator using anomaly_detection samples with generated queries."""
    
    def __init__(self, dataset_path: str, top_k: int = 5):
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.query_processor = None
        
    def initialize(self) -> bool:
        """Initialize VideoRAG."""
        try:
            print("Initializing VideoRAG...")
            self.query_processor = VideoRAGQueryProcessor()
            
            if not self.query_processor.load_index():
                print("ERROR: Could not load index")
                return False
            
            print(f"Loaded index with {len(self.query_processor.video_index.metadata)} chunks")
            return True
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_dataset(self) -> List[Dict]:
        """Load only detection_qa_pairs samples with valid anomaly types."""
        print(f"Loading dataset: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        
        # Build lookup for description answers by (video, segment) for richer context
        description_answers = {}
        classification_answers = {}
        for sample in data:
            key = (sample['video_id'], sample['segment_id'])
            if sample['question_type'] == 'description_qa_pairs':
                # Keep the first (longest) description
                if key not in description_answers or len(sample['answer']) > len(description_answers[key]):
                    description_answers[key] = sample['answer']
            elif sample['question_type'] == 'classification_qa_pairs':
                classification_answers[key] = sample['answer']
        
        # Filter to:
        # 1. Only detection_qa_pairs question type (anomaly detection questions)
        # 2. Non-normal anomaly types
        # 3. One sample per (video, segment) to avoid duplicates
        
        seen = set()
        filtered = []
        
        for sample in data:
            # Only detection_qa_pairs questions (the original label)
            if sample['question_type'] != 'detection_qa_pairs':
                continue
            
            # Skip normal
            if sample['anomaly_type'].lower() == 'normal':
                continue
            
            # Check if we have a query template for this anomaly
            primary_type = sample['anomaly_type'].split(",")[0].strip()
            if primary_type not in ANOMALY_QUERY_TEMPLATES:
                # Try to find a matching template
                found = False
                for key in ANOMALY_QUERY_TEMPLATES.keys():
                    if key.lower() in sample['anomaly_type'].lower():
                        found = True
                        break
                if not found:
                    continue
            
            # Unique per video+segment
            key = (sample['video_id'], sample['segment_id'])
            if key in seen:
                continue
            seen.add(key)
            
            # Enrich with description/classification answers
            sample['description_answer'] = description_answers.get(key, '')
            sample['classification_answer'] = classification_answers.get(key, sample['anomaly_type'])
            
            filtered.append(sample)
        
        print(f"Filtered to {len(filtered)} unique anomaly_detection segments")
        
        # Stats
        by_type = defaultdict(int)
        for s in filtered:
            by_type[s['anomaly_type'].split(",")[0].strip()] += 1
        print("\nBy anomaly type:")
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
        
        return filtered
    
    def retrieve_without_rerank(self, query: str, video_path: str) -> List[Dict]:
        """Retrieve chunks without reranking (CLIP only), filtered to same video."""
        results = self.query_processor.retrieve_similar_chunks(query, top_k=self.top_k * 3)
        
        video_name = Path(video_path).name
        same_video = [r for r in results if Path(r.get('video_path', '')).name == video_name]
        
        if same_video:
            return same_video[:self.top_k]
        return results[:self.top_k]
    
    def retrieve_with_rerank(self, query: str, video_path: str) -> List[Dict]:
        """Retrieve chunks with NEW 3-STAGE PIPELINE, filtered to same video.
        
        Stage 1: Top 15 using Visual embedding (CLIP cosine similarity)
        Stage 2: Top 10 using Textual embedding (60%) + BM25 (40%)
        Stage 3: Top 5 using Temporal context averaging (2 left + current + 2 right)
        """
        import re
        import numpy as np
        
        video_name = Path(video_path).name
        
        # Get query embedding
        query_embedding = self.query_processor.clip_extractor.encode_text(query)
        query_embedding_np = query_embedding.flatten()
        
        # Build chunk_id to metadata lookup
        chunk_id_to_metadata = {m['chunk_id']: m for m in self.query_processor.video_index.metadata}
        
        # ================================================================
        # STAGE 1: Top 15 using Visual Embedding
        # ================================================================
        visual_results = self.query_processor.video_index.search_visual(query_embedding, top_k=15)
        
        if not visual_results:
            return []
        
        # Enrich with metadata
        stage1_chunks = []
        for r in visual_results:
            chunk = r.copy()
            chunk['visual_score'] = r.get('visual_score', r.get('similarity_score', 0))
            # Get full metadata including embeddings
            if chunk['chunk_id'] in chunk_id_to_metadata:
                meta = chunk_id_to_metadata[chunk['chunk_id']]
                chunk['embedding'] = meta.get('embedding')
                chunk['caption_embedding'] = meta.get('caption_embedding')
            stage1_chunks.append(chunk)
        
        # ================================================================
        # STAGE 2: Top 10 using Textual + BM25
        # ================================================================
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        for chunk in stage1_chunks:
            # Textual score from caption embedding
            caption_emb = chunk.get('caption_embedding')
            if caption_emb and len(caption_emb) > 0:
                caption_emb = np.array(caption_emb, dtype='float32')
                caption_emb = caption_emb / (np.linalg.norm(caption_emb) + 1e-8)
                textual_score = float(np.dot(caption_emb, query_embedding_np))
            else:
                textual_score = 0.0
            
            # BM25 score from caption text
            caption = chunk.get('caption', '') or ''
            caption_tokens = list(set(re.findall(r'\w+', caption.lower())))
            
            # Simple BM25
            bm25_score = 0.0
            if caption_tokens:
                doc_tf = {}
                for term in caption_tokens:
                    doc_tf[term] = doc_tf.get(term, 0) + 1
                for term in query_tokens:
                    if term in doc_tf:
                        tf = doc_tf[term]
                        bm25_score += np.log(10) * (tf * 2.5) / (tf + 1.5)
            
            bm25_normalized = min(1.0, bm25_score / 3.0)
            
            # Combined: 60% Textual + 40% BM25
            chunk['textual_score'] = textual_score
            chunk['bm25_score'] = bm25_normalized
            chunk['stage2_score'] = 0.6 * textual_score + 0.4 * bm25_normalized
        
        # Sort by Stage 2 score and take top 10
        stage1_chunks.sort(key=lambda x: x['stage2_score'], reverse=True)
        stage2_chunks = stage1_chunks[:10]
        
        # ================================================================
        # STAGE 3: Top 5 using Temporal Context (Visual Averaging)
        # ================================================================
        def get_chunk_index(chunk_id):
            if '_chunk_' in chunk_id:
                try:
                    return int(chunk_id.split('_chunk_')[-1])
                except:
                    return -1
            return -1
        
        def get_video_name_from_chunk(chunk_id):
            if '_chunk_' in chunk_id:
                return chunk_id.rsplit('_chunk_', 1)[0]
            return chunk_id
        
        for chunk in stage2_chunks:
            chunk_id = chunk['chunk_id']
            chunk_video_name = get_video_name_from_chunk(chunk_id)
            current_idx = get_chunk_index(chunk_id)
            
            # Collect temporal embeddings
            embeddings = []
            for offset in range(-2, 3):  # -2, -1, 0, 1, 2
                neighbor_idx = current_idx + offset
                if neighbor_idx >= 0:
                    neighbor_id = f"{chunk_video_name}_chunk_{neighbor_idx:04d}"
                    if neighbor_id in chunk_id_to_metadata:
                        emb = chunk_id_to_metadata[neighbor_id].get('embedding')
                        if emb and len(emb) > 0:
                            embeddings.append(np.array(emb, dtype='float32'))
            
            # Compute temporal score
            if embeddings:
                temporal_avg = np.mean(embeddings, axis=0)
                temporal_avg = temporal_avg / (np.linalg.norm(temporal_avg) + 1e-8)
                temporal_score = float(np.dot(temporal_avg, query_embedding_np))
            else:
                temporal_score = chunk['visual_score']
            
            chunk['temporal_embeddings_count'] = len(embeddings)
            chunk['temporal_score'] = temporal_score
        
        # Sort by temporal score and take top 5
        stage2_chunks.sort(key=lambda x: x['temporal_score'], reverse=True)
        final_chunks = stage2_chunks[:self.top_k]
        
        # Filter to same video if possible
        same_video_chunks = [c for c in final_chunks if Path(c.get('video_path', '')).name == video_name]
        
        if same_video_chunks:
            return same_video_chunks[:self.top_k]
        return final_chunks[:self.top_k]
    
    def compute_metrics(self, chunks: List[Dict], gt_start: float, gt_end: float):
        """Compute retrieval metrics for a set of chunks."""
        hit_1 = hit_3 = hit_5 = False
        best_iou = best_overlap = reciprocal_rank = 0.0
        timestamps = []
        
        for i, chunk in enumerate(chunks):
            ret_start = chunk.get('start_time', 0)
            ret_end = chunk.get('end_time', 0)
            
            timestamps.append(f"{ret_start:.1f}-{ret_end:.1f}")
            
            is_hit = has_overlap(gt_start, gt_end, ret_start, ret_end)
            iou = compute_temporal_iou(gt_start, gt_end, ret_start, ret_end)
            overlap = compute_overlap_pct(gt_start, gt_end, ret_start, ret_end)
            
            best_iou = max(best_iou, iou)
            best_overlap = max(best_overlap, overlap)
            
            if is_hit:
                if i == 0: hit_1 = True
                if i < 3: hit_3 = True
                if i < 5: hit_5 = True
                if reciprocal_rank == 0:
                    reciprocal_rank = 1.0 / (i + 1)
        
        return {
            'timestamps': ", ".join(timestamps) if timestamps else "None",
            'hit_1': hit_1,
            'hit_3': hit_3,
            'hit_5': hit_5,
            'iou': best_iou,
            'overlap': best_overlap,
            'mrr': reciprocal_rank
        }
    
    def evaluate_sample(self, sample: Dict) -> RetrievalResult:
        """Evaluate one sample with both retrieval methods."""
        video_id = sample['video_id']
        video_path = sample['video_path']
        segment_id = sample['segment_id']
        gt_start = sample['start_time']
        gt_end = sample['end_time']
        anomaly_type = sample['anomaly_type']
        original_question = sample['question']
        question_type = sample['question_type']
        # Use description answer for richer context, fallback to classification
        original_answer = sample.get('description_answer', '') or sample.get('classification_answer', anomaly_type)
        
        # Generate retrieval query
        query = get_query_for_anomaly(anomaly_type)
        
        # Retrieve WITHOUT reranking (CLIP only)
        chunks_no_rerank = self.retrieve_without_rerank(query, video_path)
        metrics_no_rerank = self.compute_metrics(chunks_no_rerank, gt_start, gt_end)
        
        # Retrieve WITH reranking
        chunks_rerank = self.retrieve_with_rerank(query, video_path)
        metrics_rerank = self.compute_metrics(chunks_rerank, gt_start, gt_end)
        
        return RetrievalResult(
            sample_id=sample['sample_id'],
            video_id=video_id,
            segment_id=segment_id,
            generated_query=query,
            original_question=original_question[:150],
            original_answer=original_answer[:200] if original_answer else '',
            question_type=question_type,
            query_type=anomaly_type.split(",")[0].strip(),
            gt_start=gt_start,
            gt_end=gt_end,
            gt_anomaly_type=anomaly_type,
            # Without reranker
            no_rerank_retrieved_timestamps=metrics_no_rerank['timestamps'],
            no_rerank_hit_at_1=metrics_no_rerank['hit_1'],
            no_rerank_hit_at_3=metrics_no_rerank['hit_3'],
            no_rerank_hit_at_5=metrics_no_rerank['hit_5'],
            no_rerank_temporal_iou=metrics_no_rerank['iou'],
            no_rerank_best_overlap_pct=metrics_no_rerank['overlap'],
            no_rerank_mrr=metrics_no_rerank['mrr'],
            # With reranker
            rerank_retrieved_timestamps=metrics_rerank['timestamps'],
            rerank_hit_at_1=metrics_rerank['hit_1'],
            rerank_hit_at_3=metrics_rerank['hit_3'],
            rerank_hit_at_5=metrics_rerank['hit_5'],
            rerank_temporal_iou=metrics_rerank['iou'],
            rerank_best_overlap_pct=metrics_rerank['overlap'],
            rerank_mrr=metrics_rerank['mrr'],
        )
    
    def run(self, max_samples: int = None, output_path: str = None) -> List[RetrievalResult]:
        """Run evaluation."""
        if not self.initialize():
            return []
        
        samples = self.load_dataset()
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"\nEvaluating {len(samples)} samples...")
        results = []
        
        for sample in tqdm(samples, desc="Evaluating"):
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # Save
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(PROJECT_ROOT / "data" / f"retrieval_anomaly_eval_{timestamp}.csv")
        elif not os.path.isabs(output_path):
            output_path = str(PROJECT_ROOT / output_path)
        
        self.save_csv(results, output_path)
        self.print_summary(results)
        
        return results
    
    def save_csv(self, results: List[RetrievalResult], path: str):
        """Save to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        
        print(f"\nSaved to: {path}")
    
    def print_summary(self, results: List[RetrievalResult]):
        """Print summary comparing with and without reranker."""
        n = len(results)
        if n == 0:
            return
        
        # Without reranker metrics
        no_rerank_hit_1 = sum(r.no_rerank_hit_at_1 for r in results)
        no_rerank_hit_3 = sum(r.no_rerank_hit_at_3 for r in results)
        no_rerank_hit_5 = sum(r.no_rerank_hit_at_5 for r in results)
        no_rerank_mrr = sum(r.no_rerank_mrr for r in results) / n
        no_rerank_iou = sum(r.no_rerank_temporal_iou for r in results) / n
        no_rerank_overlap = sum(r.no_rerank_best_overlap_pct for r in results) / n
        
        # With reranker metrics
        rerank_hit_1 = sum(r.rerank_hit_at_1 for r in results)
        rerank_hit_3 = sum(r.rerank_hit_at_3 for r in results)
        rerank_hit_5 = sum(r.rerank_hit_at_5 for r in results)
        rerank_mrr = sum(r.rerank_mrr for r in results) / n
        rerank_iou = sum(r.rerank_temporal_iou for r in results) / n
        rerank_overlap = sum(r.rerank_best_overlap_pct for r in results) / n
        
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION - 3-STAGE PIPELINE vs CLIP-ONLY")
        print("="*80)
        print(f"Total samples: {n}")
        print()
        print("Pipeline: Visual(15) → Textual+BM25(10) → Temporal(5)")
        print()
        print(f"{'Metric':<20} {'CLIP Only':>15} {'3-Stage Pipeline':>17} {'Δ Change':>12}")
        print("-"*62)
        
        def fmt_pct(val, total):
            return f"{val}/{total} ({100*val/total:.1f}%)"
        
        def delta(v1, v2, total):
            d = 100*(v2 - v1)/total
            sign = "+" if d > 0 else ""
            return f"{sign}{d:.1f}%"
        
        print(f"{'Hit@1':<20} {fmt_pct(no_rerank_hit_1, n):>15} {fmt_pct(rerank_hit_1, n):>15} {delta(no_rerank_hit_1, rerank_hit_1, n):>12}")
        print(f"{'Hit@3':<20} {fmt_pct(no_rerank_hit_3, n):>15} {fmt_pct(rerank_hit_3, n):>15} {delta(no_rerank_hit_3, rerank_hit_3, n):>12}")
        print(f"{'Hit@5':<20} {fmt_pct(no_rerank_hit_5, n):>15} {fmt_pct(rerank_hit_5, n):>15} {delta(no_rerank_hit_5, rerank_hit_5, n):>12}")
        print(f"{'MRR':<20} {no_rerank_mrr:>15.3f} {rerank_mrr:>15.3f} {rerank_mrr - no_rerank_mrr:>+12.3f}")
        print(f"{'Avg Temporal IoU':<20} {no_rerank_iou:>15.3f} {rerank_iou:>15.3f} {rerank_iou - no_rerank_iou:>+12.3f}")
        print(f"{'Avg Overlap %':<20} {no_rerank_overlap:>14.1f}% {rerank_overlap:>14.1f}% {rerank_overlap - no_rerank_overlap:>+11.1f}%")
        
        print("\n--- Hit@3 by Anomaly Type (No Reranker vs With Reranker) ---")
        by_type_no = defaultdict(lambda: {'total': 0, 'hit': 0})
        by_type_yes = defaultdict(lambda: {'total': 0, 'hit': 0})
        for r in results:
            by_type_no[r.query_type]['total'] += 1
            by_type_yes[r.query_type]['total'] += 1
            if r.no_rerank_hit_at_3:
                by_type_no[r.query_type]['hit'] += 1
            if r.rerank_hit_at_3:
                by_type_yes[r.query_type]['hit'] += 1
        
        print(f"{'Anomaly Type':<20} {'No Reranker':>15} {'With Reranker':>15}")
        print("-"*50)
        for atype in sorted(by_type_no.keys(), key=lambda x: -by_type_no[x]['total']):
            no_stats = by_type_no[atype]
            yes_stats = by_type_yes[atype]
            no_rate = 100 * no_stats['hit'] / no_stats['total'] if no_stats['total'] > 0 else 0
            yes_rate = 100 * yes_stats['hit'] / yes_stats['total'] if yes_stats['total'] > 0 else 0
            print(f"  {atype:<18} {no_stats['hit']:>3}/{no_stats['total']:<3} ({no_rate:>5.1f}%)  {yes_stats['hit']:>3}/{yes_stats['total']:<3} ({yes_rate:>5.1f}%)")
        
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "ucf_surveillance_vqa_dataset.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(args.dataset, args.top_k)
    evaluator.run(args.max_samples, args.output)


if __name__ == "__main__":
    main()
