"""
VideoRAG Response Quality Evaluation

Evaluates AI Analysis responses for description_qa_pairs, cause_qa_pairs, and result_qa_pairs
by comparing generated responses (with and without reranking) against ground truth answers.

Metrics:
- ROUGE-1, ROUGE-2, ROUGE-L scores
- BERTScore (precision, recall, F1)

Usage:
    python testscripts/evaluate_response_quality.py
    python testscripts/evaluate_response_quality.py --max-samples 20 --top-k 5
"""

import os
import sys
import json
import csv
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT / "src")

from tqdm import tqdm
import importlib.util

# Helper to load modules with numeric prefixes
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load query module
_query_module = _load_module("query", PROJECT_ROOT / "src" / "05_query.py")
VideoRAGQueryProcessor = _query_module.VideoRAGQueryProcessor

# Load config module
_config_module = _load_module("config", PROJECT_ROOT / "src" / "00_config.py")
LLAVA_MODEL = _config_module.LLAVA_MODEL

# Import scoring libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge-score not installed. Run: pip install rouge-score")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("WARNING: bert-score not installed. Run: pip install bert-score")


@dataclass
class ResponseEvaluationResult:
    """Result of a single response evaluation."""
    sample_id: str
    video_id: str
    segment_id: int
    question_type: str
    anomaly_type: str
    gt_start: float
    gt_end: float
    question: str
    ground_truth_answer: str
    # Without reranker (CLIP only)
    no_rerank_response: str
    no_rerank_retrieved_video: str
    no_rerank_retrieved_timestamps: str
    no_rerank_num_chunks: int
    # No reranker scores
    no_rerank_rouge1: float
    no_rerank_rouge2: float
    no_rerank_rougeL: float
    no_rerank_bert_precision: float
    no_rerank_bert_recall: float
    no_rerank_bert_f1: float
    # With reranker
    rerank_response: str
    rerank_retrieved_video: str
    rerank_retrieved_timestamps: str
    rerank_num_chunks: int
    # Reranker scores
    rerank_rouge1: float
    rerank_rouge2: float
    rerank_rougeL: float
    rerank_bert_precision: float
    rerank_bert_recall: float
    rerank_bert_f1: float


class ResponseQualityEvaluator:
    """Evaluator for AI Analysis response quality."""
    
    def __init__(self, dataset_path: str, top_k: int = 5):
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.query_processor = None
        self.rouge_scorer = None
        self.bert_scores_cache = {"candidates": [], "references": []}
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_rouge_scores(self, candidate: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores between candidate and reference."""
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"ROUGE error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_bert_scores_batch(self, candidates: List[str], references: List[str]) -> List[Dict[str, float]]:
        """Compute BERTScore for a batch of candidates and references."""
        if not BERT_SCORE_AVAILABLE or not candidates:
            return [{'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for _ in candidates]
        
        try:
            P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
            return [
                {'precision': p.item(), 'recall': r.item(), 'f1': f.item()}
                for p, r, f in zip(P, R, F1)
            ]
        except Exception as e:
            print(f"BERTScore error: {e}")
            return [{'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for _ in candidates]
        
    def initialize(self) -> bool:
        """Initialize VideoRAG."""
        try:
            print("Initializing VideoRAG...")
            self.query_processor = VideoRAGQueryProcessor()
            
            if not self.query_processor.load_index():
                print("ERROR: Could not load index")
                return False
            
            # Check Ollama connection for LLAVA
            if not self.query_processor.check_ollama_connection():
                print("WARNING: Ollama/LLAVA not available. Responses will be limited.")
            else:
                print("✓ Ollama/LLAVA connection verified")
            
            print(f"Loaded index with {len(self.query_processor.video_index.metadata)} chunks")
            return True
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_chunks_for_video(self, video_id: str) -> List[Dict]:
        """Get all indexed chunks for a specific video_id."""
        video_name = f"{video_id}.mp4"
        chunks = []
        
        for chunk in self.query_processor.video_index.metadata:
            chunk_video = Path(chunk.get('video_path', '')).name
            if chunk_video == video_name:
                chunks.append(chunk)
        
        return chunks
    
    def filter_chunks_by_video(self, chunks: List[Dict], video_id: str) -> List[Dict]:
        """Filter chunks to only those from the target video."""
        video_name = f"{video_id}.mp4"
        return [c for c in chunks if Path(c.get('video_path', '')).name == video_name]
    
    def load_dataset(self) -> List[Dict]:
        """Load samples from description_qa_pairs, cause_qa_pairs, result_qa_pairs."""
        print(f"Loading dataset: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        
        # Filter to target question types with non-normal anomaly
        target_types = {'description_qa_pairs', 'cause_qa_pairs', 'result_qa_pairs'}
        
        # Group by (video_id, segment_id) to get one of each type per segment
        segments = defaultdict(dict)
        
        for sample in data:
            q_type = sample['question_type']
            if q_type not in target_types:
                continue
            
            # Skip normal anomaly type
            if sample['anomaly_type'].lower() == 'normal':
                continue
            
            key = (sample['video_id'], sample['segment_id'])
            
            # Keep one sample per question type per segment
            if q_type not in segments[key]:
                segments[key][q_type] = sample
        
        # Flatten and collect samples
        filtered = []
        for segment_key, type_samples in segments.items():
            for q_type, sample in type_samples.items():
                filtered.append(sample)
        
        print(f"Filtered to {len(filtered)} samples across {len(segments)} unique segments")
        
        # Stats by question type
        by_type = defaultdict(int)
        for s in filtered:
            by_type[s['question_type']] += 1
        print("\nBy question type:")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")
        
        # Stats by anomaly type
        by_anomaly = defaultdict(int)
        for s in filtered:
            by_anomaly[s['anomaly_type'].split(",")[0].strip()] += 1
        print("\nBy anomaly type (top 10):")
        for t, c in sorted(by_anomaly.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t}: {c}")
        
        return filtered
    
    def query_without_rerank(self, question: str, video_id: str) -> Dict:
        """Query VideoRAG without reranking, filtered to specific video."""
        try:
            # First, get chunks only for this video from the metadata
            video_chunks = self.get_chunks_for_video(video_id)
            
            if not video_chunks:
                return {"error": f"No chunks found for {video_id}", "answer": "No chunks found", "chunks": []}
            
            # Encode query and compute similarity for video-specific chunks
            query_embedding = self.query_processor.clip_extractor.encode_text(question)
            
            # Score each chunk
            for chunk in video_chunks:
                # Get the chunk's embedding from the index
                chunk_idx = None
                for i, meta in enumerate(self.query_processor.video_index.metadata):
                    if meta.get('chunk_id') == chunk.get('chunk_id'):
                        chunk_idx = i
                        break
                
                if chunk_idx is not None:
                    chunk_embedding = self.query_processor.video_index.visual_index.reconstruct(chunk_idx)
                    similarity = float(np.dot(query_embedding.flatten(), chunk_embedding.flatten()))
                    chunk['similarity_score'] = similarity
                else:
                    chunk['similarity_score'] = 0.0
            
            # Sort by similarity and take top_k
            video_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            video_chunks = video_chunks[:self.top_k]
            
            # Add rank field required by prepare_context_for_llava
            for i, chunk in enumerate(video_chunks):
                chunk['rank'] = i + 1
            
            # Prepare context and generate response
            context = self.query_processor.prepare_context_for_llava(question, video_chunks)
            llava_prompt = self.query_processor.generate_llava_prompt(context)
            
            # Encode keyframes
            encoded_images = []
            sample_keyframes = context['all_keyframes'][:6]
            for kf_path in sample_keyframes:
                if Path(kf_path).exists():
                    encoded_img = self.query_processor.ollama_client.encode_image_to_base64(kf_path)
                    if encoded_img:
                        encoded_images.append(encoded_img)
            
            # Get LLAVA response
            response = self.query_processor.ollama_client.generate_response(
                llava_prompt, model=LLAVA_MODEL, images=encoded_images
            )
            
            return {
                "answer": response,
                "chunks": [
                    {
                        "video": Path(c.get('video_path', '')).name,
                        "timestamp": f"{c.get('start_time', 0):.1f}s - {c.get('end_time', 0):.1f}s",
                        "similarity_score": c.get('similarity_score', 0)
                    }
                    for c in video_chunks
                ]
            }
        except Exception as e:
            print(f"Error in query_without_rerank: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "answer": f"Error: {e}", "chunks": []}
    
    def query_with_rerank(self, question: str, video_id: str) -> Dict:
        """Query VideoRAG with NEW 3-STAGE PIPELINE, filtered to specific video.
        
        Stage 1: Top 15 using Visual embedding (CLIP cosine similarity)
        Stage 2: Top 10 using Textual embedding (60%) + BM25 (40%)
        Stage 3: Top 5 using Temporal context averaging (2 left + current + 2 right)
        """
        import re
        
        try:
            # First, get chunks only for this video from the metadata
            video_chunks = self.get_chunks_for_video(video_id)
            
            if not video_chunks:
                return {"error": f"No chunks found for {video_id}", "answer": "No chunks found", "chunks": []}
            
            # Encode query
            query_embedding = self.query_processor.clip_extractor.encode_text(question)
            query_embedding_np = query_embedding.flatten()
            
            # Build chunk_id to metadata lookup
            chunk_id_to_metadata = {m['chunk_id']: m for m in self.query_processor.video_index.metadata}
            
            # ================================================================
            # STAGE 1: Score all video chunks by Visual Embedding
            # ================================================================
            for chunk in video_chunks:
                chunk_idx = None
                for i, meta in enumerate(self.query_processor.video_index.metadata):
                    if meta.get('chunk_id') == chunk.get('chunk_id'):
                        chunk_idx = i
                        break
                
                if chunk_idx is not None:
                    chunk_embedding = self.query_processor.video_index.visual_index.reconstruct(chunk_idx)
                    visual_score = float(np.dot(query_embedding_np, chunk_embedding.flatten()))
                    chunk['visual_score'] = visual_score
                    chunk['similarity_score'] = visual_score
                    # Get caption embedding for stage 2
                    if chunk['chunk_id'] in chunk_id_to_metadata:
                        meta = chunk_id_to_metadata[chunk['chunk_id']]
                        chunk['embedding'] = meta.get('embedding')
                        chunk['caption_embedding'] = meta.get('caption_embedding')
                else:
                    chunk['visual_score'] = 0.0
                    chunk['similarity_score'] = 0.0
            
            # Sort by visual score and take top 15
            video_chunks.sort(key=lambda x: x.get('visual_score', 0), reverse=True)
            stage1_chunks = video_chunks[:15]
            
            # ================================================================
            # STAGE 2: Top 10 using Textual + BM25
            # ================================================================
            query_tokens = set(re.findall(r'\w+', question.lower()))
            
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
            video_chunks = stage2_chunks[:self.top_k]
            
            # Add rank field required by prepare_context_for_llava
            for i, chunk in enumerate(video_chunks):
                chunk['rank'] = i + 1
            
            # Prepare context and generate response
            context = self.query_processor.prepare_context_for_llava(question, video_chunks)
            llava_prompt = self.query_processor.generate_llava_prompt(context)
            
            # Encode keyframes
            encoded_images = []
            sample_keyframes = context['all_keyframes'][:6]
            for kf_path in sample_keyframes:
                if Path(kf_path).exists():
                    encoded_img = self.query_processor.ollama_client.encode_image_to_base64(kf_path)
                    if encoded_img:
                        encoded_images.append(encoded_img)
            
            # Get LLAVA response
            response = self.query_processor.ollama_client.generate_response(
                llava_prompt, model=LLAVA_MODEL, images=encoded_images
            )
            
            return {
                "answer": response,
                "chunks": [
                    {
                        "video": Path(c.get('video_path', '')).name,
                        "timestamp": f"{c.get('start_time', 0):.1f}s - {c.get('end_time', 0):.1f}s",
                        "similarity_score": c.get('temporal_score', c.get('similarity_score', 0))
                    }
                    for c in video_chunks
                ]
            }
        except Exception as e:
            print(f"Error in query_with_rerank: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "answer": f"Error: {e}", "chunks": []}
    
    def extract_result_info(self, result: Dict) -> Tuple[str, str, str, int]:
        """Extract response, video, timestamps, num_chunks from query result."""
        response = result.get('answer', 'No response')
        chunks = result.get('chunks', [])
        num_chunks = len(chunks)
        
        if chunks:
            # Get the top chunk's video and timestamp
            top_chunk = chunks[0]
            video = top_chunk.get('video', 'Unknown')
            timestamp = top_chunk.get('timestamp', 'Unknown')
            
            # Collect all timestamps
            all_timestamps = [c.get('timestamp', '') for c in chunks[:5]]
            timestamps_str = " | ".join(all_timestamps)
        else:
            video = "No results"
            timestamps_str = "None"
        
        return response, video, timestamps_str, num_chunks
    
    def evaluate_sample(self, sample: Dict) -> ResponseEvaluationResult:
        """Evaluate one sample with both retrieval methods, filtered to the target video."""
        video_id = sample['video_id']
        segment_id = sample['segment_id']
        question_type = sample['question_type']
        anomaly_type = sample['anomaly_type']
        gt_start = sample['start_time']
        gt_end = sample['end_time']
        question = sample['question']
        ground_truth = sample['answer']
        
        print(f"\n--- Evaluating: {video_id} segment {segment_id} ({question_type}) ---")
        print(f"Question: {question[:80]}...")
        
        # Check how many chunks exist for this video
        video_chunks = self.get_chunks_for_video(video_id)
        print(f"  Found {len(video_chunks)} indexed chunks for {video_id}")
        
        if not video_chunks:
            print(f"  WARNING: No chunks found for video {video_id}")
            return ResponseEvaluationResult(
                sample_id=sample['sample_id'],
                video_id=video_id,
                segment_id=segment_id,
                question_type=question_type,
                anomaly_type=anomaly_type,
                gt_start=gt_start,
                gt_end=gt_end,
                question=question,
                ground_truth_answer=ground_truth,
                no_rerank_response="No chunks indexed for this video",
                no_rerank_retrieved_video="None",
                no_rerank_retrieved_timestamps="None",
                no_rerank_num_chunks=0,
                no_rerank_rouge1=0.0,
                no_rerank_rouge2=0.0,
                no_rerank_rougeL=0.0,
                no_rerank_bert_precision=0.0,
                no_rerank_bert_recall=0.0,
                no_rerank_bert_f1=0.0,
                rerank_response="No chunks indexed for this video",
                rerank_retrieved_video="None",
                rerank_retrieved_timestamps="None",
                rerank_num_chunks=0,
                rerank_rouge1=0.0,
                rerank_rouge2=0.0,
                rerank_rougeL=0.0,
                rerank_bert_precision=0.0,
                rerank_bert_recall=0.0,
                rerank_bert_f1=0.0,
            )
        
        # Query WITHOUT reranking (filtered to this video)
        print("  Querying without reranker (video-filtered)...")
        result_no_rerank = self.query_without_rerank(question, video_id)
        no_rerank_response, no_rerank_video, no_rerank_timestamps, no_rerank_num = \
            self.extract_result_info(result_no_rerank)
        
        # Query WITH reranking (filtered to this video)
        print("  Querying with reranker (video-filtered)...")
        result_rerank = self.query_with_rerank(question, video_id)
        rerank_response, rerank_video, rerank_timestamps, rerank_num = \
            self.extract_result_info(result_rerank)
        
        # Compute ROUGE scores
        print("  Computing ROUGE scores...")
        no_rerank_rouge = self.compute_rouge_scores(no_rerank_response, ground_truth)
        rerank_rouge = self.compute_rouge_scores(rerank_response, ground_truth)
        
        # Compute BERTScore (batch for efficiency)
        print("  Computing BERTScore...")
        bert_scores = self.compute_bert_scores_batch(
            [no_rerank_response, rerank_response],
            [ground_truth, ground_truth]
        )
        no_rerank_bert = bert_scores[0]
        rerank_bert = bert_scores[1]
        
        print(f"  ROUGE-L: no_rerank={no_rerank_rouge['rougeL']:.4f}, rerank={rerank_rouge['rougeL']:.4f}")
        print(f"  BERTScore F1: no_rerank={no_rerank_bert['f1']:.4f}, rerank={rerank_bert['f1']:.4f}")
        
        return ResponseEvaluationResult(
            sample_id=sample['sample_id'],
            video_id=video_id,
            segment_id=segment_id,
            question_type=question_type,
            anomaly_type=anomaly_type,
            gt_start=gt_start,
            gt_end=gt_end,
            question=question,
            ground_truth_answer=ground_truth,
            # Without reranker
            no_rerank_response=no_rerank_response,
            no_rerank_retrieved_video=no_rerank_video,
            no_rerank_retrieved_timestamps=no_rerank_timestamps,
            no_rerank_num_chunks=no_rerank_num,
            # ROUGE scores - no reranker
            no_rerank_rouge1=no_rerank_rouge['rouge1'],
            no_rerank_rouge2=no_rerank_rouge['rouge2'],
            no_rerank_rougeL=no_rerank_rouge['rougeL'],
            # BERTScore - no reranker
            no_rerank_bert_precision=no_rerank_bert['precision'],
            no_rerank_bert_recall=no_rerank_bert['recall'],
            no_rerank_bert_f1=no_rerank_bert['f1'],
            # With reranker
            rerank_response=rerank_response,
            rerank_retrieved_video=rerank_video,
            rerank_retrieved_timestamps=rerank_timestamps,
            rerank_num_chunks=rerank_num,
            # ROUGE scores - reranker
            rerank_rouge1=rerank_rouge['rouge1'],
            rerank_rouge2=rerank_rouge['rouge2'],
            rerank_rougeL=rerank_rouge['rougeL'],
            # BERTScore - reranker
            rerank_bert_precision=rerank_bert['precision'],
            rerank_bert_recall=rerank_bert['recall'],
            rerank_bert_f1=rerank_bert['f1'],
        )
    
    def run(self, max_samples: int = None, output_path: str = None, 
            question_types: List[str] = None) -> List[ResponseEvaluationResult]:
        """Run evaluation.
        
        Args:
            max_samples: Maximum samples to evaluate (None for all)
            output_path: Path to save CSV results
            question_types: Filter to specific question types (e.g., ['description_qa_pairs'])
        """
        if not self.initialize():
            return []
        
        samples = self.load_dataset()
        
        # Filter by question type if specified
        if question_types:
            samples = [s for s in samples if s['question_type'] in question_types]
            print(f"Filtered to {len(samples)} samples of types: {question_types}")
        
        # Shuffle and limit
        random.shuffle(samples)
        if max_samples:
            samples = samples[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"\nEvaluating {len(samples)} samples...")
        print("="*80)
        
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}]", end="")
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(PROJECT_ROOT / "data" / f"response_quality_eval_{timestamp}.csv")
        elif not os.path.isabs(output_path):
            output_path = str(PROJECT_ROOT / output_path)
        
        if results:
            self.save_csv(results, output_path)
            self.print_summary(results)
        
        return results
    
    def save_csv(self, results: List[ResponseEvaluationResult], path: str):
        """Save results to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                row = asdict(r)
                # Truncate long fields for CSV readability
                for key in ['ground_truth_answer', 'no_rerank_response', 'rerank_response']:
                    if len(row[key]) > 2000:
                        row[key] = row[key][:2000] + "... [truncated]"
                writer.writerow(row)
        
        print(f"\n✓ Saved {len(results)} results to: {path}")
    
    def print_summary(self, results: List[ResponseEvaluationResult]):
        """Print summary of evaluation."""
        n = len(results)
        if n == 0:
            return
        
        print("\n" + "="*80)
        print("RESPONSE QUALITY EVALUATION SUMMARY")
        print("="*80)
        print(f"Total samples evaluated: {n}")
        print("Note: All queries are filtered to the target video_id")
        print("Pipeline: CLIP-only vs 3-Stage (Visual→Textual+BM25→Temporal)")
        
        # By question type
        by_type = defaultdict(int)
        for r in results:
            by_type[r.question_type] += 1
        
        print("\nSamples by question type:")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")
        
        # Average response lengths
        no_rerank_avg_len = sum(len(r.no_rerank_response) for r in results) / n
        rerank_avg_len = sum(len(r.rerank_response) for r in results) / n
        gt_avg_len = sum(len(r.ground_truth_answer) for r in results) / n
        
        print(f"\nAverage response lengths:")
        print(f"  Ground Truth:  {gt_avg_len:.0f} chars")
        print(f"  No Reranker:   {no_rerank_avg_len:.0f} chars")
        print(f"  With Reranker: {rerank_avg_len:.0f} chars")
        
        # Count samples with valid chunks
        valid_no_rerank = sum(1 for r in results if r.no_rerank_num_chunks > 0)
        valid_rerank = sum(1 for r in results if r.rerank_num_chunks > 0)
        
        print(f"\nSamples with indexed chunks:")
        print(f"  No Reranker:   {valid_no_rerank}/{n} ({100*valid_no_rerank/n:.1f}%)")
        print(f"  With Reranker: {valid_rerank}/{n} ({100*valid_rerank/n:.1f}%)")
        
        # Average chunks retrieved
        avg_chunks_no = sum(r.no_rerank_num_chunks for r in results) / n
        avg_chunks_re = sum(r.rerank_num_chunks for r in results) / n
        print(f"\nAverage chunks retrieved:")
        print(f"  No Reranker:   {avg_chunks_no:.1f}")
        print(f"  With Reranker: {avg_chunks_re:.1f}")
        
        # ROUGE Scores
        print(f"\n{'='*40}")
        print("ROUGE SCORES (F1)")
        print(f"{'='*40}")
        
        avg_rouge1_no = sum(r.no_rerank_rouge1 for r in results) / n
        avg_rouge2_no = sum(r.no_rerank_rouge2 for r in results) / n
        avg_rougeL_no = sum(r.no_rerank_rougeL for r in results) / n
        
        avg_rouge1_re = sum(r.rerank_rouge1 for r in results) / n
        avg_rouge2_re = sum(r.rerank_rouge2 for r in results) / n
        avg_rougeL_re = sum(r.rerank_rougeL for r in results) / n
        
        print(f"\nNo Reranker:")
        print(f"  ROUGE-1: {avg_rouge1_no:.4f}")
        print(f"  ROUGE-2: {avg_rouge2_no:.4f}")
        print(f"  ROUGE-L: {avg_rougeL_no:.4f}")
        
        print(f"\nWith Reranker:")
        print(f"  ROUGE-1: {avg_rouge1_re:.4f}")
        print(f"  ROUGE-2: {avg_rouge2_re:.4f}")
        print(f"  ROUGE-L: {avg_rougeL_re:.4f}")
        
        print(f"\nImprovement (Reranker - No Reranker):")
        print(f"  ROUGE-1: {avg_rouge1_re - avg_rouge1_no:+.4f}")
        print(f"  ROUGE-2: {avg_rouge2_re - avg_rouge2_no:+.4f}")
        print(f"  ROUGE-L: {avg_rougeL_re - avg_rougeL_no:+.4f}")
        
        # BERTScore
        print(f"\n{'='*40}")
        print("BERTScore")
        print(f"{'='*40}")
        
        avg_bert_p_no = sum(r.no_rerank_bert_precision for r in results) / n
        avg_bert_r_no = sum(r.no_rerank_bert_recall for r in results) / n
        avg_bert_f1_no = sum(r.no_rerank_bert_f1 for r in results) / n
        
        avg_bert_p_re = sum(r.rerank_bert_precision for r in results) / n
        avg_bert_r_re = sum(r.rerank_bert_recall for r in results) / n
        avg_bert_f1_re = sum(r.rerank_bert_f1 for r in results) / n
        
        print(f"\nNo Reranker:")
        print(f"  Precision: {avg_bert_p_no:.4f}")
        print(f"  Recall:    {avg_bert_r_no:.4f}")
        print(f"  F1:        {avg_bert_f1_no:.4f}")
        
        print(f"\nWith Reranker:")
        print(f"  Precision: {avg_bert_p_re:.4f}")
        print(f"  Recall:    {avg_bert_r_re:.4f}")
        print(f"  F1:        {avg_bert_f1_re:.4f}")
        
        print(f"\nImprovement (Reranker - No Reranker):")
        print(f"  Precision: {avg_bert_p_re - avg_bert_p_no:+.4f}")
        print(f"  Recall:    {avg_bert_r_re - avg_bert_r_no:+.4f}")
        print(f"  F1:        {avg_bert_f1_re - avg_bert_f1_no:+.4f}")
        
        print("\n" + "="*80)
        print("Results saved. Compare 'ground_truth_answer' with generated responses.")
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VideoRAG response quality")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "ucf_surveillance_vqa_dataset.json"))
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum samples to evaluate")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Number of chunks to retrieve")
    parser.add_argument("--output", default=None, 
                        help="Output CSV path")
    parser.add_argument("--question-types", nargs='+', 
                        choices=['description_qa_pairs', 'cause_qa_pairs', 'result_qa_pairs'],
                        default=None,
                        help="Filter to specific question types")
    
    args = parser.parse_args()
    
    evaluator = ResponseQualityEvaluator(args.dataset, args.top_k)
    evaluator.run(args.max_samples, args.output, args.question_types)


if __name__ == "__main__":
    main()
