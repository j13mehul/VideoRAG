"""
Evaluation module for SurveillanceVQA-VideoRAG benchmarking.

Provides metrics and evaluation utilities for:
- Retrieval quality (Recall@K, MRR, nDCG)
- Answer generation quality (ROUGE, BLEU, BERTScore)
- Category-wise performance analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import re

# Try to import optional dependencies
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not installed. Install with: pip install bert-score")


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation."""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "Recall@1": round(self.recall_at_1, 4),
            "Recall@3": round(self.recall_at_3, 4),
            "Recall@5": round(self.recall_at_5, 4),
            "Recall@10": round(self.recall_at_10, 4),
            "MRR": round(self.mrr, 4),
            "nDCG@5": round(self.ndcg_at_5, 4),
            "nDCG@10": round(self.ndcg_at_10, 4),
        }


@dataclass
class GenerationMetrics:
    """Metrics for answer generation evaluation."""
    rouge_l: float = 0.0
    bleu_4: float = 0.0
    bert_score_f1: float = 0.0
    exact_match: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "ROUGE-L": round(self.rouge_l, 4),
            "BLEU-4": round(self.bleu_4, 4),
            "BERTScore-F1": round(self.bert_score_f1, 4),
            "ExactMatch": round(self.exact_match, 4),
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation_metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    category_metrics: Dict[str, Dict] = field(default_factory=dict)
    anomaly_type_metrics: Dict[str, Dict] = field(default_factory=dict)
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "retrieval": self.retrieval_metrics.to_dict(),
            "generation": self.generation_metrics.to_dict(),
            "by_category": self.category_metrics,
            "by_anomaly_type": self.anomaly_type_metrics,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Evaluation Results",
            "=" * 60,
            f"Number of samples: {self.num_samples}",
            "",
            "Retrieval Metrics:",
            "-" * 40,
        ]
        for k, v in self.retrieval_metrics.to_dict().items():
            lines.append(f"  {k}: {v:.4f}")
        
        lines.extend([
            "",
            "Generation Metrics:",
            "-" * 40,
        ])
        for k, v in self.generation_metrics.to_dict().items():
            lines.append(f"  {k}: {v:.4f}")
        
        if self.category_metrics:
            lines.extend([
                "",
                "Metrics by Question Category:",
                "-" * 40,
            ])
            for cat, metrics in self.category_metrics.items():
                lines.append(f"  {cat}:")
                for k, v in metrics.items():
                    lines.append(f"    {k}: {v:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class SurveillanceVQAEvaluator:
    """
    Evaluator for SurveillanceVQA-VideoRAG pipeline.
    
    Evaluates:
    1. Retrieval quality: Does the system retrieve the correct video/chunk?
    2. Generation quality: How good are the generated answers?
    3. Category-wise analysis: Performance breakdown by question type
    """
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            ground_truth_path: Path to ground truth JSON file
        """
        self.ground_truth = {}
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)
        
        # Initialize scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def load_ground_truth(self, path: str) -> None:
        """Load ground truth data from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Index by sample_id or video_id + question
        for item in data:
            key = item.get("sample_id") or f"{item['video_id']}_{item['question'][:50]}"
            self.ground_truth[key] = item
        
        print(f"Loaded {len(self.ground_truth)} ground truth samples")
    
    def evaluate_retrieval(
        self,
        predictions: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Args:
            predictions: List of prediction dicts with:
                - query_id or sample_id
                - retrieved_videos: List of retrieved video IDs
                - ground_truth_video: Expected video ID
            k_values: K values for Recall@K
        
        Returns:
            RetrievalMetrics object
        """
        recalls = {k: [] for k in k_values}
        reciprocal_ranks = []
        ndcg_5, ndcg_10 = [], []
        
        for pred in predictions:
            gt_video = pred.get("ground_truth_video")
            retrieved = pred.get("retrieved_videos", [])
            
            if not gt_video or not retrieved:
                continue
            
            # Find rank of ground truth video
            try:
                rank = retrieved.index(gt_video) + 1
            except ValueError:
                rank = float('inf')
            
            # Recall@K
            for k in k_values:
                recalls[k].append(1.0 if rank <= k else 0.0)
            
            # MRR
            reciprocal_ranks.append(1.0 / rank if rank != float('inf') else 0.0)
            
            # nDCG
            ndcg_5.append(self._calculate_ndcg(rank, 5))
            ndcg_10.append(self._calculate_ndcg(rank, 10))
        
        return RetrievalMetrics(
            recall_at_1=np.mean(recalls[1]) if recalls[1] else 0.0,
            recall_at_3=np.mean(recalls[3]) if recalls[3] else 0.0,
            recall_at_5=np.mean(recalls[5]) if recalls[5] else 0.0,
            recall_at_10=np.mean(recalls[10]) if recalls[10] else 0.0,
            mrr=np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            ndcg_at_5=np.mean(ndcg_5) if ndcg_5 else 0.0,
            ndcg_at_10=np.mean(ndcg_10) if ndcg_10 else 0.0,
        )
    
    def _calculate_ndcg(self, rank: int, k: int) -> float:
        """Calculate nDCG for a single query."""
        if rank > k or rank == float('inf'):
            return 0.0
        
        # DCG: relevance / log2(rank + 1)
        dcg = 1.0 / np.log2(rank + 1)
        
        # Ideal DCG (best case: rank 1)
        idcg = 1.0 / np.log2(2)
        
        return dcg / idcg
    
    def evaluate_generation(
        self,
        predictions: List[Dict],
        use_bertscore: bool = True
    ) -> GenerationMetrics:
        """
        Evaluate answer generation quality.
        
        Args:
            predictions: List of prediction dicts with:
                - predicted_answer
                - ground_truth_answer
            use_bertscore: Whether to compute BERTScore (slow)
        
        Returns:
            GenerationMetrics object
        """
        rouge_scores = []
        bleu_scores = []
        exact_matches = []
        
        preds_for_bert = []
        refs_for_bert = []
        
        for pred in tqdm(predictions, desc="Evaluating generation"):
            predicted = pred.get("predicted_answer", "")
            ground_truth = pred.get("ground_truth_answer", "")
            
            if not predicted or not ground_truth:
                continue
            
            # Normalize text
            predicted = self._normalize_text(predicted)
            ground_truth = self._normalize_text(ground_truth)
            
            # Exact match
            exact_matches.append(1.0 if predicted == ground_truth else 0.0)
            
            # ROUGE-L
            if ROUGE_AVAILABLE:
                score = self.rouge_scorer.score(ground_truth, predicted)
                rouge_scores.append(score['rougeL'].fmeasure)
            
            # BLEU-4
            if BLEU_AVAILABLE:
                try:
                    bleu = sentence_bleu(
                        [ground_truth.split()],
                        predicted.split(),
                        smoothing_function=SmoothingFunction().method1
                    )
                    bleu_scores.append(bleu)
                except:
                    bleu_scores.append(0.0)
            
            # Collect for BERTScore
            if use_bertscore and BERTSCORE_AVAILABLE:
                preds_for_bert.append(predicted)
                refs_for_bert.append(ground_truth)
        
        # Calculate BERTScore in batch
        bert_f1 = 0.0
        if use_bertscore and BERTSCORE_AVAILABLE and preds_for_bert:
            try:
                _, _, f1_scores = bert_score(
                    preds_for_bert, refs_for_bert,
                    lang='en', verbose=False
                )
                bert_f1 = float(np.mean(f1_scores.numpy()))
            except Exception as e:
                print(f"BERTScore failed: {e}")
        
        return GenerationMetrics(
            rouge_l=np.mean(rouge_scores) if rouge_scores else 0.0,
            bleu_4=np.mean(bleu_scores) if bleu_scores else 0.0,
            bert_score_f1=bert_f1,
            exact_match=np.mean(exact_matches) if exact_matches else 0.0,
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def evaluate_by_category(
        self,
        predictions: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Evaluate performance by question category.
        
        Args:
            predictions: List of predictions with 'category' field
        
        Returns:
            Dict mapping category to metrics
        """
        category_preds = defaultdict(list)
        
        for pred in predictions:
            category = pred.get("category", "unknown")
            category_preds[category].append(pred)
        
        results = {}
        for category, preds in category_preds.items():
            gen_metrics = self.evaluate_generation(preds, use_bertscore=False)
            results[category] = gen_metrics.to_dict()
            results[category]["num_samples"] = len(preds)
        
        return results
    
    def evaluate_by_anomaly_type(
        self,
        predictions: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Evaluate performance by anomaly type.
        
        Args:
            predictions: List of predictions with 'anomaly_type' field
        
        Returns:
            Dict mapping anomaly type to metrics
        """
        type_preds = defaultdict(list)
        
        for pred in predictions:
            anomaly_type = pred.get("anomaly_type", "unknown")
            type_preds[anomaly_type].append(pred)
        
        results = {}
        for anomaly_type, preds in type_preds.items():
            gen_metrics = self.evaluate_generation(preds, use_bertscore=False)
            results[anomaly_type] = gen_metrics.to_dict()
            results[anomaly_type]["num_samples"] = len(preds)
        
        return results
    
    def evaluate_full(
        self,
        predictions: List[Dict],
        include_category_analysis: bool = True,
        include_anomaly_analysis: bool = True,
        use_bertscore: bool = True
    ) -> EvaluationResult:
        """
        Run full evaluation pipeline.
        
        Args:
            predictions: List of prediction dicts
            include_category_analysis: Include category breakdown
            include_anomaly_analysis: Include anomaly type breakdown
            use_bertscore: Compute BERTScore
        
        Returns:
            EvaluationResult object
        """
        result = EvaluationResult(num_samples=len(predictions))
        
        # Retrieval metrics
        print("Evaluating retrieval...")
        result.retrieval_metrics = self.evaluate_retrieval(predictions)
        
        # Generation metrics
        print("Evaluating generation...")
        result.generation_metrics = self.evaluate_generation(
            predictions, use_bertscore=use_bertscore
        )
        
        # Category analysis
        if include_category_analysis:
            print("Analyzing by category...")
            result.category_metrics = self.evaluate_by_category(predictions)
        
        # Anomaly type analysis
        if include_anomaly_analysis:
            print("Analyzing by anomaly type...")
            result.anomaly_type_metrics = self.evaluate_by_anomaly_type(predictions)
        
        return result


def run_benchmark(
    query_processor,
    benchmark_path: str,
    output_path: Optional[str] = None,
    max_queries: Optional[int] = None
) -> EvaluationResult:
    """
    Run benchmark evaluation.
    
    Args:
        query_processor: VideoRAGQueryProcessor instance
        benchmark_path: Path to benchmark JSON file
        output_path: Path to save results
        max_queries: Maximum queries to evaluate
    
    Returns:
        EvaluationResult object
    """
    # Load benchmark
    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)
    
    if max_queries:
        benchmark = benchmark[:max_queries]
    
    predictions = []
    
    print(f"Running benchmark on {len(benchmark)} queries...")
    
    for item in tqdm(benchmark):
        query = item["question"]
        
        # Run query
        result = query_processor.query(query)
        
        # Extract retrieved videos
        retrieved_videos = []
        if "retrieved_chunks" in result:
            for chunk in result["retrieved_chunks"]:
                video_id = Path(chunk.get("video_path", "")).stem
                if video_id not in retrieved_videos:
                    retrieved_videos.append(video_id)
        
        predictions.append({
            "query_id": item.get("sample_id"),
            "question": query,
            "ground_truth_video": item.get("video_id"),
            "ground_truth_answer": item.get("answer"),
            "predicted_answer": result.get("answer", ""),
            "retrieved_videos": retrieved_videos,
            "category": item.get("question_category"),
            "anomaly_type": item.get("anomaly_type"),
        })
    
    # Evaluate
    evaluator = SurveillanceVQAEvaluator()
    results = evaluator.evaluate_full(predictions)
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                "results": results.to_dict(),
                "predictions": predictions
            }, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    print(results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SurveillanceVQA-VideoRAG")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to benchmark JSON")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--max-queries", type=int, help="Maximum queries to evaluate")
    
    args = parser.parse_args()
    
    # Import query processor
    from query import VideoRAGQueryProcessor
    
    processor = VideoRAGQueryProcessor()
    processor.load_index()
    
    run_benchmark(
        query_processor=processor,
        benchmark_path=args.benchmark,
        output_path=args.output,
        max_queries=args.max_queries
    )
