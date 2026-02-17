"""
LLM-as-Judge Evaluation for VideoRAG Response Quality
Uses Ollama to score responses across 4 dimensions:
- Contextual Integration (CI): Factual accuracy
- Detail Orientation (DO): Completeness and specificity  
- Contextual Understanding (CU): Alignment with narrative/emotional tone
- Temporal Understanding (TU): Event sequence and time-related logic

Average Score = 0.25 * CI + 0.25 * DO + 0.25 * CU + 0.25 * TU
"""

import os
import sys
import csv
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Dimension prompts
DIMENSION_PROMPTS = {
    "CI": {
        "name": "Contextual Integration (Factual Accuracy)",
        "system": (
            "You are an AI assistant tasked with evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent."
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the factual information presented in the video and should not contain any misinterpretations or misinformation.\n"
            "- Consider synonyms or paraphrases as valid matches, but only if the response is factually accurate and aligns with the video content.\n"
            "- Evaluate the factual accuracy of the prediction compared to the answer, do not assume anything from the world knowledge.\n"
            "- Assign a factual accuracy score between 0 and 5, where 5 indicates the highest level of factual consistency.\n"
            "- Base your evaluation on the following scale:\n"
            "  5: PERFECT match in terms of correctness with no factual errors.\n"
            "  4: Very little discrepancies in details, but the information generated is mostly correct and aligns with the video content.\n"
            "  3: Mostly correct information with minor discrepancies.\n"
            "  2: Very little correct information, though some parts are correct.\n"
            "  1: Mostly incorrect or irrelevant details, though some parts are correct.\n"
            "  0: COMPLETELY incorrect response with no factual consistency.\n"
        )
    },
    "DO": {
        "name": "Detail Orientation",
        "system": (
            "You are an AI assistant tasked with evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity."
            "------"
            "##INSTRUCTIONS: "
            "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
            "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Do not assume anything from the world knowledge."
            "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.\n"
            "- Assign a detail orientation score between 0 and 5, where 5 indicates the highest level of detail orientation.\n"
            "- Base your evaluation on the following scale:\n"
            "  5: PERFECT match in terms of completeness and specificity with no errors.\n"
            "  4: Very little omissions or lack of specific details, but mostly complete.\n"
            "  3: Most of the specific details are correct with minor unnoticeable omissions or discrepancies.\n"
            "  2: Very little correct details, though some details are correct.\n"
            "  1: Mostly incorrect details.\n"
            "  0: COMPLETELY incorrect and incomplete response with generic points only."
        )
    },
    "CU": {
        "name": "Contextual Understanding",
        "system": (
            "You are an AI assistant tasked with evaluating the contextual understanding in results for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content."
            "------"
            "##INSTRUCTIONS: "
            "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
            "- The predicted answer must capture the main themes and sentiments of the video.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Provide a single evaluation score that reflects the level of contextual understanding of the prediction compared to the answer.\n"
            "- Assign a contextual understanding score between 0 and 5, where 5 indicates the highest level of contextual understanding.\n"
            "- Base your evaluation on the following scale:\n"
            "  5: PERFECT match in terms of context, themes, and sentiments.\n"
            "  4: Very little misalignments in context or themes, but mostly correct.\n"
            "  3: Mostly correct themes or sentiments, but minor misalignments.\n"
            "  2: Very little correct elements, though parts are relevant.\n"
            "  1: Mostly incorrect context or themes, though some correct elements.\n"
            "  0: COMPLETELY incorrect context or themes with no correct elements."
        )
    },
    "TU": {
        "name": "Temporal Understanding",
        "system": (
            "You are an AI assistant tasked with evaluating the temporal understanding in results for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events or the specific details of an event in the video content."
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video.\n"
            "- Consider synonyms or paraphrases as valid matches, but only if the temporal order and specific details are maintained.\n"
            "- Evaluate the temporal accuracy of the prediction compared to the answer.\n"
            "- Assign a temporal accuracy score between 0 and 5, where 5 indicates the highest level of temporal consistency.\n"
            "- Base your evaluation on the following scale:\n"
            "  5: PERFECT match in terms of correctness, sequence and details.\n"
            "  4: Very little discrepancies in details, but the sequence or event descriptions are mostly correct.\n"
            "  3: Mostly correct depiction of sequences, but minor discrepancies in details.\n"
            "  2: Very little correct elements, though some events are correct.\n"
            "  1: Mostly incorrect sequence or event description, very few correct temporal or contextual elements.\n"
            "  0: COMPLETELY incorrect sequence or event description with no correct temporal or contextual elements."
        )
    }
}


def get_dimension_score(question: str, answer: str, pred: str, dimension: str, 
                        model: str = "llama3.2:latest", max_retries: int = 3) -> int:
    """
    Evaluates a specific dimension of the prediction using Ollama LLM.
    
    Args:
        question: The question asked
        answer: Ground truth answer
        pred: Predicted answer
        dimension: One of 'CI', 'DO', 'CU', 'TU'
        model: Ollama model to use
        max_retries: Number of retry attempts
        
    Returns:
        Score between 0 and 5, or -1 if failed
    """
    if dimension not in DIMENSION_PROMPTS:
        raise ValueError(f"Unknown dimension: {dimension}. Must be one of {list(DIMENSION_PROMPTS.keys())}")
    
    system_prompt = DIMENSION_PROMPTS[dimension]["system"]
    
    user_prompt = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a score where the score is an integer value between 0 and 5, with 5 indicating the highest level of quality. "
        "Provide your evaluation only as a score, an integer between 0 and 5, and ONLY the number. "
        "Do not include any explanations, text, or JSON format. Just return the integer value as a plain number."
    )
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            content = result['message']['content'].strip()
            
            # Extract just the number from the response
            content_clean = ''.join(c for c in content if c.isdigit())
            if content_clean:
                score = int(content_clean[0])  # Take first digit
                if 0 <= score <= 5:
                    return score
            
            # Try to parse the full content
            score = int(content.replace("\n", "").strip())
            if 0 <= score <= 5:
                return score
                
        except Exception as e:
            if attempt < max_retries - 1:
                continue
    
    return -1  # If all retries fail


def evaluate_all_dimensions(question: str, answer: str, pred: str, 
                           model: str = "llama3.2:latest") -> dict:
    """
    Evaluate prediction across all 4 dimensions and compute average.
    
    Returns:
        Dict with CI, DO, CU, TU scores and Avg (weighted average)
    """
    scores = {}
    for dim in ["CI", "DO", "CU", "TU"]:
        scores[dim] = get_dimension_score(question, answer, pred, dim, model)
    
    # Calculate average (only if all scores are valid)
    valid_scores = [s for s in scores.values() if s >= 0]
    if len(valid_scores) == 4:
        scores["Avg"] = round(0.25 * sum(valid_scores), 2)
    else:
        # Partial average if some scores are valid
        if valid_scores:
            scores["Avg"] = round(sum(valid_scores) / len(valid_scores), 2)
        else:
            scores["Avg"] = -1
    
    return scores


def evaluate_csv(input_csv: str, output_csv: str = None, model: str = "llama3.2:latest"):
    """
    Evaluate responses in CSV file using LLM-as-Judge across 4 dimensions.
    
    Dimensions:
        - CI: Contextual Integration (Factual Accuracy)
        - DO: Detail Orientation (Completeness)
        - CU: Contextual Understanding (Narrative alignment)
        - TU: Temporal Understanding (Event sequence)
        - Avg: Weighted average (0.25 * each dimension)
    
    Args:
        input_csv: Path to input CSV file with responses
        output_csv: Path to output CSV file (default: adds _llm_scored suffix)
        model: Ollama model to use for evaluation
    """
    input_path = Path(input_csv)
    
    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_llm_scored.csv"
    
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Model: {model}")
    print(f"\nEvaluating 4 dimensions per response:")
    for dim, info in DIMENSION_PROMPTS.items():
        print(f"  - {dim}: {info['name']}")
    print(f"  - Avg: Weighted average (0.25 * each)")
    
    # Read CSV
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"\nLoaded {len(rows)} samples")
    
    # Add new columns for LLM scores (4 dimensions + Avg for each pipeline)
    dimension_cols = ['CI', 'DO', 'CU', 'TU', 'Avg']
    new_cols = []
    for prefix in ['no_rerank', 'rerank']:
        for dim in dimension_cols:
            new_cols.append(f'{prefix}_{dim}')
    
    new_fieldnames = list(fieldnames) + new_cols
    
    # Process each row
    results = []
    for i, row in enumerate(tqdm(rows, desc="Evaluating with LLM")):
        question = row['question']
        ground_truth = row['ground_truth_answer']
        no_rerank_response = row['no_rerank_response']
        rerank_response = row['rerank_response']
        
        print(f"\n[{i+1}/{len(rows)}] {row['video_id']} - {row['question_type']}")
        
        # Score no_rerank response across all dimensions
        print("  Scoring no_rerank response (CI, DO, CU, TU)...")
        no_rerank_scores = evaluate_all_dimensions(question, ground_truth, no_rerank_response, model)
        print(f"    CI={no_rerank_scores['CI']} DO={no_rerank_scores['DO']} CU={no_rerank_scores['CU']} TU={no_rerank_scores['TU']} Avg={no_rerank_scores['Avg']}")
        
        # Score rerank response across all dimensions
        print("  Scoring rerank response (CI, DO, CU, TU)...")
        rerank_scores = evaluate_all_dimensions(question, ground_truth, rerank_response, model)
        print(f"    CI={rerank_scores['CI']} DO={rerank_scores['DO']} CU={rerank_scores['CU']} TU={rerank_scores['TU']} Avg={rerank_scores['Avg']}")
        
        # Add scores to row
        for dim in dimension_cols:
            row[f'no_rerank_{dim}'] = no_rerank_scores[dim]
            row[f'rerank_{dim}'] = rerank_scores[dim]
        
        results.append(row)
    
    # Write output CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved results to: {output_csv}")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results):
    """Print summary of LLM evaluation scores across all dimensions."""
    n = len(results)
    if n == 0:
        return
    
    dimensions = ['CI', 'DO', 'CU', 'TU', 'Avg']
    
    print("\n" + "="*80)
    print("LLM-AS-JUDGE MULTI-DIMENSION EVALUATION SUMMARY")
    print("="*80)
    print(f"Total samples: {n}")
    print("\nDimensions:")
    print("  CI = Contextual Integration (Factual Accuracy)")
    print("  DO = Detail Orientation (Completeness)")
    print("  CU = Contextual Understanding (Narrative Alignment)")
    print("  TU = Temporal Understanding (Event Sequence)")
    print("  Avg = Weighted Average (0.25 * each dimension)")
    
    # Calculate averages for each dimension and pipeline
    print("\n" + "-"*80)
    print("OVERALL SCORES BY DIMENSION")
    print("-"*80)
    print(f"{'Dimension':<12} {'No Reranker':>12} {'With Reranker':>14} {'Difference':>12}")
    print("-"*80)
    
    for dim in dimensions:
        no_rerank_col = f'no_rerank_{dim}'
        rerank_col = f'rerank_{dim}'
        
        # Get valid scores
        no_rerank_valid = [float(r[no_rerank_col]) for r in results if float(r[no_rerank_col]) >= 0]
        rerank_valid = [float(r[rerank_col]) for r in results if float(r[rerank_col]) >= 0]
        
        no_rerank_avg = sum(no_rerank_valid) / len(no_rerank_valid) if no_rerank_valid else 0
        rerank_avg = sum(rerank_valid) / len(rerank_valid) if rerank_valid else 0
        diff = rerank_avg - no_rerank_avg
        
        diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
        print(f"{dim:<12} {no_rerank_avg:>11.2f}/5 {rerank_avg:>13.2f}/5 {diff_str:>12}")
    
    # Score distribution for Avg
    print("\n" + "-"*80)
    print("AVERAGE SCORE DISTRIBUTION")
    print("-"*80)
    
    for prefix, label in [('no_rerank', 'No Reranker'), ('rerank', 'With Reranker')]:
        col = f'{prefix}_Avg'
        valid_scores = [float(r[col]) for r in results if float(r[col]) >= 0]
        if valid_scores:
            # Bucket into ranges
            buckets = {
                '0.0-1.0': 0, '1.0-2.0': 0, '2.0-3.0': 0, 
                '3.0-4.0': 0, '4.0-5.0': 0
            }
            for s in valid_scores:
                if s < 1.0:
                    buckets['0.0-1.0'] += 1
                elif s < 2.0:
                    buckets['1.0-2.0'] += 1
                elif s < 3.0:
                    buckets['2.0-3.0'] += 1
                elif s < 4.0:
                    buckets['3.0-4.0'] += 1
                else:
                    buckets['4.0-5.0'] += 1
            
            print(f"\n{label}:")
            for bucket, count in buckets.items():
                pct = 100 * count / len(valid_scores)
                bar = '█' * int(pct / 5)
                print(f"  {bucket}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # By question type
    print("\n" + "-"*80)
    print("SCORES BY QUESTION TYPE")
    print("-"*80)
    
    by_type = {}
    for r in results:
        qt = r['question_type']
        if qt not in by_type:
            by_type[qt] = {f'{prefix}_{dim}': [] for prefix in ['no_rerank', 'rerank'] for dim in dimensions}
        
        for prefix in ['no_rerank', 'rerank']:
            for dim in dimensions:
                col = f'{prefix}_{dim}'
                if float(r[col]) >= 0:
                    by_type[qt][col].append(float(r[col]))
    
    for qt in sorted(by_type.keys()):
        print(f"\n{qt}:")
        print(f"  {'Dim':<6} {'No Rerank':>10} {'Rerank':>10} {'Diff':>8}")
        for dim in dimensions:
            no_scores = by_type[qt][f'no_rerank_{dim}']
            re_scores = by_type[qt][f'rerank_{dim}']
            no_avg = sum(no_scores) / len(no_scores) if no_scores else 0
            re_avg = sum(re_scores) / len(re_scores) if re_scores else 0
            diff = re_avg - no_avg
            print(f"  {dim:<6} {no_avg:>9.2f}/5 {re_avg:>9.2f}/5 {diff:>+7.2f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge multi-dimension evaluation for VideoRAG responses")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file path")
    parser.add_argument("--model", "-m", default="llama3.2:latest", 
                        help="Ollama model to use (default: llama3.2:latest)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("VideoRAG Multi-Dimension LLM-as-Judge Evaluation")
    print("="*80)
    
    evaluate_csv(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
