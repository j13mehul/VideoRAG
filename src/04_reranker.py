"""
Re-ranking Utilities for VideoRAG.

This module provides helper functions for the 3-stage retrieval pipeline
and a legacy HierarchicalReranker class for backward compatibility.

PIPELINE STAGES (implemented in 05_query.py):
1. STAGE 1 - Visual Retrieval (Top 15):
   - CLIP visual embeddings + cosine similarity
   
2. STAGE 2 - Textual + BM25 (Top 10):
   - Caption embeddings (60%) + BM25 term matching (40%)
   
3. STAGE 3 - Temporal Context (Top 5):
   - Average 5 visual embeddings (2 left + current + 2 right)
   - Cosine similarity with query for final ranking
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import json
import re

# Import config using importlib (needed due to numeric prefix)
import importlib.util
_config_path = Path(__file__).parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

OUTPUT_DIR = _config.OUTPUT_DIR
LLAVA_MODEL = _config.LLAVA_MODEL


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def tokenize(text: str) -> Set[str]:
    """Simple word tokenization for BM25 scoring."""
    return set(re.findall(r'\w+', text.lower()))


def compute_bm25_score(query_tokens: Set[str], doc_tokens: List[str], 
                       k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute BM25 relevance score.
    
    Args:
        query_tokens: Set of query terms
        doc_tokens: List of document terms
        k1: Term frequency saturation
        b: Length normalization
        
    Returns:
        BM25 score (higher = more relevant)
    """
    if not doc_tokens:
        return 0.0
    
    doc_tf = {}
    for term in doc_tokens:
        doc_tf[term] = doc_tf.get(term, 0) + 1
    
    avg_dl = 20.0
    doc_len = len(doc_tokens)
    score = 0.0
    
    for term in query_tokens:
        if term in doc_tf:
            tf = doc_tf[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            idf = np.log((10 - 1 + 0.5) / (1 + 0.5))
            score += idf * (numerator / denominator)
    
    return score


def get_chunk_index_from_id(chunk_id: str) -> int:
    """Extract chunk index from chunk_id like 'Video_chunk_0005' → 5."""
    if '_chunk_' in chunk_id:
        try:
            return int(chunk_id.split('_chunk_')[-1])
        except:
            return -1
    return -1


def get_video_name_from_id(chunk_id: str) -> str:
    """Extract video name from chunk_id like 'Video_chunk_0005' → 'Video'."""
    if '_chunk_' in chunk_id:
        return chunk_id.rsplit('_chunk_', 1)[0]
    return chunk_id


def load_metadata_mapping(metadata_path: Path = None) -> Dict[str, Dict]:
    """Load metadata.jsonl and create chunk_id to metadata mapping."""
    if metadata_path is None:
        metadata_path = Path(OUTPUT_DIR) / "metadata.jsonl"
    
    chunk_id_to_metadata = {}
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    chunk_id = entry.get('chunk_id', '')
                    if chunk_id:
                        chunk_id_to_metadata[chunk_id] = entry
    
    return chunk_id_to_metadata


def compute_temporal_score(chunk: Dict[str, Any], 
                           chunk_id_to_metadata: Dict[str, Dict],
                           query_embedding: np.ndarray,
                           n_left: int = 2, 
                           n_right: int = 2) -> tuple:
    """
    Compute temporal context score by averaging neighboring embeddings.
    
    Args:
        chunk: Current chunk with chunk_id
        chunk_id_to_metadata: Mapping of chunk_id to metadata
        query_embedding: Query embedding vector
        n_left: Chunks to look before
        n_right: Chunks to look after
        
    Returns:
        Tuple of (temporal_score, num_embeddings_used)
    """
    chunk_id = chunk['chunk_id']
    video_name = get_video_name_from_id(chunk_id)
    current_idx = get_chunk_index_from_id(chunk_id)
    
    if current_idx < 0:
        return chunk.get('visual_score', chunk.get('similarity_score', 0)), 1
    
    # Collect embeddings from temporal neighbors
    embeddings = []
    for offset in range(-n_left, n_right + 1):
        neighbor_idx = current_idx + offset
        if neighbor_idx >= 0:
            neighbor_id = f"{video_name}_chunk_{neighbor_idx:04d}"
            if neighbor_id in chunk_id_to_metadata:
                emb = chunk_id_to_metadata[neighbor_id].get('embedding')
                if emb and len(emb) > 0:
                    embeddings.append(np.array(emb, dtype='float32'))
    
    if not embeddings:
        return chunk.get('visual_score', chunk.get('similarity_score', 0)), 0
    
    # Compute weighted average based on each frame's similarity to query
    query_flat = query_embedding.flatten() if hasattr(query_embedding, 'flatten') else query_embedding
    query_norm = np.linalg.norm(query_flat)
    if query_norm > 0:
        query_flat = query_flat / query_norm
    
    # Calculate similarity weights for each frame
    weights = []
    for emb in embeddings:
        emb_flat = emb.flatten()
        emb_norm = np.linalg.norm(emb_flat)
        if emb_norm > 0:
            emb_flat = emb_flat / emb_norm
            sim = float(np.dot(emb_flat, query_flat))
            # Use softmax-like weighting (shift to positive and amplify differences)
            weights.append(max(0.1, sim + 1.0))  # Shift by 1 to make positive, min 0.1
        else:
            weights.append(0.1)
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Weighted average of embeddings
    temporal_avg = np.zeros_like(embeddings[0].flatten())
    for emb, w in zip(embeddings, weights):
        temporal_avg += w * emb.flatten()
    
    # Normalize the weighted average
    norm = np.linalg.norm(temporal_avg)
    if norm > 0:
        temporal_avg = temporal_avg / norm
    
    # Cosine similarity with query
    temporal_score = float(np.dot(temporal_avg, query_flat))
    
    return temporal_score, len(embeddings)


# =============================================================================
# LEGACY CLASS FOR BACKWARD COMPATIBILITY (used by image_query)
# =============================================================================

class HierarchicalReranker:
    """
    Legacy reranker class for backward compatibility with image_query.
    
    For text queries, the 3-stage pipeline is implemented directly in 05_query.py.
    This class is kept for image-based queries that use the old API.
    """
    
    def __init__(self, ollama_client, clip_extractor):
        """
        Initialize the reranker.
        
        Args:
            ollama_client: OllamaClient for LLAVA (optional, can be None)
            clip_extractor: CLIPFeatureExtractor for embeddings
        """
        self.ollama_client = ollama_client
        self.clip_extractor = clip_extractor
        self.chunk_id_to_metadata = load_metadata_mapping()
        
        # Parameters
        self.temporal_neighbors_left = 2
        self.temporal_neighbors_right = 2
        self.local_weight = 0.55
        self.global_weight = 0.45
        
        # Caches
        self._text_embedding_cache = {}
        self._keyframe_cache = {}
    
    def _load_video_keyframes_metadata(self, video_name: str) -> List[Dict[str, Any]]:
        """Load all keyframe metadata for a specific video."""
        if video_name in self._keyframe_cache:
            return self._keyframe_cache[video_name]
        
        keyframes = []
        for chunk_id, entry in self.chunk_id_to_metadata.items():
            entry_video = Path(entry.get('video_path', '')).name
            if entry_video == video_name:
                keyframes.append({
                    'chunk_id': entry.get('chunk_id', ''),
                    'video_name': video_name,
                    'start_time': entry.get('start_time', 0),
                    'end_time': entry.get('end_time', 0),
                    'keyframes': entry.get('keyframes', []),
                    'caption': entry.get('caption', ''),
                    'embedding': entry.get('embedding', []),
                    'caption_embedding': entry.get('caption_embedding', [])
                })
        
        keyframes.sort(key=lambda x: x['start_time'])
        self._keyframe_cache[video_name] = keyframes
        return keyframes
    
    def _find_temporal_neighbors_by_index(self, chunk: Dict[str, Any], 
                                           n_left: int = 2, 
                                           n_right: int = 2) -> List[np.ndarray]:
        """Find neighboring chunks and return their embeddings."""
        video_name = Path(chunk.get('video_path', '')).name
        chunk_id = chunk.get('chunk_id', '')
        current_embedding = chunk.get('embedding', [])
        
        current_idx = get_chunk_index_from_id(chunk_id)
        
        if not video_name or current_idx < 0:
            if current_embedding and len(current_embedding) > 0:
                return [np.array(current_embedding)]
            return []
        
        all_video_chunks = self._load_video_keyframes_metadata(video_name)
        
        if not all_video_chunks:
            if current_embedding and len(current_embedding) > 0:
                return [np.array(current_embedding)]
            return []
        
        # Build index to embedding map
        chunk_index_to_embedding = {}
        for vc in all_video_chunks:
            vc_id = vc.get('chunk_id', '')
            vc_idx = get_chunk_index_from_id(vc_id)
            emb = vc.get('embedding', [])
            if vc_idx >= 0 and emb and len(emb) > 0:
                chunk_index_to_embedding[vc_idx] = np.array(emb)
        
        # Collect embeddings
        embeddings = []
        for i in range(current_idx - n_left, current_idx):
            if i >= 0 and i in chunk_index_to_embedding:
                embeddings.append(chunk_index_to_embedding[i])
        
        if current_idx in chunk_index_to_embedding:
            embeddings.append(chunk_index_to_embedding[current_idx])
        elif current_embedding and len(current_embedding) > 0:
            embeddings.append(np.array(current_embedding))
        
        for i in range(current_idx + 1, current_idx + n_right + 1):
            if i in chunk_index_to_embedding:
                embeddings.append(chunk_index_to_embedding[i])
        
        return embeddings
    
    def _compute_temporal_embedding_from_precomputed(self, embeddings: List[np.ndarray], 
                                                       query_embedding: np.ndarray = None) -> Optional[np.ndarray]:
        """Compute weighted averaged embedding from pre-computed embeddings.
        
        Uses cosine similarity with query as weights - frames more similar to
        the query contribute more to the temporal context embedding.
        """
        if not embeddings:
            return None
        
        valid_embeddings = [emb.flatten() for emb in embeddings if len(emb) > 0]
        if not valid_embeddings:
            return None
        
        # If no query provided, fall back to simple average
        if query_embedding is None:
            avg_embedding = np.mean(valid_embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding
        
        # Normalize query
        query_flat = query_embedding.flatten()
        query_norm = np.linalg.norm(query_flat)
        if query_norm > 0:
            query_flat = query_flat / query_norm
        
        # Calculate similarity-based weights for each frame
        weights = []
        for emb in valid_embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb_normalized = emb / emb_norm
                sim = float(np.dot(emb_normalized, query_flat))
                # Shift to positive range and ensure minimum weight
                weights.append(max(0.1, sim + 1.0))
            else:
                weights.append(0.1)
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Compute weighted average
        weighted_avg = np.zeros_like(valid_embeddings[0])
        for emb, w in zip(valid_embeddings, weights):
            weighted_avg += w * emb
        
        # Normalize result
        norm = np.linalg.norm(weighted_avg)
        if norm > 0:
            weighted_avg = weighted_avg / norm
        
        return weighted_avg
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _cached_encode_text(self, text: str) -> np.ndarray:
        """Encode text with caching."""
        if text not in self._text_embedding_cache:
            self._text_embedding_cache[text] = self.clip_extractor.encode_text(text)
        return self._text_embedding_cache[text]
    
    def _bm25_score(self, query: str, document: str) -> float:
        """Simple BM25 scoring."""
        if not query or not document:
            return 0.0
        
        query_tokens = tokenize(query)
        doc_tokens = list(tokenize(document))
        score = compute_bm25_score(query_tokens, doc_tokens)
        
        # Normalize to [0, 1]
        max_possible = len(query_tokens) * 2.5
        return min(1.0, max(0.0, score / max_possible))
    
    def local_rerank(self, chunks: List[Dict[str, Any]], query: str, 
                     use_llava: bool = False, llava_sample_size: int = 5) -> List[Dict[str, Any]]:
        """
        Local re-ranking using CLIP similarity and BM25.
        
        Args:
            chunks: Retrieved chunks
            query: User query
            use_llava: Whether to use LLAVA scoring (simplified - uses CLIP instead)
            llava_sample_size: Ignored (kept for API compatibility)
            
        Returns:
            Re-ranked chunks with local_score
        """
        print("\n🔍 LOCAL RE-RANKING...")
        
        for i, chunk in enumerate(chunks):
            # CLIP score (already computed)
            clip_score = chunk.get('similarity_score', 0.5)
            
            # BM25 score
            caption = chunk.get('caption', '')
            bm25_score = self._bm25_score(query, caption)
            
            # Combined local score: 70% CLIP + 30% BM25
            local_score = 0.7 * clip_score + 0.3 * bm25_score
            
            chunk['local_score'] = local_score
            chunk['score_components'] = {
                'clip_score': clip_score,
                'bm25_score': bm25_score
            }
        
        # Sort by local score
        chunks.sort(key=lambda x: x['local_score'], reverse=True)
        
        print(f"   ✅ Local re-ranking complete")
        return chunks
    
    def global_rerank(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Global re-ranking using temporal context.
        
        Args:
            chunks: Chunks with local_score
            query: User query
            
        Returns:
            Chunks with global_score and final_score
        """
        print("\n🌐 GLOBAL RE-RANKING (Temporal Context)...")
        
        if len(chunks) <= 1:
            for chunk in chunks:
                chunk['global_score'] = chunk.get('local_score', 0.5)
                chunk['final_score'] = chunk['global_score']
                chunk['temporal_info'] = {'temporal_similarity': chunk.get('similarity_score', 0.5)}
            return chunks
        
        # Encode query
        query_embedding = self._cached_encode_text(query)
        if query_embedding is not None:
            query_embedding = query_embedding.flatten()
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
        
        print("   📊 Computing temporal context with WEIGHTED averaging...")
        print(f"   Using chunk_index to find 2 left + current + 2 right = 5 embeddings per chunk")
        print(f"   Weights based on cosine similarity with query (more similar → higher weight)")
        
        temporal_scores = []
        temporal_info_list = []
        
        # Debug first chunk
        if chunks:
            first_chunk = chunks[0]
            print(f"   DEBUG: First chunk_id: {first_chunk.get('chunk_id', 'N/A')}")
            print(f"   DEBUG: First chunk has 'embedding': {'embedding' in first_chunk}, len={len(first_chunk.get('embedding', []))}")
        
        for i, chunk in enumerate(chunks):
            temporal_embeddings = self._find_temporal_neighbors_by_index(
                chunk, 
                n_left=self.temporal_neighbors_left, 
                n_right=self.temporal_neighbors_right
            )
            
            if i < 3:
                chunk_idx = get_chunk_index_from_id(chunk.get('chunk_id', ''))
                print(f"   DEBUG chunk {i}: chunk_id={chunk.get('chunk_id')}, chunk_idx={chunk_idx}, temporal_embs={len(temporal_embeddings)}")
            
            temporal_embedding = self._compute_temporal_embedding_from_precomputed(
                temporal_embeddings, query_embedding=query_embedding
            )
            
            if temporal_embedding is not None and query_embedding is not None:
                temporal_sim = self._cosine_similarity(temporal_embedding, query_embedding)
            else:
                temporal_sim = chunk.get('similarity_score', 0.5)
            
            temporal_scores.append(temporal_sim)
            neighbors_found = len(temporal_embeddings) - 1
            temporal_info_list.append({
                'embeddings_used': len(temporal_embeddings),
                'neighbors_found': max(0, neighbors_found),
                'temporal_similarity': temporal_sim
            })
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Processed {i + 1}/{len(chunks)} chunks")
        
        # Normalize scores
        temporal_scores = np.array(temporal_scores)
        if temporal_scores.max() > temporal_scores.min():
            temporal_scores_normalized = (temporal_scores - temporal_scores.min()) / (temporal_scores.max() - temporal_scores.min())
        else:
            temporal_scores_normalized = temporal_scores
        
        # Compute final scores
        print("   📊 Computing final scores...")
        reranked_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            local_score = chunk.get('local_score', chunk.get('similarity_score', 0.5))
            global_score = float(temporal_scores_normalized[i])
            final_score = self.local_weight * local_score + self.global_weight * global_score
            
            chunk_copy['global_score'] = global_score
            chunk_copy['final_score'] = final_score
            chunk_copy['temporal_info'] = temporal_info_list[i]
            chunk_copy['temporal_info']['raw_temporal_similarity'] = float(temporal_scores[i])
            
            reranked_chunks.append(chunk_copy)
        
        reranked_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Summary
        avg_embeddings = np.mean([t['embeddings_used'] for t in temporal_info_list])
        avg_temporal_sim = np.mean(temporal_scores)
        print(f"   📊 Temporal context summary:")
        print(f"      - Average embeddings per chunk: {avg_embeddings:.1f}/5")
        print(f"      - Average temporal similarity: {avg_temporal_sim:.3f}")
        print(f"   ✅ Global re-ranking complete (Temporal Context)")
        
        return reranked_chunks
    
    def hierarchical_rerank(self, chunks: List[Dict[str, Any]], query: str,
                           use_llava: bool = False, llava_sample_size: int = 5) -> List[Dict[str, Any]]:
        """
        Main entry point for hierarchical re-ranking.
        
        Applies Local and Global re-ranking in sequence:
        1. Local: CLIP + BM25 scoring
        2. Global: Temporal context scoring
        
        Args:
            chunks: Initial chunks from retrieval
            query: User query
            use_llava: Ignored (simplified pipeline)
            llava_sample_size: Ignored
            
        Returns:
            Re-ranked chunks with scores
        """
        print("\n" + "="*70)
        print("🔄 HIERARCHICAL RE-RANKING FRAMEWORK (OPTIMIZED)")
        print("="*70)
        print(f"Query: '{query[:100]}...' " if len(query) > 100 else f"Query: '{query}'")
        print(f"Initial candidates: {len(chunks)}")
        print(f"\n📍 LOCAL: RRF | Cross-Encoder | BM25 | LLAVA (top {llava_sample_size})")
        print(f"📍 GLOBAL: Temporal Context (±{self.temporal_neighbors_left}/{self.temporal_neighbors_right} keyframes)")
        print(f"📍 FUSION: {self.local_weight:.0%} Local + {self.global_weight:.0%} Global")
        print(f"⚡ Optimizations: Embedding cache | LLAVA sampling | Query cache")
        
        if not chunks:
            return []
        
        # Enrich chunks with metadata
        for chunk in chunks:
            if chunk['chunk_id'] in self.chunk_id_to_metadata:
                meta = self.chunk_id_to_metadata[chunk['chunk_id']]
                if 'embedding' not in chunk:
                    chunk['embedding'] = meta.get('embedding')
                if 'caption_embedding' not in chunk:
                    chunk['caption_embedding'] = meta.get('caption_embedding')
        
        # Phase 1: Local
        locally_ranked = self.local_rerank(chunks, query, use_llava=use_llava)
        
        # Phase 2: Global
        globally_ranked = self.global_rerank(locally_ranked, query)
        
        # Summary
        print("\n📋 RE-RANKING SUMMARY:")
        print("-" * 50)
        
        for i, chunk in enumerate(globally_ranked[:5]):
            video = Path(chunk.get('video_path', '')).name
            ts = f"{chunk.get('start_time', 0):.1f}s"
            
            scores = chunk.get('score_components', {})
            local_s = chunk.get('local_score', 0)
            global_s = chunk.get('global_score', 0)
            final_s = chunk.get('final_score', 0)
            
            temporal = chunk.get('temporal_info', {})
            
            print(f"   {i+1}. {video} @ {ts}")
            print(f"      Local: CLIP={scores.get('clip_score', 0):.3f} | "
                  f"CrossEnc={scores.get('cross_encoder_score', 0):.3f} | "
                  f"BM25={scores.get('bm25_score', 0):.3f}")
            print(f"      Temporal: {temporal.get('temporal_keyframes', 1)} keyframes | "
                  f"Sim={temporal.get('raw_temporal_similarity', 0):.3f}")
            print(f"      Scores: Local={local_s:.3f} | Global={global_s:.3f} | Final={final_s:.3f}")
        
        print("="*70 + "\n")
        
        return globally_ranked
    
    def clear_cache(self):
        """Clear all caches."""
        self._text_embedding_cache.clear()
        self._keyframe_cache.clear()
