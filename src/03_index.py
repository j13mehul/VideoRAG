"""
Indexing system for VideoRAG using FAISS vector database.
Builds searchable index from video chunk embeddings (visual + textual) and metadata.
Supports dual-embedding retrieval with fusion scoring.
"""

import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Import from 00_config.py (can't use direct import due to numeric prefix)
import importlib.util
_config_path = Path(__file__).parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

FAISS_INDEX_PATH = _config.FAISS_INDEX_PATH
METADATA_PATH = _config.METADATA_PATH
OUTPUT_DIR = _config.OUTPUT_DIR
EMBEDDING_DIM = _config.EMBEDDING_DIM


class VideoIndex:
    """FAISS-based vector index for video chunks with dual embeddings (visual + textual)."""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.visual_index = None      # FAISS index for visual (CLIP image) embeddings
        self.textual_index = None     # FAISS index for textual (CLIP text from BLIP caption) embeddings
        self.metadata = []
        self.chunk_id_to_idx = {}
        
    def build_index(self, features_metadata_path: str) -> None:
        """Build dual FAISS indices from features metadata."""
        print(f"Loading features metadata from: {features_metadata_path}")
        
        # Load processed chunks with features
        with open(features_metadata_path, 'r') as f:
            chunks = json.load(f)
        
        print(f"Building dual indices for {len(chunks)} chunks...")
        
        # Extract embeddings and metadata
        visual_embeddings = []
        textual_embeddings = []
        metadata = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Get visual embedding (from CLIP image encoder)
            visual_emb = chunk.get('embedding', [])
            if not visual_emb or len(visual_emb) != self.embedding_dim:
                print(f"Warning: Invalid visual embedding for chunk {chunk.get('chunk_id', 'unknown')}")
                visual_emb = [0.0] * self.embedding_dim
            
            # Get textual embedding (from CLIP text encoder on BLIP caption)
            textual_emb = chunk.get('caption_embedding', [])
            if not textual_emb or len(textual_emb) != self.embedding_dim:
                # Fallback: use visual embedding if caption embedding not available
                textual_emb = visual_emb.copy() if isinstance(visual_emb, list) else visual_emb.tolist()
            
            visual_embeddings.append(visual_emb)
            textual_embeddings.append(textual_emb)
            
            # Store metadata (INCLUDING embeddings for reranking)
            metadata_entry = {
                'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                'video_path': chunk.get('video_path', ''),
                'start_time': chunk.get('start_time', 0),
                'end_time': chunk.get('end_time', 0),
                'duration': chunk.get('duration', 0),
                'keyframes': chunk.get('keyframes', []),
                'caption': chunk.get('caption', ''),
                'object_counts': chunk.get('object_counts', {}),
                'total_objects': chunk.get('total_objects', 0),
                'keyframe_captions': chunk.get('keyframe_captions', []),
                'embedding': visual_emb,  # Pre-computed visual embedding for reranking
                'caption_embedding': textual_emb,  # Pre-computed textual embedding for reranking
                'index_id': i  # FAISS index ID
            }
            
            metadata.append(metadata_entry)
            self.chunk_id_to_idx[metadata_entry['chunk_id']] = i
        
        # Convert to numpy arrays
        visual_array = np.array(visual_embeddings, dtype=np.float32)
        textual_array = np.array(textual_embeddings, dtype=np.float32)
        
        # Normalize embeddings (important for cosine similarity)
        faiss.normalize_L2(visual_array)
        faiss.normalize_L2(textual_array)
        
        # Create dual FAISS indices using HNSW for faster approximate search
        # M = number of connections per layer (higher = more accurate, slower build)
        # efConstruction = search depth during construction (higher = better quality)
        M = 32  # Good balance between speed and accuracy
        efConstruction = 64  # Higher values give better recall at build time
        
        print(f"Building FAISS HNSW indices (M={M}, efConstruction={efConstruction})...")
        
        # Create HNSW indices with inner product similarity
        self.visual_index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        self.textual_index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        
        # Set construction-time search depth
        self.visual_index.hnsw.efConstruction = efConstruction
        self.textual_index.hnsw.efConstruction = efConstruction
        
        # Set search-time depth (can be tuned for speed vs accuracy trade-off)
        self.visual_index.hnsw.efSearch = 64  # Higher = more accurate search
        self.textual_index.hnsw.efSearch = 64
        
        self.visual_index.add(visual_array)
        self.textual_index.add(textual_array)
        
        # Store metadata
        self.metadata = metadata
        
        print(f"Dual indices built successfully with {self.visual_index.ntotal} vectors each")
    
    def save_index(self, index_path: str = None, metadata_path: str = None) -> None:
        """Save dual FAISS indices and metadata to disk."""
        if index_path is None:
            index_path = FAISS_INDEX_PATH
        if metadata_path is None:
            metadata_path = METADATA_PATH
        
        # Ensure output directory exists
        Path(index_path).parent.mkdir(exist_ok=True, parents=True)
        Path(metadata_path).parent.mkdir(exist_ok=True, parents=True)
        
        # Save visual FAISS index
        visual_index_path = str(index_path).replace('.bin', '_visual.bin')
        print(f"Saving visual FAISS index to: {visual_index_path}")
        faiss.write_index(self.visual_index, visual_index_path)
        
        # Save textual FAISS index
        textual_index_path = str(index_path).replace('.bin', '_textual.bin')
        print(f"Saving textual FAISS index to: {textual_index_path}")
        faiss.write_index(self.textual_index, textual_index_path)
        
        # Also save combined index for backward compatibility
        print(f"Saving combined index to: {index_path}")
        faiss.write_index(self.visual_index, str(index_path))
        
        # Save metadata as JSONL
        print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            for item in self.metadata:
                f.write(json.dumps(item) + '\n')
        
        print("Dual indices and metadata saved successfully!")
    
    def load_index(self, index_path: str = None, metadata_path: str = None) -> None:
        """Load dual FAISS indices and metadata from disk."""
        if index_path is None:
            index_path = FAISS_INDEX_PATH
        if metadata_path is None:
            metadata_path = METADATA_PATH
        
        # Paths for dual indices
        visual_index_path = str(index_path).replace('.bin', '_visual.bin')
        textual_index_path = str(index_path).replace('.bin', '_textual.bin')
        
        # Try loading dual indices first
        if Path(visual_index_path).exists() and Path(textual_index_path).exists():
            print(f"Loading visual FAISS index from: {visual_index_path}")
            self.visual_index = faiss.read_index(visual_index_path)
            
            print(f"Loading textual FAISS index from: {textual_index_path}")
            self.textual_index = faiss.read_index(textual_index_path)
        else:
            # Fallback: load single index (backward compatibility)
            print(f"Loading FAISS index from: {index_path}")
            if not Path(index_path).exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            self.visual_index = faiss.read_index(str(index_path))
            self.textual_index = self.visual_index  # Use same index for both
            print("Warning: Using single index for both visual and textual search (backward compatibility)")
        
        # Load metadata
        print(f"Loading metadata from: {metadata_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = []
        self.chunk_id_to_idx = {}
        
        with open(metadata_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.metadata.append(item)
                self.chunk_id_to_idx[item['chunk_id']] = item['index_id']
        
        print(f"Loaded indices with {self.visual_index.ntotal} vectors and {len(self.metadata)} metadata entries")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar video chunks using query embedding (visual search only).
        
        For backward compatibility - uses visual index only.
        For fusion search, use search_fusion() instead.
        """
        return self.search_visual(query_embedding, top_k)
    
    def search_visual(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using visual embeddings only."""
        if self.visual_index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search visual index
        scores, indices = self.visual_index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['visual_score'] = float(score)
                result['textual_score'] = 0.0
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def search_textual(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using textual embeddings only."""
        if self.textual_index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search textual index
        scores, indices = self.textual_index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['visual_score'] = 0.0
                result['textual_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def search_fusion(self, query_embedding: np.ndarray, top_k: int = 5, 
                      alpha: float = 0.5, candidate_multiplier: int = 3) -> List[Dict[str, Any]]:
        """Search using fusion of visual and textual embeddings.
        
        Args:
            query_embedding: CLIP text embedding of the query
            top_k: Number of results to return
            alpha: Weight for visual score (1-alpha for textual score)
                   alpha=0.5 means equal weighting
            candidate_multiplier: How many more candidates to retrieve for fusion
        
        Returns:
            List of results sorted by fusion score
        
        Formula: final_score = alpha * visual_score + (1 - alpha) * textual_score
        """
        if self.visual_index is None or self.textual_index is None:
            raise ValueError("Indices not loaded. Call load_index() or build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Get more candidates from both indices
        retrieve_k = top_k * candidate_multiplier
        
        # Search visual index
        visual_scores, visual_indices = self.visual_index.search(query_embedding, retrieve_k)
        
        # Search textual index
        textual_scores, textual_indices = self.textual_index.search(query_embedding, retrieve_k)
        
        # Create score maps for fusion
        visual_score_map = {}
        textual_score_map = {}
        
        for score, idx in zip(visual_scores[0], visual_indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                visual_score_map[idx] = float(score)
        
        for score, idx in zip(textual_scores[0], textual_indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                textual_score_map[idx] = float(score)
        
        # Collect all candidate indices
        all_indices = set(visual_score_map.keys()) | set(textual_score_map.keys())
        
        # Compute fusion scores
        fusion_results = []
        for idx in all_indices:
            v_score = visual_score_map.get(idx, 0.0)
            t_score = textual_score_map.get(idx, 0.0)
            
            # Fusion formula
            fusion_score = alpha * v_score + (1 - alpha) * t_score
            
            result = self.metadata[idx].copy()
            result['similarity_score'] = fusion_score
            result['visual_score'] = v_score
            result['textual_score'] = t_score
            result['fusion_alpha'] = alpha
            fusion_results.append(result)
        
        # Sort by fusion score
        fusion_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Assign ranks and return top_k
        for i, result in enumerate(fusion_results[:top_k]):
            result['rank'] = i + 1
        
        return fusion_results[:top_k]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by chunk ID."""
        if chunk_id in self.chunk_id_to_idx:
            idx = self.chunk_id_to_idx[chunk_id]
            return self.metadata[idx]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.metadata:
            return {"total_chunks": 0}
        
        # Calculate statistics
        total_chunks = len(self.metadata)
        total_duration = sum(item['duration'] for item in self.metadata)
        unique_videos = len(set(item['video_path'] for item in self.metadata))
        
        # Object statistics
        all_objects = {}
        for item in self.metadata:
            for obj, count in item['object_counts'].items():
                if obj not in all_objects:
                    all_objects[obj] = 0
                all_objects[obj] += count
        
        top_objects = sorted(all_objects.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_chunks": total_chunks,
            "total_duration_seconds": total_duration,
            "unique_videos": unique_videos,
            "average_chunk_duration": total_duration / total_chunks if total_chunks > 0 else 0,
            "top_objects": top_objects,
            "total_unique_objects": len(all_objects)
        }


def main():
    """Main indexing function."""
    # Initialize video index
    video_index = VideoIndex()
    
    # Path to features metadata
    features_path = OUTPUT_DIR / "features_metadata.json"
    
    if not features_path.exists():
        print(f"Features metadata not found: {features_path}")
        print("Please run extract_features.py first.")
        return
    
    # Build index
    video_index.build_index(str(features_path))
    
    # Save index
    video_index.save_index()
    
    # Print statistics
    stats = video_index.get_statistics()
    print("\n=== Index Statistics ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total duration: {stats['total_duration_seconds']:.1f} seconds")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"Average chunk duration: {stats['average_chunk_duration']:.1f} seconds")
    print(f"Total unique objects: {stats['total_unique_objects']}")
    
    if stats['top_objects']:
        print("\nTop 10 detected objects:")
        for obj, count in stats['top_objects']:
            print(f"  {obj}: {count}")
    
    print(f"\nIndex saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()