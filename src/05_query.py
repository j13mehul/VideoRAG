"""
Query processing system for VideoRAG.
Handles natural language queries, retrieval, and LLAVA integration via Ollama.

3-STAGE RETRIEVAL PIPELINE:
- Stage 1: Top 15 using Visual embedding (CLIP visual cosine similarity)
- Stage 2: Top 10 using Textual embedding (60%) + BM25 (40%)
- Stage 3: Top 5 using Temporal context averaging (2 left + current + 2 right)

Key Components:
- OllamaClient: Interface to Ollama for LLAVA multimodal analysis
- VideoRAGQueryProcessor: Main query processing with 3-stage reranking
"""

import requests
import json
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import io

# Import from 00_config.py (can't use direct import due to numeric prefix)
import importlib.util
_config_path = Path(__file__).parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

OLLAMA_BASE_URL = _config.OLLAMA_BASE_URL
LLAVA_MODEL = _config.LLAVA_MODEL
TOP_K_RETRIEVAL = _config.TOP_K_RETRIEVAL
CONFIDENCE_THRESHOLD = _config.CONFIDENCE_THRESHOLD
OUTPUT_DIR = _config.OUTPUT_DIR
DATA_DIR = _config.DATA_DIR

# Helper function to load modules with numeric prefixes
def _load_module(name, filename):
    """Load a module from the src directory by filename."""
    module_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import local modules
from importlib.util import spec_from_file_location, module_from_spec

# Load 03_index module
_index_path = Path(__file__).parent / "03_index.py"
_index_spec = spec_from_file_location("index", _index_path)
_index_module = module_from_spec(_index_spec)
_index_spec.loader.exec_module(_index_module)
VideoIndex = _index_module.VideoIndex

# Load 02_extract_features module
_features_path = Path(__file__).parent / "02_extract_features.py"
_features_spec = spec_from_file_location("extract_features", _features_path)
_features_module = module_from_spec(_features_spec)
_features_spec.loader.exec_module(_features_module)
CLIPFeatureExtractor = _features_module.CLIPFeatureExtractor

# Load 04_reranker module
_reranker_path = Path(__file__).parent / "04_reranker.py"
_reranker_spec = spec_from_file_location("reranker", _reranker_path)
_reranker_module = module_from_spec(_reranker_spec)
_reranker_spec.loader.exec_module(_reranker_module)
HierarchicalReranker = _reranker_module.HierarchicalReranker


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                return model_name in available_models
            return False
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def generate_response(self, prompt: str, model: str = LLAVA_MODEL, 
                         images: List[str] = None) -> str:
        """Generate response using Ollama."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            # Add images if provided (base64 encoded)
            if images:
                payload["images"] = images
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=1000  # 5 minutes timeout for CPU-based LLaVA inference
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error generating response from Ollama"
                
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string for Ollama."""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return ""


class VideoRAGQueryProcessor:
    """Main query processing system for VideoRAG."""
    
    def __init__(self):
        self.video_index = VideoIndex()
        self.clip_extractor = CLIPFeatureExtractor()
        self.ollama_client = OllamaClient()
        self.hierarchical_reranker = HierarchicalReranker(self.ollama_client, self.clip_extractor)
        self.index_loaded = False
    
    def load_index(self) -> bool:
        """Load the video index."""
        try:
            self.video_index.load_index()
            self.index_loaded = True
            print("Video index loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Test connection
            response = requests.get(f"{self.ollama_client.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if LLAVA model is available
            if not self.ollama_client.is_model_available(LLAVA_MODEL):
                print(f"Warning: {LLAVA_MODEL} model not found in Ollama.")
                print(f"Available models: {[model['name'] for model in response.json().get('models', [])]}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return False
    
    def retrieve_similar_chunks(self, query: str, top_k: int = TOP_K_RETRIEVAL, 
                                 use_fusion: bool = True, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Retrieve similar video chunks for a query using fusion scoring.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            use_fusion: If True, uses fusion of visual and textual embeddings
            alpha: Weight for visual score (0.5 = equal weighting)
        
        Returns:
            List of retrieved chunks with similarity scores
        """
        if not self.index_loaded:
            if not self.load_index():
                return []
        
        # Encode query with CLIP text encoder
        query_embedding = self.clip_extractor.encode_text(query)
        
        if use_fusion:
            # Use fusion search (visual + textual)
            results = self.video_index.search_fusion(
                query_embedding, 
                top_k=top_k,
                alpha=alpha,
                candidate_multiplier=3
            )
        else:
            # Use visual search only (backward compatible)
            results = self.video_index.search(query_embedding, top_k=top_k)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= CONFIDENCE_THRESHOLD
        ]
        
        return filtered_results
    
    def retrieve_similar_chunks_by_image(self, image_path: str, top_k: int = TOP_K_RETRIEVAL, 
                                         use_hybrid: bool = False, description: str = "") -> List[Dict[str, Any]]:
        """Retrieve similar video chunks for an image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to retrieve
            use_hybrid: If True, combines image and text search for better results (especially for cropped images)
            description: Optional text description to boost hybrid search
        """
        if not self.index_loaded:
            if not self.load_index():
                return []
        
        # Encode image with CLIP
        image_embedding = self.clip_extractor.encode_image(image_path)
        
        # For hybrid search (better for cropped/partial images)
        if use_hybrid and description:
            print("🔀 Using hybrid search (image + text) for better results with cropped/partial images...")
            
            # Encode text description
            text_embedding = self.clip_extractor.encode_text(description)
            
            # Combine embeddings with weighted average (60% image, 40% text)
            import numpy as np
            combined_embedding = 0.6 * image_embedding + 0.4 * text_embedding
            
            # Normalize the combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Search with combined embedding
            results = self.video_index.search(combined_embedding, top_k=top_k)
            print(f"   ✓ Hybrid search combined visual and semantic information")
        else:
            # Standard image-only search
            results = self.video_index.search(image_embedding, top_k=top_k)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= CONFIDENCE_THRESHOLD
        ]
        
        return filtered_results
    
    def prepare_context_for_llava(self, query: str, retrieved_chunks: List[Dict[str, Any]], query_type: str = 'text') -> Dict[str, Any]:
        """Prepare context package for LLAVA."""
        context = {
            'query': query,
            'query_type': query_type,
            'num_chunks': len(retrieved_chunks),
            'chunks': [],
            'all_keyframes': [],
            'summary': ''
        }
        
        # Process each retrieved chunk
        for i, chunk in enumerate(retrieved_chunks):
            chunk_context = {
                'rank': chunk['rank'],
                'chunk_id': chunk['chunk_id'],
                'video_path': Path(chunk['video_path']).name,  # Just filename
                'timestamp': f"{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s",
                'duration': f"{chunk['duration']:.1f}s",
                'similarity_score': chunk['similarity_score'],
                'caption': chunk['caption'],
                'object_counts': chunk['object_counts'],
                'total_objects': chunk['total_objects'],
                'keyframes': chunk['keyframes']
            }
            
            context['chunks'].append(chunk_context)
            
            # Collect all keyframes for LLAVA
            context['all_keyframes'].extend(chunk['keyframes'])
        
        # Create summary
        total_objects = sum(chunk['total_objects'] for chunk in retrieved_chunks)
        all_objects = {}
        for chunk in retrieved_chunks:
            for obj, count in chunk['object_counts'].items():
                if obj not in all_objects:
                    all_objects[obj] = 0
                all_objects[obj] += count
        
        top_objects = sorted(all_objects.items(), key=lambda x: x[1], reverse=True)[:5]
        
        context['summary'] = (
            f"Retrieved {len(retrieved_chunks)} relevant video segments "
            f"with {total_objects} total objects detected. "
            f"Top objects: {', '.join([f'{obj} ({count})' for obj, count in top_objects])}"
        )
        
        return context
    
    def generate_llava_prompt(self, context: Dict[str, Any]) -> str:
        """Generate structured prompt for LLAVA aligned with UCF Surveillance VQA format."""
        query_type = context.get('query_type', 'text')
        
        if query_type == 'image':
            prompt = f"""You are a surveillance video analyst. The user has provided a reference image and wants to find similar scenes in the video database. Analyze the provided video segments and identify which ones contain similar content to the reference image.

User Request: {context['query']}

Context: {context['summary']}
"""
        else:
            prompt = f"""You are a surveillance video analyst. Your task is to analyze the video segment and provide a comprehensive response addressing the user's question.

User Question: {context['query']}

Context: {context['summary']}
"""
        prompt += """

Video Segments Analysis:
"""
        
        for chunk in context['chunks']:
            prompt += f"""
Segment {chunk['rank']} (Similarity: {chunk['similarity_score']:.3f}):
- Video: {chunk['video_path']}
- Time: {chunk['timestamp']} (Duration: {chunk['duration']})
- Scene Description: {chunk['caption']}
- Objects Detected: {chunk['total_objects']} total
"""
            
            if chunk['object_counts']:
                objects_list = [f"{obj}: {count}" for obj, count in chunk['object_counts'].items()]
                prompt += f"- Object Breakdown: {', '.join(objects_list[:5])}\n"  # Show top 5
        
        prompt += f"""

RESPONSE GUIDELINES (Keep response under 300 words):

Based on the video segment analysis, provide a response that covers these aspects as relevant to the user's question:

1. **ENVIRONMENT DESCRIPTION**: Describe the setting (location type, urban/indoor environment, time of day if visible). Mention specific elements like intersections, vehicles (colors, types), pedestrians, and infrastructure visible in the scene.

2. **ACTIONS & EVENTS**: Detail the sequence of actions observed chronologically. Be specific about:
   - Who is involved (people, vehicles, objects)
   - What actions are occurring 
   - The progression of events (beginning, middle, end)
   - Any abnormal or notable behaviors

3. **CAUSE ANALYSIS** (if the question asks about causes): Identify what led to the observed event. List specific factors that contributed to the situation. Reference actions or conditions visible in the footage.

4. **RESULT/IMPACT** (if the question asks about outcomes): Describe what happens as a result of the event:
   - Immediate aftermath and reactions
   - Impact on surrounding environment and people
   - Potential consequences or chain reactions

Be SPECIFIC and DETAILED about what you observe. Reference specific visual elements like vehicle colors, clothing, positions, and movements. Write in a clear, narrative style suitable for surveillance video analysis.

IMPORTANT: Keep your response concise (under 300 words) while covering the relevant aspects thoroughly.
"""
        
        return prompt
    
    def query(self, user_query: str, top_k: int = TOP_K_RETRIEVAL, 
              use_hierarchical_rerank: bool = True,
              use_llava: bool = False,
              progress_callback: callable = None) -> Dict[str, Any]:
        """Process a user query end-to-end with 3-stage reranking pipeline.
        
        Args:
            user_query: Natural language query
            top_k: Number of top results to return (default: 5)
            use_hierarchical_rerank: If True, applies 3-stage re-ranking
            use_llava: If True, uses LLAVA for multimodal response generation
            progress_callback: Optional callback function(stage, message, details) for progress updates
        
        Pipeline:
            Stage 1: Top 15 using Visual embedding (cosine similarity)
            Stage 2: Top 10 using Textual embedding + BM25
            Stage 3: Top 5 using Temporal context averaging (Visual)
        """
        # Helper to report progress
        def report_progress(stage: str, message: str, details: list = None):
            print(message)
            if progress_callback:
                progress_callback(stage, message, details or [])
        
        report_progress("init", f"Processing query: '{user_query}'")
        
        # Check prerequisites only if LLAVA is enabled
        if use_llava and not self.check_ollama_connection():
            return {
                "error": "Ollama is not running or LLAVA model is not available",
                "suggestion": f"Please ensure Ollama is running and {LLAVA_MODEL} model is installed"
            }
        
        # Load metadata for temporal context
        metadata_path = Path(OUTPUT_DIR) / "metadata.jsonl"
        metadata_list = []
        chunk_id_to_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    m = json.loads(line)
                    metadata_list.append(m)
                    chunk_id_to_metadata[m['chunk_id']] = m
        
        # Get query embedding
        query_embedding = self.clip_extractor.encode_text(user_query)
        query_embedding_np = query_embedding.flatten()
        
        if use_hierarchical_rerank:
            report_progress("pipeline", "🔄 3-STAGE RERANKING PIPELINE", [
                "Stage 1: Top 15 → Visual cosine similarity",
                "Stage 2: Top 10 → Textual cosine + BM25",
                "Stage 3: Top 5  → Temporal context (Visual avg)"
            ])
            
            # ================================================================
            # STAGE 1: Top 15 using Visual Embedding
            # ================================================================
            report_progress("stage1", "📊 STAGE 1: Retrieve TOP 15 using VISUAL Embedding...")
            visual_results = self.video_index.search_visual(query_embedding, top_k=15)
            
            if not visual_results:
                return {
                    "query": user_query,
                    "answer": "No relevant video segments found for your query.",
                    "chunks": [],
                    "confidence": 0.0
                }
            
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
            
            stage1_details = [f"{i+1}. {c['chunk_id']} - Visual: {c['visual_score']:.4f}" 
                             for i, c in enumerate(stage1_chunks[:5])]
            report_progress("stage1_done", f"   ✓ Retrieved {len(stage1_chunks)} chunks by visual similarity", stage1_details)
            
            # ================================================================
            # STAGE 2: Top 10 using Textual + BM25
            # ================================================================
            report_progress("stage2", "📊 STAGE 2: Rerank to TOP 10 using TEXTUAL + BM25...")
            
            # Tokenize query for BM25
            query_tokens = set(re.findall(r'\w+', user_query.lower()))
            
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
            
            stage2_details = [f"{i+1}. {c['chunk_id']} - Text: {c['textual_score']:.4f}, BM25: {c['bm25_score']:.4f}" 
                             for i, c in enumerate(stage2_chunks[:5])]
            report_progress("stage2_done", f"   ✓ Selected top 10 by Textual + BM25", stage2_details)
            
            # ================================================================
            # STAGE 3: Top 5 using Temporal Context (Visual Averaging)
            # ================================================================
            report_progress("stage3", "📊 STAGE 3: Rerank to TOP 5 using TEMPORAL CONTEXT...", 
                           ["Averaging 2 left + current + 2 right visual embeddings"])
            
            def get_chunk_index(chunk_id):
                if '_chunk_' in chunk_id:
                    try:
                        return int(chunk_id.split('_chunk_')[-1])
                    except:
                        return -1
                return -1
            
            def get_video_name(chunk_id):
                if '_chunk_' in chunk_id:
                    return chunk_id.rsplit('_chunk_', 1)[0]
                return chunk_id
            
            for chunk in stage2_chunks:
                chunk_id = chunk['chunk_id']
                video_name = get_video_name(chunk_id)
                current_idx = get_chunk_index(chunk_id)
                
                # Collect temporal embeddings
                embeddings = []
                for offset in range(-2, 3):  # -2, -1, 0, 1, 2
                    neighbor_idx = current_idx + offset
                    if neighbor_idx >= 0:
                        neighbor_id = f"{video_name}_chunk_{neighbor_idx:04d}"
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
            retrieved_chunks = stage2_chunks[:top_k]
            
            stage3_details = [f"{i+1}. {c['chunk_id']} - Temporal: {c['temporal_score']:.4f} ({c['temporal_embeddings_count']} embs)" 
                             for i, c in enumerate(retrieved_chunks)]
            report_progress("stage3_done", f"   ✓ Selected top {top_k} by Temporal Context", stage3_details)
            
            # Update final scores and info
            for i, chunk in enumerate(retrieved_chunks):
                chunk['rank'] = i + 1
                chunk['final_score'] = chunk['temporal_score']
                chunk['similarity_score'] = chunk['visual_score']  # For compatibility
                chunk['reranking_info'] = {
                    'stage1_visual': chunk.get('visual_score', 0.0),
                    'stage2_textual': chunk.get('textual_score', 0.0),
                    'stage2_bm25': chunk.get('bm25_score', 0.0),
                    'stage3_temporal': chunk.get('temporal_score', 0.0),
                    'temporal_embeddings': chunk.get('temporal_embeddings_count', 0)
                }
            
            reranking_method = "3-Stage: Visual(15) → Textual+BM25(10) → Temporal(5)"
            report_progress("complete", "✅ Pipeline complete!")
        else:
            # No reranking, just use visual search
            report_progress("simple", "Using visual similarity only (no reranking)")
            retrieved_chunks = self.video_index.search_visual(query_embedding, top_k=top_k)
            for i, chunk in enumerate(retrieved_chunks):
                chunk['rank'] = i + 1
                chunk['final_score'] = chunk.get('visual_score', chunk.get('similarity_score', 0))
            reranking_method = "Visual similarity only"
        
        # Report final results
        final_details = [f"{chunk['rank']}. {Path(chunk['video_path']).name} @ {chunk['start_time']:.1f}s" 
                        for chunk in retrieved_chunks]
        report_progress("results", "📋 FINAL RESULTS:", final_details)
        
        # Collect keyframes for display
        sample_keyframes = []
        for chunk in retrieved_chunks:
            for kf in chunk.get('keyframes', []):
                if Path(kf).exists():
                    sample_keyframes.append(kf)
                if len(sample_keyframes) >= 6:
                    break
            if len(sample_keyframes) >= 6:
                break
        
        # Generate response based on use_llava setting
        if use_llava:
            # Step 4: Prepare context for LLAVA
            context = self.prepare_context_for_llava(user_query, retrieved_chunks)
            
            # Step 5: Generate LLAVA prompt
            llava_prompt = self.generate_llava_prompt(context)
            
            # Step 6: Encode keyframes for LLAVA
            report_progress("llava", "Encoding keyframes for multimodal analysis...")
            encoded_images = []
            llava_keyframes = context['all_keyframes'][:2]  # Limit to 2 images for CPU performance
            
            for keyframe_path in llava_keyframes:
                if Path(keyframe_path).exists():
                    encoded_img = self.ollama_client.encode_image_to_base64(keyframe_path)
                    if encoded_img:
                        encoded_images.append(encoded_img)
            
            # Step 7: Get LLAVA response
            report_progress("llava_gen", "🤖 Generating multimodal response with LLAVA...")
            llava_response = self.ollama_client.generate_response(
                llava_prompt, 
                model=LLAVA_MODEL,
                images=encoded_images
            )
        else:
            # Skip LLAVA - generate simple response from retrieval results
            report_progress("skip_llava", "⏩ Skipping LLAVA analysis (disabled)")
            llava_response = f"Found {len(retrieved_chunks)} relevant video segments matching your query: '{user_query}'."
            encoded_images = []
        
        report_progress("done", "✅ Analysis complete!")
        
        # Step 8: Prepare final result
        result = {
            "query": user_query,
            "answer": llava_response,
            "retrieved_chunks": len(retrieved_chunks),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "video": Path(chunk["video_path"]).name,
                    "timestamp": f"{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s",
                    "final_score": chunk.get("final_score", chunk["similarity_score"]),
                    "fusion_score": chunk.get("similarity_score"),
                    "visual_score": chunk.get("visual_score") or chunk.get("global_visual_score"),
                    "textual_score": chunk.get("textual_score") or chunk.get("local_textual_score"),
                    "reranking_info": chunk.get("reranking_info"),
                    "caption": chunk["caption"],
                    "keyframes": len(chunk["keyframes"])
                }
                for chunk in retrieved_chunks
            ],
            "supporting_keyframes": sample_keyframes,
            "processing_info": {
                "retrieval_method": "3-Stage Pipeline (Visual → Textual+BM25 → Temporal)",
                "reranking_method": reranking_method,
                "three_stage_reranking": use_hierarchical_rerank,
                "total_keyframes_analyzed": len(encoded_images)
            }
        }
        
        return result
    
    def query_by_image(self, image_path: str, description: str = "", top_k: int = TOP_K_RETRIEVAL, 
                       use_llava: bool = True, enhance_for_crops: bool = True,
                       use_hierarchical_rerank: bool = True) -> Dict[str, Any]:
        """Process an image-based query to find similar video segments.
        
        Args:
            image_path: Path to the query image
            description: Optional text description of what to look for
            top_k: Number of top results to retrieve
            use_llava: Whether to use LLAVA for reranking (requires Ollama)
            enhance_for_crops: If True, uses LLAVA to analyze cropped images and improve search
        
        Returns:
            Dictionary containing query results with matched video segments
        """
        print(f"Processing image query: '{image_path}'")
        if description:
            print(f"Description: '{description}'")
        
        # Verify image exists
        if not Path(image_path).exists():
            return {
                "error": f"Image file not found: {image_path}",
                "suggestion": "Please provide a valid image path"
            }
        
        # Enhanced handling for cropped/partial images
        enhanced_description = description
        use_hybrid = False
        
        if enhance_for_crops and self.check_ollama_connection():
            print("🔍 Analyzing query image to improve search accuracy...")
            
            # Use LLAVA to understand what's in the cropped image
            image_b64 = self.ollama_client.encode_image_to_base64(image_path)
            if image_b64:
                analysis_prompt = """Analyze this image carefully and provide a detailed description focusing on:
1. Main subjects/objects (people, vehicles, animals, etc.)
2. Actions or activities taking place
3. Setting/environment (indoor/outdoor, location type)
4. Notable colors, clothing, or distinctive features
5. Any text or signs visible

Provide a concise but descriptive summary (2-3 sentences) that captures the key visual elements."""
                
                try:
                    llava_description = self.ollama_client.generate_response(
                        analysis_prompt,
                        model=LLAVA_MODEL,
                        images=[image_b64]
                    )
                    
                    print(f"   ✓ Image analysis: {llava_description[:100]}...")
                    
                    # Combine with user description if provided
                    if description:
                        enhanced_description = f"{description}. {llava_description}"
                    else:
                        enhanced_description = llava_description
                    
                    use_hybrid = True
                    print("   ✓ Using hybrid search with AI-enhanced description")
                    
                except Exception as e:
                    print(f"   ⚠ Image analysis failed: {e}")
                    use_hybrid = bool(description)
        elif description:
            use_hybrid = True
        
        # Step 1: Retrieve similar chunks using image (with hybrid enhancement if applicable)
        retrieval_k = top_k * 3 if use_hierarchical_rerank else top_k  # Get more candidates for re-ranking
        print(f"Retrieving {retrieval_k} video segment candidates using image...")
        retrieved_chunks = self.retrieve_similar_chunks_by_image(
            image_path, 
            top_k=retrieval_k,
            use_hybrid=use_hybrid,
            description=enhanced_description if use_hybrid else ""
        )
        
        if not retrieved_chunks:
            return {
                "query": f"Image search: {Path(image_path).name}",
                "query_image": image_path,
                "answer": "No relevant video segments found for the provided image.",
                "chunks": [],
                "confidence": 0.0
            }
        
        print(f"Found {len(retrieved_chunks)} initial candidates")
        
        # Step 2: Apply Hierarchical Re-ranking if enabled
        if use_hierarchical_rerank and use_llava and self.check_ollama_connection():
            print("\n🔄 Applying Hierarchical (Local + Global) Re-ranking...")
            query_for_rerank = enhanced_description if enhanced_description else f"Image similar to {Path(image_path).name}"
            reranked_chunks = self.hierarchical_reranker.hierarchical_rerank(
                retrieved_chunks, 
                query_for_rerank,
                use_llava=True
            )
            # Take top_k after re-ranking
            retrieved_chunks = reranked_chunks[:top_k]
            reranking_method = "Hierarchical (Local + Global) Re-ranking"
        else:
            retrieved_chunks = retrieved_chunks[:top_k]
            reranking_method = "CLIP similarity only"
        
        print(f"Selected top {len(retrieved_chunks)} segments after re-ranking")
        
        # Prepare query text
        if description:
            query_text = f"Find video segments similar to this image: {description}"
        else:
            query_text = f"Find video segments similar to the provided reference image"
        
        # If LLAVA is not requested or not available, return just retrieval results
        if not use_llava or not self.check_ollama_connection():
            result = {
                "query": query_text,
                "query_image": image_path,
                "answer": f"Found {len(retrieved_chunks)} video segments with visual similarity to the provided image.",
                "retrieved_chunks": len(retrieved_chunks),
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "video": Path(chunk["video_path"]).name,
                        "timestamp": f"{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s",
                        "similarity_score": chunk["similarity_score"],
                        "caption": chunk["caption"],
                        "keyframes": len(chunk["keyframes"])
                    }
                    for chunk in retrieved_chunks
                ],
                "processing_info": {
                    "retrieval_method": "CLIP Image Embedding + FAISS",
                    "reranking_method": "None (LLAVA not used)"
                }
            }
            return result
        
        # Step 2: Prepare context for LLAVA (with image query type)
        context = self.prepare_context_for_llava(query_text, retrieved_chunks, query_type='image')
        
        # Step 3: Generate LLAVA prompt
        llava_prompt = self.generate_llava_prompt(context)
        
        # Step 4: Encode reference image and keyframes for LLAVA
        print("Encoding images for multimodal analysis...")
        encoded_images = []
        
        # First, add the reference query image
        if Path(image_path).exists():
            encoded_img = self.ollama_client.encode_image_to_base64(image_path)
            if encoded_img:
                encoded_images.append(encoded_img)
        
        # Then add sample keyframes from retrieved chunks
        sample_keyframes = context['all_keyframes'][:5]  # Limit to 5 to leave room for query image
        
        for keyframe_path in sample_keyframes:
            if Path(keyframe_path).exists():
                encoded_img = self.ollama_client.encode_image_to_base64(keyframe_path)
                if encoded_img:
                    encoded_images.append(encoded_img)
        
        # Step 5: Get LLAVA response
        print("Generating multimodal response with LLAVA...")
        llava_response = self.ollama_client.generate_response(
            llava_prompt, 
            model=LLAVA_MODEL,
            images=encoded_images
        )
        
        # Step 6: Prepare final result
        result = {
            "query": query_text,
            "query_image": image_path,
            "answer": llava_response,
            "retrieved_chunks": len(retrieved_chunks),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "video": Path(chunk["video_path"]).name,
                    "timestamp": f"{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s",
                    "similarity_score": chunk.get("final_score", chunk["similarity_score"]),
                    "local_score": chunk.get("local_score"),
                    "global_score": chunk.get("global_score"),
                    "coherence_info": chunk.get("coherence_info"),
                    "caption": chunk["caption"],
                    "keyframes": len(chunk["keyframes"])
                }
                for chunk in retrieved_chunks
            ],
            "supporting_keyframes": sample_keyframes,
            "processing_info": {
                "retrieval_method": "CLIP Image Embedding + FAISS",
                "reranking_method": reranking_method,
                "hierarchical_reranking": use_hierarchical_rerank,
                "total_keyframes_analyzed": len(encoded_images),
                "query_image_included": True
            }
        }
        
        return result
    
    def deep_analysis(self, chunk_info: Dict[str, Any], user_query: str, 
                     extend_duration: float = 6.0, frames_per_second: int = 2) -> Dict[str, Any]:
        """Perform deep analysis on a specific video chunk with extended context.
        
        Args:
            chunk_info: Information about the chunk to analyze (must include video_path, start_time, end_time)
            user_query: Original user query for context
            extend_duration: How many seconds to extend analysis (before + after)
            frames_per_second: Frames to extract per second
        
        Returns:
            Detailed analysis with step-by-step breakdown
        """
        print("\n" + "="*80)
        print("PERFORMING DEEP ANALYSIS")
        print("="*80)
        
        # Check Ollama availability
        if not self.check_ollama_connection():
            return {
                "error": "Ollama is not running or LLAVA model is not available",
                "suggestion": "Deep analysis requires LLAVA model for detailed scene understanding"
            }
        
        # Import preprocessor for frame extraction (using importlib due to numeric prefix)
        _preprocess = _load_module("preprocess", "01_preprocess.py")
        VideoPreprocessor = _preprocess.VideoPreprocessor
        preprocessor = VideoPreprocessor()
        
        # Get video path and timing
        video_path = chunk_info.get('video_path', '')
        if not video_path or not Path(video_path).exists():
            # Try to find video in data directories
            video_name = chunk_info.get('video', '')
            potential_path = DATA_DIR / video_name
            if potential_path.exists():
                video_path = str(potential_path)
            else:
                return {"error": f"Video file not found: {video_name}"}
        
        start_time = float(chunk_info.get('start_time', 0))
        end_time = float(chunk_info.get('end_time', start_time + 3))
        center_time = (start_time + end_time) / 2
        
        print(f"📹 Video: {Path(video_path).name}")
        print(f"🎯 Target segment: {start_time:.1f}s - {end_time:.1f}s")
        print(f"🔍 Extended analysis: ±{extend_duration/2:.1f}s around center")
        print(f"⏱️  Extracting {frames_per_second} frames/second...")
        
        # Extract extended frames
        try:
            extended_frames = preprocessor.extract_extended_frames(
                video_path,
                center_time,
                duration=extend_duration,
                frames_per_second=frames_per_second
            )
            
            if not extended_frames:
                return {"error": "Failed to extract frames for deep analysis"}
            
            print(f"✅ Extracted {len(extended_frames)} frames")
            
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return {"error": f"Frame extraction failed: {str(e)}"}
        
        # Encode frames for LLAVA
        print("📸 Encoding frames for AI analysis...")
        encoded_images = []
        
        for frame_path in extended_frames[:6]:  # Limit to 6 frames max for performance
            if Path(frame_path).exists():
                encoded_img = self.ollama_client.encode_image_to_base64(frame_path)
                if encoded_img:
                    encoded_images.append(encoded_img)
        
        print(f"✅ Encoded {len(encoded_images)} frames")
        
        # Generate detailed analysis prompt aligned with UCF Surveillance VQA format
        prompt = f"""You are a surveillance video analyst performing a DETAILED ANALYSIS of a video segment. Your response should be structured to address the user's question comprehensively.

USER QUERY: {user_query}

VIDEO CONTEXT:
- Video: {Path(video_path).name}
- Target Timestamp: {start_time:.1f}s - {end_time:.1f}s
- Extended Analysis Range: {center_time - extend_duration/2:.1f}s to {center_time + extend_duration/2:.1f}s
- Frames Analyzed: {len(encoded_images)} frames across {extend_duration} seconds

You are viewing a sequence of {len(encoded_images)} frames extracted chronologically from this surveillance video segment. Analyze them carefully.

PROVIDE YOUR ANALYSIS FOLLOWING THIS STRUCTURE (Keep under 300 words total):

**ENVIRONMENT & SETTING**
Describe the scene: location type (intersection, street, indoor, etc.), visible infrastructure, time indicators, and overall environment. Mention specific elements like vehicle types, colors, and positions.

**DETAILED ACTION SEQUENCE**
Describe what happens chronologically:
- BEGINNING: What is the initial state of the scene? Who/what is present?
- DEVELOPMENT: What actions or events unfold? Be specific about movements, interactions.
- OUTCOME: How does the sequence conclude? What is the final state?

**ANOMALY/EVENT ANALYSIS**
If there is an abnormal or notable event:
- What is the detected anomaly or key event?
- What appears to have CAUSED this event? (List specific contributing factors)
- What are the RESULTS or consequences? (Impact on people, vehicles, environment)

**DIRECT ANSWER TO QUERY**
Specifically address: "{user_query}"
Reference specific visual evidence from the frames.

**KEY OBSERVATIONS**
- Notable details: clothing colors, vehicle descriptions, distinctive features
- People involved: positions, actions, movements
- Environmental factors: traffic flow, pedestrian activity, conditions

Be SPECIFIC about what you observe. Reference visual elements like vehicle colors (white car, silver van), clothing descriptions, spatial positions, and movement directions. Write clearly and concisely.
"""
        
        # Get LLAVA detailed analysis
        print("🤖 Generating deep analysis with LLAVA...")
        print("   (This may take 30-60 seconds...)")
        
        try:
            analysis = self.ollama_client.generate_response(
                prompt,
                model=LLAVA_MODEL,
                images=encoded_images
            )
        except Exception as e:
            return {"error": f"LLAVA analysis failed: {str(e)}"}
        
        # Prepare result
        result = {
            "query": user_query,
            "video": Path(video_path).name,
            "target_segment": f"{start_time:.1f}s - {end_time:.1f}s",
            "analysis_range": f"{center_time - extend_duration/2:.1f}s - {center_time + extend_duration/2:.1f}s",
            "frames_analyzed": len(encoded_images),
            "detailed_analysis": analysis,
            "extended_frames": extended_frames,
            "processing_info": {
                "method": "Deep Analysis with Extended Context",
                "frames_per_second": frames_per_second,
                "total_duration": extend_duration,
                "model": LLAVA_MODEL
            }
        }
        
        print("\n✅ Deep analysis complete!")
        return result


def main():
    """Interactive query interface."""
    processor = VideoRAGQueryProcessor()
    
    print("=== VideoRAG Query System ===")
    print("Starting up...")
    
    # Check if index exists
    if not processor.load_index():
        print("Error: Could not load video index. Please run the indexing pipeline first.")
        return
    
    # Check Ollama connection
    if not processor.check_ollama_connection():
        print("Error: Ollama is not available. Please ensure Ollama is running with LLAVA model.")
        return
    
    print("System ready! Enter your queries (or 'quit' to exit):")
    print("Example queries:")
    print("- 'Where did the fight start?'")
    print("- 'How many cars were involved in the collision?'")
    print("- 'Show me scenes with people running'")
    print()
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            result = processor.query(query)
            
            # Display result
            if "error" in result:
                print(f"Error: {result['error']}")
                if "suggestion" in result:
                    print(f"Suggestion: {result['suggestion']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                print(f"\nRetrieved {result['retrieved_chunks']} relevant segments:")
                for chunk in result['chunks']:
                    print(f"  - {chunk['video']} ({chunk['timestamp']}) - Score: {chunk['similarity_score']:.3f}")
                print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()