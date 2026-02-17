"""
Feature extraction module for VideoRAG system.
Handles CLIP embeddings and BLIP captioning.
"""

import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Import from 00_config.py (can't use direct import due to numeric prefix)
import importlib.util
from pathlib import Path as _Path
_config_path = _Path(__file__).parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

CLIP_MODEL_NAME = _config.CLIP_MODEL_NAME
CLIP_PRETRAINED = _config.CLIP_PRETRAINED
OUTPUT_DIR = _config.OUTPUT_DIR
EMBEDDING_DIM = _config.EMBEDDING_DIM


class CLIPFeatureExtractor:
    """Extracts CLIP embeddings from images."""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def encode_image(self, image_path: str, robust: bool = True) -> np.ndarray:
        """Encode a single image to CLIP embedding.
        
        Args:
            image_path: Path to the image file
            robust: If True, applies additional preprocessing for robustness
                   (handles crops, rotations, lighting variations)
        
        Returns:
            Normalized CLIP embedding vector
        
        The robust mode handles:
        - Different image sizes (auto-resized by CLIP preprocessing)
        - Cropped images (CLIP processes center/important regions)
        - Color variations (CLIP is trained on diverse lighting)
        - Orientations (EXIF auto-rotation applied)
        """
        try:
            # Load and convert to RGB
            image = Image.open(image_path)
            
            # Auto-rotate based on EXIF data (handles phone photos with wrong orientation)
            if robust and hasattr(image, '_getexif') and image._getexif() is not None:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            image = image.convert('RGB')
            
            # CLIP's preprocess already handles:
            # - Resizing to 224x224 (handles different sizes)
            # - Center crop (handles different aspect ratios)
            # - Normalization (handles different color intensities)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # L2 normalization makes embeddings invariant to brightness/contrast
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return np.zeros(EMBEDDING_DIM)
    
    def encode_image_from_pil(self, image: Image.Image, robust: bool = True) -> np.ndarray:
        """Encode a PIL Image object directly to CLIP embedding.
        
        Useful for in-memory images or when preprocessing externally.
        
        Args:
            image: PIL Image object
            robust: If True, applies additional preprocessing
        
        Returns:
            Normalized CLIP embedding vector
        """
        try:
            # Auto-rotate based on EXIF data
            if robust and hasattr(image, '_getexif') and image._getexif() is not None:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply CLIP preprocessing
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error encoding PIL image: {e}")
            return np.zeros(EMBEDDING_DIM)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to CLIP embedding."""
        try:
            text_tokens = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return np.zeros(EMBEDDING_DIM)
    
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple images in batches."""
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            
            for path in batch_paths:
                embedding = self.encode_image(path)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class BLIPCaptioner:
    """Generates captions for images using BLIP model with batch processing."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP captioning model on {self.device}...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("BLIP model loaded successfully.")
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image for BLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def generate_caption(self, image_path: str, max_length: int = 50) -> str:
        """Generate a caption for a single image."""
        image = self._load_image(image_path)
        if image is None:
            return "Unable to process image."
        
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
        
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return "Unable to generate caption."
    
    def generate_captions_batch(self, image_paths: List[str], batch_size: int = 8, 
                                 max_length: int = 50) -> List[str]:
        """Generate captions for multiple images using batch processing."""
        captions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load valid images
            for idx, path in enumerate(batch_paths):
                image = self._load_image(path)
                if image is not None:
                    batch_images.append(image)
                    valid_indices.append(idx)
            
            if not batch_images:
                # All images in batch failed to load
                captions.extend(["Unable to process image."] * len(batch_paths))
                continue
            
            try:
                # Process batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                batch_captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
                
                # Map captions back to original positions
                result = ["Unable to process image."] * len(batch_paths)
                for j, valid_idx in enumerate(valid_indices):
                    result[valid_idx] = batch_captions[j]
                
                captions.extend(result)
                
            except Exception as e:
                print(f"Error in batch caption generation: {e}")
                # Fallback to single image processing
                for path in batch_paths:
                    captions.append(self.generate_caption(path, max_length))
        
        return captions


class FeatureExtractor:
    """Combined feature extraction pipeline with CLIP and BLIP (no YOLO)."""
    
    def __init__(self):
        self.clip_extractor = CLIPFeatureExtractor()
        self.blip_captioner = BLIPCaptioner()
    
    def process_chunk(self, chunk_info: Dict[str, Any], precomputed_captions: Dict[str, str] = None) -> Dict[str, Any]:
        """Process a single video chunk to extract all features.
        
        Args:
            chunk_info: Chunk metadata including keyframe paths
            precomputed_captions: Optional dict mapping keyframe paths to BLIP captions
                                  (used when batch processing captions separately)
        """
        chunk_id = chunk_info['chunk_id']
        keyframes = chunk_info.get('keyframes', [])
        
        if not keyframes:
            print(f"No keyframes found for chunk {chunk_id}")
            return chunk_info
        
        # Initialize features
        embeddings = []
        captions = []
        
        # Process each keyframe
        for keyframe_path in keyframes:
            # Extract CLIP visual embedding
            embedding = self.clip_extractor.encode_image(keyframe_path)
            embeddings.append(embedding)
            
            # Get BLIP caption (from precomputed or generate)
            if precomputed_captions and keyframe_path in precomputed_captions:
                caption = precomputed_captions[keyframe_path]
            else:
                caption = self.blip_captioner.generate_caption(keyframe_path)
            captions.append(caption)
        
        # Aggregate visual embeddings across keyframes
        if embeddings:
            # Average embeddings across keyframes
            chunk_embedding = np.mean(embeddings, axis=0)
            # L2 normalize the averaged embedding
            norm = np.linalg.norm(chunk_embedding)
            if norm > 0:
                chunk_embedding = chunk_embedding / norm
        else:
            chunk_embedding = np.zeros(EMBEDDING_DIM)
        
        # Combine BLIP captions - deduplicate similar captions
        unique_captions = []
        for cap in captions:
            if cap not in unique_captions and cap != "Unable to process image." and cap != "Unable to generate caption.":
                unique_captions.append(cap)
        
        if unique_captions:
            combined_caption = " ".join(unique_captions)
        else:
            combined_caption = "No visual description available."
        
        # Generate CLIP text embedding from caption for textual retrieval
        caption_embedding = self.clip_extractor.encode_text(combined_caption)
        
        # Update chunk info with extracted features
        chunk_info.update({
            'embedding': chunk_embedding.tolist(),
            'caption_embedding': caption_embedding.tolist(),
            'caption': combined_caption,
            'keyframe_captions': captions
        })
        
        return chunk_info
    
    def process_all_chunks(self, chunks_metadata_path: str, caption_batch_size: int = 16, max_chunks: int = None) -> List[Dict[str, Any]]:
        """Process all chunks from preprocessing metadata with batch BLIP captioning.
        
        Args:
            chunks_metadata_path: Path to preprocessing metadata JSON
            caption_batch_size: Batch size for BLIP captioning
            max_chunks: Maximum number of chunks to process (for testing). None = all chunks
        """
        # Load chunks metadata
        with open(chunks_metadata_path, 'r') as f:
            chunks = json.load(f)
        
        # Limit chunks for testing if specified
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
            print(f"TEST MODE: Processing only {len(chunks)} chunks (limited from full dataset)")
        else:
            print(f"Processing {len(chunks)} chunks for feature extraction...")
        
        # Collect all keyframe paths
        all_keyframes = []
        keyframe_to_chunk = {}  # Map keyframe path to chunk index
        
        for chunk_idx, chunk_info in enumerate(chunks):
            keyframes = chunk_info.get('keyframes', [])
            for kf_path in keyframes:
                all_keyframes.append(kf_path)
                keyframe_to_chunk[kf_path] = chunk_idx
        
        print(f"Generating BLIP captions for {len(all_keyframes)} keyframes (one by one)...")
        
        # Generate captions ONE BY ONE with progress bar (more reliable than batch)
        start_time = time.time()
        all_captions = []
        for kf_path in tqdm(all_keyframes, desc="BLIP Captioning", unit="img"):
            caption = self.blip_captioner.generate_caption(kf_path)
            all_captions.append(caption)
        
        caption_time = time.time() - start_time
        print(f"✓ BLIP Captioning Complete:")
        print(f"  Total images: {len(all_keyframes)}")
        print(f"  Total time: {caption_time:.1f} seconds ({caption_time/60:.1f} minutes)")
        print(f"  Speed: {len(all_keyframes)/caption_time:.2f} images/second")
        
        # Create mapping from keyframe path to caption
        precomputed_captions = dict(zip(all_keyframes, all_captions))
        
        print("Processing chunks with CLIP embeddings and object detection...")
        
        clip_start = time.time()
        processed_chunks = []
        for chunk_info in tqdm(chunks, desc="CLIP + YOLO"):
            try:
                processed_chunk = self.process_chunk(chunk_info, precomputed_captions)
                processed_chunks.append(processed_chunk)
            except Exception as e:
                print(f"Error processing chunk {chunk_info.get('chunk_id', 'unknown')}: {e}")
                continue
        
        clip_time = time.time() - clip_start
        total_time = time.time() - start_time
        
        # Save processed chunks
        output_path = OUTPUT_DIR / "features_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(processed_chunks, f, indent=2)
        
        print(f"\n✓ Feature extraction complete!")
        print(f"  Processed {len(processed_chunks)} chunks")
        print(f"  Timing:")
        print(f"    BLIP Captioning: {caption_time:.1f}s ({caption_time/60:.1f} min)")
        print(f"    CLIP + YOLO:     {clip_time:.1f}s ({clip_time/60:.1f} min)")
        print(f"    Total:           {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Features saved to: {output_path}")
        
        return processed_chunks


def main():
    """Main feature extraction function."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract features from video chunks')
    parser.add_argument('--max-chunks', type=int, default=None, 
                        help='Maximum chunks to process (for testing). Default: all')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for BLIP captioning. Default: 16')
    args = parser.parse_args()
    
    feature_extractor = FeatureExtractor()
    
    # Path to preprocessing metadata
    metadata_path = OUTPUT_DIR / "preprocessing_metadata.json"
    
    if not metadata_path.exists():
        print(f"Preprocessing metadata not found: {metadata_path}")
        print("Please run preprocess.py first.")
        return
    
    # Extract features from chunks (limited if --max-chunks specified)
    processed_chunks = feature_extractor.process_all_chunks(
        str(metadata_path), 
        caption_batch_size=args.batch_size,
        max_chunks=args.max_chunks
    )
    print(f"Feature extraction complete! Processed {len(processed_chunks)} chunks.")


if __name__ == "__main__":
    main()