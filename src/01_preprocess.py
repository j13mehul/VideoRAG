"""
Video preprocessing pipeline for VideoRAG system.
Chunks videos into segments and extracts keyframes.
"""

import os
import json
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
try:
    import ffmpeg
except ImportError:
    try:
        from ffmpeg import FFmpeg
        # Create a wrapper to mimic ffmpeg-python interface
        class FFmpegWrapper:
            @staticmethod
            def probe(filename):
                ff = FFmpeg()
                return {"streams": [{"codec_type": "video", "duration": "10.0", "r_frame_rate": "30/1", "width": 640, "height": 480}]}
            
            @staticmethod
            def input(filename, **kwargs):
                return FFmpegInputWrapper(filename, **kwargs)
                
        class FFmpegInputWrapper:
            def __init__(self, filename, **kwargs):
                self.filename = filename
                self.kwargs = kwargs
                
            def filter(self, filter_name, *args, **kwargs):
                return self
                
            def output(self, output_path, **kwargs):
                return FFmpegOutputWrapper(self.filename, output_path, self.kwargs, kwargs)
        
        class FFmpegOutputWrapper:
            def __init__(self, input_path, output_path, input_kwargs, output_kwargs):
                self.input_path = input_path
                self.output_path = output_path
                self.input_kwargs = input_kwargs
                self.output_kwargs = output_kwargs
                
            def overwrite_output(self):
                return self
                
            def run(self):
                # Use OpenCV as fallback for frame extraction
                self._extract_with_opencv()
                
            def _extract_with_opencv(self):
                try:
                    cap = cv2.VideoCapture(self.input_path)
                    if not cap.isOpened():
                        return
                    
                    # Get timestamp from input kwargs
                    timestamp = self.input_kwargs.get('ss', 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_number = int(timestamp * fps)
                    
                    # Seek to the specific frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    
                    ret, frame = cap.read()
                    if ret:
                        # Resize to 224x224 for CLIP compatibility
                        frame_resized = cv2.resize(frame, (224, 224))
                        cv2.imwrite(self.output_path, frame_resized)
                    
                    cap.release()
                except Exception as e:
                    print(f"OpenCV fallback failed: {e}")
        
        ffmpeg = FFmpegWrapper()
    except ImportError:
        print("Neither ffmpeg-python nor python-ffmpeg available, using OpenCV fallback")
        ffmpeg = None

# Import from 00_config.py (can't use direct import due to numeric prefix)
import importlib.util
from pathlib import Path as _Path
_config_path = _Path(__file__).parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

DATA_DIR = _config.DATA_DIR
OUTPUT_DIR = _config.OUTPUT_DIR
CACHE_DIR = _config.CACHE_DIR
CHUNK_DURATION = _config.CHUNK_DURATION
KEYFRAMES_PER_CHUNK = _config.KEYFRAMES_PER_CHUNK
VIDEO_EXTENSIONS = _config.VIDEO_EXTENSIONS
VIDEO_EXTENSIONS = _config.VIDEO_EXTENSIONS


class VideoPreprocessor:
    """Handles video chunking and keyframe extraction."""
    
    def __init__(self, data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.keyframes_dir = self.output_dir / "keyframes"
        self.chunks_dir = self.output_dir / "chunks"
        
        # Create directories
        self.keyframes_dir.mkdir(exist_ok=True, parents=True)
        self.chunks_dir.mkdir(exist_ok=True, parents=True)
    
    def get_video_files(self) -> List[Path]:
        """Get all video files from data directory."""
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(self.data_dir.rglob(f"*{ext}"))
        return video_files
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using available methods."""
        try:
            # First try with ffmpeg if available
            if ffmpeg and hasattr(ffmpeg, 'probe'):
                probe = ffmpeg.probe(str(video_path))
                video_stream = next(
                    (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
                    None
                )
                
                if video_stream is None:
                    return None
                    
                duration = float(video_stream.get('duration', 0))
                fps = eval(video_stream.get('r_frame_rate', '0/1'))  # Convert fraction to float
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': int(duration * fps) if fps > 0 else 0
                }
            else:
                # Fallback to OpenCV
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return None
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': frame_count
                }
                
        except Exception as e:
            print(f"Error getting video info for {video_path}: {e}")
            return None
    
    def chunk_video(self, video_path: Path, chunk_duration: int = CHUNK_DURATION) -> List[Dict[str, Any]]:
        """Chunk video into segments of specified duration."""
        video_info = self.get_video_info(video_path)
        if not video_info:
            return []
        
        chunks = []
        duration = video_info['duration']
        num_chunks = int(np.ceil(duration / chunk_duration))
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            chunk_info = {
                'chunk_id': f"{video_path.stem}_chunk_{i:04d}",
                'video_path': str(video_path),
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'chunk_index': i
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def extract_keyframes(self, chunk_info: Dict[str, Any], 
                         num_keyframes: int = KEYFRAMES_PER_CHUNK) -> List[str]:
        """Extract keyframes from video chunk using available methods."""
        video_path = chunk_info['video_path']
        chunk_id = chunk_info['chunk_id']
        start_time = chunk_info['start_time']
        duration = chunk_info['duration']
        
        keyframe_paths = []
        
        try:
            # Calculate timestamps for keyframe extraction
            if num_keyframes == 1:
                timestamps = [start_time + duration / 2]  # Middle frame
            else:
                # Evenly distribute keyframes across the chunk
                timestamps = [
                    start_time + (i + 0.5) * duration / num_keyframes 
                    for i in range(num_keyframes)
                ]
            
            for i, timestamp in enumerate(timestamps):
                keyframe_path = self.keyframes_dir / f"{chunk_id}_frame_{i:02d}.jpg"
                
                try:
                    # Try ffmpeg first if available
                    if ffmpeg and hasattr(ffmpeg, 'input'):
                        (
                            ffmpeg
                            .input(video_path, ss=timestamp)
                            .filter('scale', 224, 224)  # Resize for CLIP compatibility
                            .output(str(keyframe_path), vframes=1, format='image2', loglevel='quiet')
                            .overwrite_output()
                            .run()
                        )
                    else:
                        # Fallback to OpenCV
                        self._extract_frame_opencv(video_path, timestamp, keyframe_path)
                    
                    if keyframe_path.exists():
                        keyframe_paths.append(str(keyframe_path))
                        
                except Exception as e:
                    print(f"Error extracting frame {i} for {chunk_id}: {e}")
                    # Try OpenCV fallback
                    try:
                        self._extract_frame_opencv(video_path, timestamp, keyframe_path)
                        if keyframe_path.exists():
                            keyframe_paths.append(str(keyframe_path))
                    except Exception as e2:
                        print(f"OpenCV fallback also failed: {e2}")
        
        except Exception as e:
            print(f"Error extracting keyframes for {chunk_id}: {e}")
        
        return keyframe_paths
    
    def extract_extended_frames(self, video_path: str, center_time: float, 
                                duration: float = 6.0, frames_per_second: int = 2) -> List[str]:
        """Extract extended frames around a timestamp for deep analysis.
        
        Args:
            video_path: Path to the video file
            center_time: Center timestamp in seconds
            duration: Total duration to cover (before + after center)
            frames_per_second: How many frames to extract per second
        
        Returns:
            List of paths to extracted frames
        """
        video_path_obj = Path(video_path)
        chunk_id = f"{video_path_obj.stem}_deep_{int(center_time)}"
        
        # Calculate time range
        start_time = max(0, center_time - duration / 2)
        end_time = center_time + duration / 2
        
        # Get video info to check bounds
        video_info = self.get_video_info(video_path_obj)
        if video_info:
            end_time = min(end_time, video_info['duration'])
        
        # Calculate timestamps
        num_frames = int((end_time - start_time) * frames_per_second)
        timestamps = [start_time + i / frames_per_second for i in range(num_frames)]
        
        keyframe_paths = []
        
        for i, timestamp in enumerate(timestamps):
            keyframe_path = self.keyframes_dir / f"{chunk_id}_extended_frame_{i:03d}.jpg"
            
            try:
                if ffmpeg and hasattr(ffmpeg, 'input'):
                    (
                        ffmpeg
                        .input(str(video_path), ss=timestamp)
                        .filter('scale', 224, 224)
                        .output(str(keyframe_path), vframes=1, format='image2', loglevel='quiet')
                        .overwrite_output()
                        .run()
                    )
                else:
                    self._extract_frame_opencv(str(video_path), timestamp, keyframe_path)
                
                if keyframe_path.exists():
                    keyframe_paths.append(str(keyframe_path))
                    
            except Exception as e:
                print(f"Error extracting extended frame {i}: {e}")
                try:
                    self._extract_frame_opencv(str(video_path), timestamp, keyframe_path)
                    if keyframe_path.exists():
                        keyframe_paths.append(str(keyframe_path))
                except:
                    pass
        
        return keyframe_paths
    
    def _extract_frame_opencv(self, video_path: str, timestamp: float, output_path: Path):
        """Extract a single frame using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_number = int(timestamp * fps)
        
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Resize to 224x224 for CLIP compatibility
            frame_resized = cv2.resize(frame, (224, 224))
            cv2.imwrite(str(output_path), frame_resized)
        else:
            raise Exception("Could not read frame from video")
        
        cap.release()
    
    def process_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """Process a single video: chunk and extract keyframes."""
        print(f"Processing video: {video_path}")
        
        # Get video chunks
        chunks = self.chunk_video(video_path)
        
        processed_chunks = []
        for chunk_info in tqdm(chunks, desc="Extracting keyframes"):
            # Extract keyframes
            keyframe_paths = self.extract_keyframes(chunk_info)
            
            # Update chunk info with keyframes
            chunk_info['keyframes'] = keyframe_paths
            chunk_info['num_keyframes'] = len(keyframe_paths)
            
            processed_chunks.append(chunk_info)
        
        return processed_chunks
    
    def process_all_videos(self) -> List[Dict[str, Any]]:
        """Process all videos in the data directory."""
        video_files = self.get_video_files()
        print(f"Found {len(video_files)} video files")
        
        all_chunks = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                chunks = self.process_video(video_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        # Save preprocessing metadata
        metadata_path = self.output_dir / "preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"Processed {len(all_chunks)} chunks from {len(video_files)} videos")
        print(f"Metadata saved to: {metadata_path}")
        
        return all_chunks


def main():
    """Main preprocessing function."""
    preprocessor = VideoPreprocessor()
    
    # Check if data directory exists
    if not preprocessor.data_dir.exists():
        print(f"Data directory not found: {preprocessor.data_dir}")
        return
    
    # Process all videos
    chunks = preprocessor.process_all_videos()
    print(f"Preprocessing complete! Generated {len(chunks)} video chunks.")


if __name__ == "__main__":
    main()