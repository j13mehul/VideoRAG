"""
Streamlit Application for VideoRAG System
Allows users to query videos, upload new videos, and view results interactively.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import shutil
import tempfile
from PIL import Image
import base64
import importlib.util

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Helper to load modules with numeric prefixes
def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules using importlib (needed due to numeric prefixes in filenames)
_config = load_module("config", src_dir / "00_config.py")
_preprocess = load_module("preprocess", src_dir / "01_preprocess.py")
_features = load_module("extract_features", src_dir / "02_extract_features.py")
_index = load_module("index", src_dir / "03_index.py")
_query = load_module("query", src_dir / "05_query.py")

# Import classes
VideoRAGQueryProcessor = _query.VideoRAGQueryProcessor
VideoPreprocessor = _preprocess.VideoPreprocessor
FeatureExtractor = _features.FeatureExtractor
VideoIndex = _index.VideoIndex
OUTPUT_DIR = _config.OUTPUT_DIR
DATA_DIR = _config.DATA_DIR

# Page configuration
st.set_page_config(
    page_title="VideoRAG - Video Search & Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .video-result {
        border: 1px solid #ddd;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    .similarity-score {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []

# Helper functions
def initialize_query_processor():
    """Initialize the VideoRAG query processor."""
    try:
        # Get selected index directory
        index_dir = st.session_state.get('current_index_dir', OUTPUT_DIR)
        index_path = index_dir / "faiss_index.bin"
        metadata_path = index_dir / "metadata.jsonl"
        
        if not index_path.exists() or not metadata_path.exists():
            st.warning(f"⚠️ Index not found in {index_dir}")
            return False
        
        with st.spinner(f"Loading VideoRAG system from {index_dir}..."):
            processor = VideoRAGQueryProcessor()
            
            # Load index from selected directory
            processor.video_index.load_index(str(index_path), str(metadata_path))
            processor.index_loaded = True
            
            st.session_state.query_processor = processor
            st.session_state.loaded_index_path = str(index_path)
            
        st.success(f"✅ VideoRAG system loaded from: {index_dir.name}/")
        return True
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def process_uploaded_video(video_file, uploaded_output_dir, clear_existing=False):
    """Process an uploaded video through full pipeline and save to uploaded_output."""
    try:
        # Clear existing uploaded_output folder if requested
        if clear_existing and uploaded_output_dir.exists():
            shutil.rmtree(uploaded_output_dir)
            st.info("🗑️ Cleared previous uploaded data")
        
        # Create uploaded output directories
        uploaded_output_dir.mkdir(exist_ok=True, parents=True)
        uploaded_data_dir = uploaded_output_dir / "videos"
        uploaded_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Save uploaded file
        video_name = video_file.name
        dest_path = uploaded_data_dir / video_name
        with open(dest_path, 'wb') as f:
            f.write(video_file.read())
        
        st.success(f"✅ Video uploaded: {video_name}")
        
        # Process video through pipeline
        with st.spinner("Preprocessing video..."):
            preprocessor = VideoPreprocessor()
            # Temporarily override output directories
            original_output = preprocessor.output_dir
            original_chunks = preprocessor.chunks_dir
            original_keyframes = preprocessor.keyframes_dir
            
            preprocessor.output_dir = uploaded_output_dir
            preprocessor.chunks_dir = uploaded_output_dir / "chunks"
            preprocessor.keyframes_dir = uploaded_output_dir / "keyframes"
            preprocessor.chunks_dir.mkdir(exist_ok=True, parents=True)
            preprocessor.keyframes_dir.mkdir(exist_ok=True, parents=True)
            
            chunks = preprocessor.process_video(dest_path)
            
            # Restore original paths
            preprocessor.output_dir = original_output
            preprocessor.chunks_dir = original_chunks
            preprocessor.keyframes_dir = original_keyframes
            
            st.success(f"✅ Created {len(chunks)} chunks")
        
        with st.spinner("Extracting features..."):
            extractor = FeatureExtractor()
            processed_chunks = []
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                processed_chunk = extractor.process_chunk(chunk)
                processed_chunks.append(processed_chunk)
                progress_bar.progress((i + 1) / len(chunks))
            st.success(f"✅ Extracted features from {len(processed_chunks)} chunks")
        
        with st.spinner("Building index..."):
            # Save features metadata
            features_path = uploaded_output_dir / "features_metadata.json"
            with open(features_path, 'w') as f:
                json.dump(processed_chunks, f, indent=2)
            
            # Build index
            video_index = VideoIndex()
            video_index.build_index(str(features_path))
            
            # Save index to uploaded_output
            index_path = uploaded_output_dir / "faiss_index.bin"
            metadata_path = uploaded_output_dir / "metadata.jsonl"
            video_index.save_index(str(index_path), str(metadata_path))
            
            st.success(f"✅ Index created with {len(processed_chunks)} chunks")
        
        st.balloons()
        st.success("🎉 Video processed successfully! You can now search using the uploaded videos.")
        
        return True
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def display_results(results):
    """Display query results in a formatted way."""
    if not results or 'chunks' not in results:
        st.warning("No results found.")
        return
    
    # Display query
    st.markdown(f"### 🔍 Query: *{results['query']}*")
    st.markdown("---")
    
    # Display LLAVA analysis
    if 'answer' in results and results['answer']:
        st.markdown("### 🤖 AI Analysis")
        with st.expander("View LLAVA Analysis", expanded=True):
            st.markdown(results['answer'])
    
    st.markdown("---")
    
    # Display video chunks
    chunks = results.get('chunks', [])
    
    if chunks:
        # Deduplicate: keep only highest score chunk per video
        video_best_chunks = {}
        for chunk in chunks:
            video_name = chunk['video']
            # Use final_score (from 3-stage pipeline) or fallback to similarity_score
            score = chunk.get('final_score', chunk.get('similarity_score', 0))
            
            best_score = video_best_chunks.get(video_name, {}).get('final_score', 
                         video_best_chunks.get(video_name, {}).get('similarity_score', 0))
            if video_name not in video_best_chunks or score > best_score:
                video_best_chunks[video_name] = chunk
        
        # Convert back to list and sort by score
        deduplicated_chunks = sorted(video_best_chunks.values(), 
                                     key=lambda x: x.get('final_score', x.get('similarity_score', 0)), 
                                     reverse=True)
        
        st.markdown(f"### 📹 Found {len(deduplicated_chunks)} Relevant Video(s)")
        if len(deduplicated_chunks) < len(chunks):
            st.info(f"ℹ️ Showing {len(deduplicated_chunks)} unique videos (deduplicated from {len(chunks)} segments)")
        
        for idx, chunk in enumerate(deduplicated_chunks, 1):
            # Display timestamp prominently
            timestamp = chunk.get('timestamp', 'N/A')
            final_score = chunk.get('final_score', chunk.get('similarity_score', 0)) or 0
            visual_score = chunk.get('visual_score', 0) or 0
            textual_score = chunk.get('textual_score', 0) or 0
            
            st.markdown(f"#### {idx}. {chunk['video']}")
            
            # Build score display - show detailed scores only if available
            score_html = f"<span class='similarity-score'>Score: {final_score:.4f}</span>"
            if visual_score > 0 or textual_score > 0:
                score_html += f" | Visual: {visual_score:.4f} | Textual: {textual_score:.4f}"
            
            st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                    <h4 style='margin: 0; color: #1976d2;'>⏱️ Relevant Segment: {timestamp}</h4>
                    <p style='margin: 0.5rem 0 0 0;'>{score_html}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Find video file path from current data directory
            video_name = chunk['video']
            current_data_dir = st.session_state.get('current_data_dir', DATA_DIR)
            video_path = current_data_dir / video_name
            
            # Display full video with player
            if video_path.exists():
                st.video(str(video_path), start_time=int(float(timestamp.split('s')[0])) if 's' in timestamp else 0)
            else:
                st.warning(f"⚠️ Video file not found: {video_path}")
            
            # Display metadata in expandable section
            with st.expander("📋 View Details", expanded=False):
                if 'caption' in chunk:
                    st.markdown(f"**📝 Scene Description:** {chunk['caption']}")
            
            # Deep Analysis button with unique session-based keys
            search_session_id = st.session_state.get('search_session_id', 'default')
            deep_analysis_key = f"deep_analysis_{search_session_id}_{idx}_{chunk.get('chunk_id', idx)}"
            deep_result_key = f"deep_result_{search_session_id}_{idx}_{chunk.get('chunk_id', idx)}"
            
            # Button to trigger deep analysis
            if st.button(f"🔬 Deep Analysis", key=deep_analysis_key, help="Perform detailed frame-by-frame analysis"):
                with st.spinner("🔍 Performing deep analysis... This may take 30-60 seconds"):
                    try:
                        # Prepare chunk info for deep analysis
                        chunk_for_analysis = {
                            'video': video_name,
                            'video_path': str(video_path),
                            'start_time': float(timestamp.split('s')[0]) if 's' in timestamp else 0,
                            'end_time': float(timestamp.split('-')[1].strip().replace('s', '')) if '-' in timestamp else 3,
                            'chunk_id': chunk.get('chunk_id', f'chunk_{idx}')
                        }
                        
                        # Get original query
                        original_query = results.get('query', 'Analyze this video segment')
                        
                        # Perform deep analysis
                        processor = st.session_state.query_processor
                        deep_result = processor.deep_analysis(chunk_for_analysis, original_query)
                        
                        # Store result in session state
                        st.session_state[deep_result_key] = deep_result
                        
                    except Exception as e:
                        st.error(f"❌ Deep analysis error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            # Display stored deep analysis results if available
            if deep_result_key in st.session_state:
                deep_result = st.session_state[deep_result_key]
                
                if 'error' in deep_result:
                    st.error(f"❌ {deep_result['error']}")
                else:
                    # Display deep analysis results
                    st.success("✅ Deep Analysis Complete!")
                    
                    st.markdown("### 🔬 Detailed Analysis")
                    st.markdown(f"**🎯 Target:** {deep_result['target_segment']}")
                    st.markdown(f"**📊 Analysis Range:** {deep_result['analysis_range']}")
                    st.markdown(f"**📸 Frames Analyzed:** {deep_result['frames_analyzed']}")
                    
                    st.markdown("---")
                    st.markdown("#### 📝 Step-by-Step Breakdown:")
                    st.markdown(deep_result['detailed_analysis'])
                    
                    # Optional: Show extracted frames
                    show_frames_key = f"show_frames_{search_session_id}_{idx}_{chunk.get('chunk_id', idx)}"
                    if st.checkbox(f"Show extracted frames ({deep_result['frames_analyzed']} frames)", key=show_frames_key):
                        st.markdown("##### 📸 Extracted Frames (Chronological Order):")
                        frames_to_show = deep_result.get('extended_frames', [])[:6]  # Limit display
                        
                        if frames_to_show:
                            cols = st.columns(4)
                            for frame_idx, frame_path in enumerate(frames_to_show):
                                with cols[frame_idx % 4]:
                                    if Path(frame_path).exists():
                                        st.image(str(frame_path), caption=f"Frame {frame_idx+1}", use_container_width=True)
                                    else:
                                        st.warning(f"Frame {frame_idx+1} not found")
                        else:
                            st.info("No frames available to display")
            
            st.markdown("---")
    else:
        st.info("No video segments matched your query. Try adjusting your search terms or lowering the confidence threshold.")

# Main UI
def main():
    # Header
    st.title("🎬 VideoRAG - Intelligent Video Search & Analysis")
    st.markdown("*Search through videos using natural language queries powered by AI*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["🔍 Search by Text", "🖼️ Search by Image", "📤 Upload New Videos"],
            index=0
        )
        
        st.markdown("---")
        
        # Index selection for search modes
        if mode in ["🔍 Search by Text", "🖼️ Search by Image"]:
            st.subheader("📂 Index Selection")
            
            # Check which indexes are available
            existing_index = OUTPUT_DIR / "faiss_index.bin"
            uploaded_index = Path('uploaded_output') / "faiss_index.bin"
            
            index_options = []
            if existing_index.exists():
                index_options.append("📁 Existing Index (sampledata)")
            if uploaded_index.exists():
                index_options.append("📤 Uploaded Videos Index")
            
            if not index_options:
                st.warning("⚠️ No indexes found")
                selected_index = None
            else:
                selected_index = st.radio(
                    "Select which index to search:",
                    index_options,
                    index=len(index_options) - 1 if uploaded_index.exists() else 0
                )
            
            st.session_state.selected_index = selected_index
            
            st.markdown("---")
        
        # Configuration
        st.subheader("Search Configuration")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        confidence = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        
        # Check if indexes exist
        existing_index_path = OUTPUT_DIR / "faiss_index.bin"
        existing_metadata_path = OUTPUT_DIR / "metadata.jsonl"
        uploaded_index_path = Path('uploaded_output') / "faiss_index.bin"
        uploaded_metadata_path = Path('uploaded_output') / "metadata.jsonl"
        
        if existing_index_path.exists():
            st.success("✅ Existing index loaded")
            try:
                with open(existing_metadata_path, 'r') as f:
                    lines = f.readlines()
                    st.info(f"📹 {len(lines)} chunks in existing index")
            except:
                pass
        
        if uploaded_index_path.exists():
            st.success("✅ Uploaded index loaded")
            try:
                with open(uploaded_metadata_path, 'r') as f:
                    lines = f.readlines()
                    st.info(f"📤 {len(lines)} chunks in uploaded index")
            except:
                pass
        
        if not existing_index_path.exists() and not uploaded_index_path.exists():
            st.warning("⚠️ No indexes found")
        
        # Initialize button
        if st.button("🔄 Initialize/Reload System"):
            st.session_state.query_processor = None
            initialize_query_processor()
    
    # Main content area
    if mode == "🔍 Search by Text":
        st.header("🔍 Search Videos by Text Query")
        
        # Check if index is selected
        selected_index = st.session_state.get('selected_index')
        if not selected_index:
            st.warning("⚠️ No index available. Please upload videos or ensure the existing index exists.")
            st.stop()
        
        # Determine which index to use
        if "Uploaded" in selected_index:
            index_dir = Path('uploaded_output')
            data_dir = index_dir / "videos"
        else:
            index_dir = OUTPUT_DIR
            data_dir = DATA_DIR
        
        # Check if index selection changed - reload if needed
        previous_index_dir = st.session_state.get('current_index_dir')
        if previous_index_dir != index_dir:
            st.session_state.query_processor = None
            st.info(f"📍 Switched to: {index_dir.name}/")
        
        # Store paths in session state
        st.session_state.current_index_dir = index_dir
        st.session_state.current_data_dir = data_dir
        
        # Initialize processor if needed or if index changed
        if st.session_state.query_processor is None:
            if not initialize_query_processor():
                st.stop()
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="Example: Find videos of a car accident, Show me videos with people fighting, etc.",
            height=100
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Search Options", expanded=False):
            use_hierarchical_rerank = st.checkbox(
                "🔄 Use Hierarchical Re-ranking (Local + Global)", 
                value=True,
                help="Improves result quality using fine-grained semantic analysis (Local) "
                     "and temporal/contextual coherence (Global). May take longer but filters out false positives."
            )
            
            use_llava = st.checkbox(
                "🤖 Use AI analysis (LLAVA) for response generation",
                value=False,
                help="Uses LLAVA multimodal model to generate detailed responses. Requires Ollama to be running. Takes longer on CPU."
            )
            
            if use_hierarchical_rerank:
                st.info("**Hierarchical Re-ranking enabled:**\n"
                       "• **Local**: Each frame scored for semantic relevance\n"
                       "• **Global**: Temporal clusters analyzed, isolated results penalized")
            
            if use_llava:
                st.warning("⚠️ **LLAVA enabled**: Response generation will take longer (especially on CPU).")
        
        # Search button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.results = None
                # Clear all deep analysis results
                for key in list(st.session_state.keys()):
                    if key.startswith('deep_result_'):
                        del st.session_state[key]
                st.rerun()
        
        # Execute search
        if search_button and query:
            # Create a container for progress logs
            progress_container = st.container()
            
            # Store progress logs
            progress_logs = []
            
            def update_progress(stage, message, details=None):
                """Callback to update progress in UI"""
                progress_logs.append({
                    'stage': stage,
                    'message': message,
                    'details': details or []
                })
            
            try:
                # Clear previous deep analysis results when new search is performed
                for key in list(st.session_state.keys()):
                    if key.startswith('deep_result_'):
                        del st.session_state[key]
                
                # Generate unique search session ID
                import time
                st.session_state.search_session_id = str(int(time.time() * 1000))
                
                # Show progress UI
                with progress_container:
                    progress_placeholder = st.empty()
                    
                    def progress_callback(stage, message, details):
                        """Real-time progress update"""
                        update_progress(stage, message, details)
                        
                        # Build progress display
                        progress_html = "<div style='font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; margin: 10px 0;'>"
                        progress_html += "<strong style='color: #569cd6;'>🔄 3-STAGE RERANKING PIPELINE</strong><br><br>"
                        
                        for log in progress_logs:
                            stage_name = log['stage']
                            msg = log['message']
                            
                            # Color coding for different stages
                            if 'STAGE 1' in msg:
                                color = '#4ec9b0'  # teal
                            elif 'STAGE 2' in msg:
                                color = '#dcdcaa'  # yellow
                            elif 'STAGE 3' in msg:
                                color = '#ce9178'  # orange
                            elif '✓' in msg or '✅' in msg:
                                color = '#6a9955'  # green
                            elif '🤖' in msg or '📋' in msg:
                                color = '#c586c0'  # purple
                            else:
                                color = '#d4d4d4'  # default
                            
                            progress_html += f"<span style='color: {color};'>{msg}</span><br>"
                            
                            # Show details for completed stages
                            if log['details'] and stage_name.endswith('_done'):
                                for detail in log['details'][:3]:  # Show top 3
                                    progress_html += f"<span style='color: #808080; margin-left: 20px;'>   {detail}</span><br>"
                        
                        progress_html += "</div>"
                        progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
                    
                    results = st.session_state.query_processor.query(
                        query, 
                        top_k=top_k,
                        use_hierarchical_rerank=use_hierarchical_rerank,
                        use_llava=use_llava,
                        progress_callback=progress_callback
                    )
                    st.session_state.results = results
                    
                    # Clear progress display after completion
                    progress_placeholder.empty()
                    
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
        
        # Display results
        if st.session_state.results:
            # Ensure search_session_id exists for result display
            if 'search_session_id' not in st.session_state:
                import time
                st.session_state.search_session_id = str(int(time.time() * 1000))
            st.markdown("---")
            display_results(st.session_state.results)
    
    elif mode == "🖼️ Search by Image":
        st.header("🖼️ Search Videos by Image")
        
        # Check if index is selected
        selected_index = st.session_state.get('selected_index')
        if not selected_index:
            st.warning("⚠️ No index available. Please upload videos or ensure the existing index exists.")
            st.stop()
        
        # Determine which index to use
        if "Uploaded" in selected_index:
            index_dir = Path('uploaded_output')
            data_dir = index_dir / "videos"
        else:
            index_dir = OUTPUT_DIR
            data_dir = DATA_DIR
        
        # Check if index selection changed - reload if needed
        previous_index_dir = st.session_state.get('current_index_dir')
        if previous_index_dir != index_dir:
            st.session_state.query_processor = None
            st.info(f"📍 Switched to: {index_dir.name}/")
        
        # Store paths in session state
        st.session_state.current_index_dir = index_dir
        st.session_state.current_data_dir = data_dir
        
        # Initialize processor if needed or if index changed
        if st.session_state.query_processor is None:
            if not initialize_query_processor():
                st.stop()
        
        # Image upload
        st.markdown("""
        Upload an image to find similar scenes in videos. For example:
        - 🚗 Upload a car image to find videos with that car
        - 👤 Upload a person's photo to find their appearances
        - 🏢 Upload a building/location to find matching scenes
        """)
        
        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to search for similar content in videos"
        )
        
        # Optional description
        description = st.text_input(
            "Optional: Add a description (helps refine search)",
            placeholder="Example: red car, person wearing blue shirt, building exterior, etc."
        )
        
        # Display uploaded image
        if uploaded_image:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_image, caption="Uploaded Query Image", use_container_width=True)
            with col2:
                st.info(f"📸 Image: {uploaded_image.name}")
                if description:
                    st.info(f"📝 Description: {description}")
        
        # Search options
        st.markdown("#### 🎛️ Search Options")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_llava = st.checkbox("Use AI analysis (LLAVA) for better results", value=False, 
                                   help="Requires Ollama to be running. Provides more contextual results but takes longer.")
        with col_opt2:
            enhance_for_crops = st.checkbox("📐 Enhance for cropped/partial images", value=False,
                                           help="Uses AI to analyze cropped images and improve search accuracy. Recommended for close-ups or partial views.")
        
        # Hierarchical re-ranking option
        use_hierarchical_rerank = st.checkbox(
            "🔄 Use Hierarchical Re-ranking (Local + Global)", 
            value=True,
            help="Improves result quality using fine-grained semantic analysis (Local) "
                 "and temporal/contextual coherence (Global)."
        )
        
        if enhance_for_crops:
            st.info("💡 **Tip for cropped images**: The AI will analyze your image and automatically enhance the search. "
                   "Adding a description further improves accuracy!")
        
        # Search button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            search_button = st.button("🔍 Search", type="primary", disabled=uploaded_image is None)
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.results = None
                # Clear all deep analysis results
                for key in list(st.session_state.keys()):
                    if key.startswith('deep_result_'):
                        del st.session_state[key]
                st.rerun()
        
        # Execute image search
        if search_button and uploaded_image:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix) as tmp_file:
                tmp_file.write(uploaded_image.read())
                tmp_image_path = tmp_file.name
            
            try:
                with st.spinner("🔎 Searching through videos using image..." + (" (with hierarchical re-ranking)" if use_hierarchical_rerank else "")):
                    # Clear previous deep analysis results
                    for key in list(st.session_state.keys()):
                        if key.startswith('deep_result_'):
                            del st.session_state[key]
                    
                    # Generate unique search session ID
                    import time
                    st.session_state.search_session_id = str(int(time.time() * 1000))
                    
                    results = st.session_state.query_processor.query_by_image(
                        tmp_image_path, 
                        description=description,
                        top_k=top_k,
                        use_llava=use_llava,
                        enhance_for_crops=enhance_for_crops,
                        use_hierarchical_rerank=use_hierarchical_rerank
                    )
                    st.session_state.results = results
            except Exception as e:
                st.error(f"❌ Error during image search: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_image_path).unlink()
                except:
                    pass
        
        # Display results
        if st.session_state.results:
            # Ensure search_session_id exists for result display
            if 'search_session_id' not in st.session_state:
                import time
                st.session_state.search_session_id = str(int(time.time() * 1000))
            st.markdown("---")
            
            # Show query image in results if available
            if 'query_image' in st.session_state.results:
                with st.expander("🖼️ View Query Image", expanded=False):
                    try:
                        # Re-upload to display (since temp file is deleted)
                        if uploaded_image:
                            uploaded_image.seek(0)  # Reset file pointer
                            st.image(uploaded_image, caption="Your Query Image", width=300)
                    except:
                        pass
            
            display_results(st.session_state.results)
    
    else:  # Upload mode
        st.header("📤 Upload New Videos")
        
        st.info("Upload videos to add them to the searchable index. Supported formats: MP4, AVI, MOV, MKV")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=['mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 Selected {len(uploaded_files)} video(s)")
            
            if st.button("🚀 Process Videos", type="primary"):
                uploaded_output_dir = Path('uploaded_output')
                success_count = 0
                for idx, video_file in enumerate(uploaded_files):
                    st.markdown(f"### Processing: {video_file.name}")
                    # Clear existing data only for the first video
                    clear_existing = (idx == 0)
                    if process_uploaded_video(video_file, uploaded_output_dir, clear_existing=clear_existing):
                        success_count += 1
                
                st.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} videos!")
                
                if success_count > 0:
                    st.info("💡 Switch to Search mode and select 'Uploaded Videos Index' to search your uploaded videos.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 1rem;'>
            <small>VideoRAG System | Powered by CLIP, YOLO, and LLAVA</small>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
