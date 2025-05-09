import os
import sys
import subprocess
import tempfile
import streamlit as st
from werkzeug.utils import secure_filename
from argparse import Namespace
import gdown
import time

import asyncio
import os
import asyncio
import sys

import types
import torch

# Patch torch.classes to avoid issues with Streamlit's source watcher
torch.classes.__path__ = types.SimpleNamespace(_path=[])

if sys.platform.startswith('linux') or sys.platform == 'darwin':  # Linux or macOS
    try:
        from asyncio import get_event_loop_policy
        import asyncio
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception:
        pass

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="Noise Assassins - Audio Denoiser",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

#https://drive.google.com/file/d/1yqU7jYETAqlJNAYvfbY6o6xmWCwYtQKc/view?usp=drive_link
import streamlit as st

st.markdown("""
<style>
    :root, [data-theme="light"], [data-theme="dark"] {
        --primary-color: #7afff0;
        --text-color-light: #07f5da; /* Purple text */
        --text-color-dark: #07f5da;
        --background-color-light: #0a0a0a; /* White background */
        --background-color-dark: #0a0a0a;
        --border-color-light: #7AC6D2;
        --border-color-dark: #2C2C2C;
    }

    body, .stApp, .block-container {
        background-color: var(--background-color-light) !important;
        color: var(--text-color-light) !important;
    }

    [data-theme="dark"] body, [data-theme="dark"] .stApp, [data-theme="dark"] .block-container {
        background-color: var(--background-color-dark) !important;
        color: var(--text-color-dark) !important;
    }

    .logo {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 1rem;
    }

    .tagline {
        color: inherit;
        font-size: 1.2rem; 
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .upload-container {
        background-color: transparent;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        border: 2px dashed var(--background-color-dark);
    }

    .warning {
        padding: 10px;
        border-radius: 8px;
        background-color: #0a0a0a;
        margin-bottom: 20px;
        color: inherit;
    }

    .team-section {
        margin-top: 3rem;
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color-light);
    }

    .team-members {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
    }

    .team-member {
        background-color: rgba(58, 89, 209, 0.1);
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# PREDICT_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'Wave-U-Net-Pytorch')
# )

# # Step 2: Add to system path
# sys.path.append(PREDICT_DIR)

# # Import the main function from predict.py
# from predict import main
# Create the directory structure
def setup_directories():
    input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Inputs'))
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Outputs'))
    model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
    checkpoint_folder = os.path.abspath(os.path.join(model_folder, 'checkpoints'))  # <-- shared

    # Create separate folders inside checkpoints later
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    return input_folder, output_folder, model_folder, checkpoint_folder

# Unpack as needed
INPUT_FOLDER, OUTPUT_FOLDER, WAVE_U_NET_DIR, CHECKPOINT_DIR = setup_directories()

# Add Wave-U-Net directory to path so we can import prediction module
if WAVE_U_NET_DIR not in sys.path:
    sys.path.append(WAVE_U_NET_DIR)

# Define model configurations with dummy GDrive IDs (replace these with your actual IDs)
MODEL_CONFIGS = {
    "wave_u_net": {
        "name": "Wave-U-Net",
        "gdrive_id": "1yqU7jYETAqlJNAYvfbY6o6xmWCwYtQKc",  # Replace with actual ID
        "checkpoint_filename": "checkpoint_wave_u_net",
        "checkpoint_path": None  # Will be set after download
    },
    "segan": {
        "name": "SEGAN",
        "gdrive_id": "1YHtTxEMqrjmoPtbkdhafdwOwpjJfqjfp",  # Replace with actual ID
        "checkpoint_filename": "checkpoint_segan",
        "checkpoint_path": None  # Will be set after download
    }
}

# Function to download models from Google Drive
@st.cache_resource
def download_models_from_gdrive():
    """Download all model checkpoints from Google Drive"""
    for model_id, model_config in MODEL_CONFIGS.items():
        # Route based on model_id
        if model_id == "wave_u_net":
            model_subdir = os.path.join(CHECKPOINT_DIR, "Wave-U-Net")
        elif model_id == "segan":
            model_subdir = os.path.join(CHECKPOINT_DIR, "SEGAN")
        # else:
        #     model_subdir = CHECKPOINT_DIR  # Default fallback

        os.makedirs(model_subdir, exist_ok=True)

        checkpoint_path = os.path.join(model_subdir, model_config["checkpoint_filename"])
        model_config["checkpoint_path"] = checkpoint_path  # Update config with full path

        if os.path.exists(checkpoint_path):
            continue  # Already downloaded

        try:
            with st.spinner(f"Downloading '{model_config['name']}'..."):
                gdown.download(id=model_config["gdrive_id"], output=checkpoint_path, quiet=False)
            if not os.path.exists(checkpoint_path):
                st.warning(f"❌ Failed to download '{model_config['name']}'")
        except Exception as e:
            st.warning(f"⚠️ Download error for '{model_config['name']}': {e}")
    return MODEL_CONFIGS
    
# Function to get audio duration using ffprobe
def get_audio_duration(file_path):
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout)
        return duration
    except Exception as e:
        st.error(f"Error getting duration: {e}")
        return None

# Function to trim audio to 10 seconds using ffmpeg
def trim_audio(input_path, output_path, duration=10.0):
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-t', str(duration),
        '-acodec', 'pcm_s16le',  # Use PCM format for maximum compatibility
        '-ar', '16000',  # Set sample rate to match model requirements
        '-ac', '1',      # Set to mono for model compatibility
        '-y',            # Overwrite output file if it exists
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        st.error(f"Error trimming audio: {e}")
        return False

def process_audio(uploaded_file, model_checkpoint, model_type):
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_input_path = temp_file.name
            with open(temp_input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

        # Check duration
        duration = get_audio_duration(temp_input_path)
        was_trimmed = False

        # Define filenames and paths
        original_filename = secure_filename(uploaded_file.name)
        base_filename = os.path.splitext(original_filename)[0]
        input_filename = f"{base_filename}.wav"
        input_path = os.path.join(INPUT_FOLDER, input_filename)

        # Process the audio file
        if duration and duration > 10.0:
            # Trim to 10 seconds
            trim_success = trim_audio(temp_input_path, input_path)
            if not trim_success:
                os.unlink(temp_input_path)  # Clean up temp file
                raise Exception("Failed to process audio file")
            was_trimmed = True
        else:
            # Just copy the file to the input folder
            with open(input_path, 'wb') as f:
                with open(temp_input_path, 'rb') as tf:
                    f.write(tf.read())

        os.unlink(temp_input_path)
        # Run different model logic
        if model_type == MODEL_CONFIGS['wave_u_net']['name']:
            try:
                # Path to predict.py
                PREDICT_DIR = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'Wave-U-Net-Pytorch')
                )
                sys.path.append(PREDICT_DIR)

                # Import the main function from predict.py
                from predict import main as predict_main

            except ImportError:
                raise Exception("Wave-U-Net prediction module not found.")

            args = Namespace(
                instruments=['clean'],
                cuda=False,
                features=32,
                load_model=model_checkpoint,
                batch_size=4,
                levels=6,
                depth=1,
                sr=16000,
                channels=1,
                kernel_size=5,
                output_size=2.0,
                strides=4,
                conv_type='gn',
                res='fixed',
                separate=0,
                feature_growth='double',
                input=input_path,
                output=OUTPUT_FOLDER,
                patience=10
            )

            predict_main(args)

            # Output paths
            cleaned_path = os.path.join(OUTPUT_FOLDER, f"{input_filename}_clean.wav")
            noise_path = os.path.join(OUTPUT_FOLDER, f"{input_filename}_noise.wav")

            if not os.path.exists(cleaned_path):
                raise Exception("Clean output not found.")

            return input_path, cleaned_path, was_trimmed

        elif model_type == MODEL_CONFIGS['segan']['name']:
            SEGAN_PREDICT_DIR = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'SEGAN')
            )
            sys.path.append(SEGAN_PREDICT_DIR)

            # Import SEGAN prediction function
            from predict_segan import denoise
            cleaned_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_enhanced.wav")
            denoise(input_path =input_path,output_path = cleaned_path,model_path =model_checkpoint)

            if not os.path.exists(cleaned_path):
                raise Exception("Cleaned segan output not found.")

            return input_path, cleaned_path, was_trimmed

        else:
            raise Exception(f"Unsupported model type: {model_type}")

    except Exception as e:
        raise Exception(f"Error during processing: {str(e)}")
    
# Main app
def main():
    # App header and branding
    st.markdown('<div class="logo">Noise Assassins</div>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Professional Audio Denoising Solutions</p>', unsafe_allow_html=True)
    #st.title("Audio Denoiser")
    st.markdown(
    """
    <div style='text-align: center; font-size: 1.2rem;'>
        Upload a noisy audio file and get a clean, noise-free version instantly
    </div>
    """,
    unsafe_allow_html=True
)
    
    # Download models from Google Drive
    models = download_models_from_gdrive()
    
    # Upload form
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select an audio file:", type=["wav", "mp3", "ogg"])
    st.caption("Note: Files longer than 10 seconds will be automatically trimmed to the first 10 seconds for processing.")
    
    # Model selection
    model_options = {model_config["name"]: model_id for model_id, model_config in models.items()}
    selected_model_name = st.selectbox("Select denoising model:", list(model_options.keys()))
    selected_model_id = model_options[selected_model_name]
    
    process_button = st.button("Upload & Denoise")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'original_file' not in st.session_state:
        st.session_state.original_file = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'was_trimmed' not in st.session_state:
        st.session_state.was_trimmed = False
    
    # Process the uploaded file when the button is clicked
    if process_button and uploaded_file is not None:
        try:
            with st.spinner("Processing audio..."):
                # Get the selected model's checkpoint path
                model_checkpoint = models[selected_model_id]["checkpoint_path"]
                
                # Process the audio
                original_file, processed_file, was_trimmed = process_audio(
                    uploaded_file, model_checkpoint, selected_model_name
                )
                
                # Update session state
                st.session_state.original_file = original_file
                st.session_state.processed_file = processed_file
                st.session_state.was_trimmed = was_trimmed
                st.session_state.processed = True
                
                # Rerun to update UI
                st.rerun()
        
        except Exception as e:
            st.error(str(e))
    
    # Display trimmed warning if applicable
    if st.session_state.was_trimmed:
        st.markdown('<div class="warning">Your audio was longer than 10 seconds and has been trimmed to the first 10 seconds for processing.</div>', unsafe_allow_html=True)
    
    # Display results if processing was successful
    if st.session_state.processed and st.session_state.original_file and st.session_state.processed_file:
        st.header("Denoising Complete!")
        
        # Create two columns for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Audio")
            st.audio(st.session_state.original_file)
        
        with col2:
            st.subheader("Cleaned Audio")
            st.audio(st.session_state.processed_file)
        
        # Download button for cleaned audio
        with open(st.session_state.processed_file, "rb") as file:
            st.download_button(
                label="Download Cleaned Audio",
                data=file,
                file_name="cleaned_audio.wav",
                mime="audio/wav",
            )
    
    # Team section
    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    st.markdown('<div class="team-title">Noise Assassins Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="team-members">
        <div class="team-member">Aalekhya Mukhopadhyay</div>
        <div class="team-member">Ayush Yadav</div>
        <div class="team-member">Kalyani Gohokar</div>
        <div class="team-member">Mayank Nagar</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
