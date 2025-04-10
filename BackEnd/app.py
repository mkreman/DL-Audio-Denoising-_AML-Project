import os
import sys
import subprocess
import tempfile
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from argparse import Namespace

# Path to predict.py
PREDICT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'DL-Audio-Denoising-_AML-Project-Wave-U-Ner-V1', 'src', 'models', 'Wave-U-Net-Pytorch')
)
sys.path.append(PREDICT_DIR)

# Import the main function from predict.py
from predict import main

# Flask app setup
app = Flask(__name__)

# Input/output folders
INPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Inputs'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Outputs'))

app.config['UPLOAD_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Available models
MODELS = {
    "wave_u_net": {
        "name": "Wave-U-Net (Default)",
        "checkpoint": "DL-Audio-Denoising-_AML-Project-Wave-U-Ner-V1/src/models/Wave-U-Net-Pytorch/checkpoints/checkpoint"
    },
    "alternative_model": {
        "name": "Alternative Model",
        "checkpoint": "path/to/alternative/model/checkpoint"  # You'll replace this with the actual path
    }
}

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
        print(f"Error getting duration: {e}")
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
        print(f"Error trimming audio: {e}")
        return False

@app.route('/')
def index():
    # Get info about the last processed file if it exists
    processed_file = request.args.get('processed_file', None)
    original_file = request.args.get('original_file', None)
    error = request.args.get('error', None)
    was_trimmed = request.args.get('trimmed', 'false') == 'true'
    
    return render_template('index.html', 
                          models=MODELS, 
                          processed_file=processed_file,
                          original_file=original_file,
                          error=error,
                          was_trimmed=was_trimmed)

@app.route('/denoise', methods=['POST'])
def denoise():
    if 'file' not in request.files:
        return redirect(url_for('index', error="No file part in the request"))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error="No selected file"))
    
    # Get selected model
    model_id = request.form.get('model', 'wave_u_net')
    if model_id not in MODELS:
        model_id = 'wave_u_net'  # Default to Wave-U-Net if invalid selection
    
    # Create temp file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_input_path = temp_file.name
        file.save(temp_input_path)
    
    # Check audio duration
    duration = get_audio_duration(temp_input_path)
    was_trimmed = False
    
    # Generate filename for the processed audio
    original_filename = secure_filename(file.filename)
    base_filename = os.path.splitext(original_filename)[0]
    input_filename = f"{base_filename}.wav"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    
    if duration and duration > 10.0:
        # Trim to 10 seconds
        trim_success = trim_audio(temp_input_path, input_path)
        if not trim_success:
            os.unlink(temp_input_path)  # Clean up temp file
            return redirect(url_for('index', error="Failed to process audio file"))
        was_trimmed = True
    else:
        # Just copy the file to the input folder
        try:
            with open(input_path, 'wb') as f:
                with open(temp_input_path, 'rb') as tf:
                    f.write(tf.read())
        except Exception as e:
            os.unlink(temp_input_path)  # Clean up temp file
            return redirect(url_for('index', error=f"Error saving file: {str(e)}"))
    
    # Clean up temp file
    os.unlink(temp_input_path)
    
    output_path = app.config['OUTPUT_FOLDER']
    
    try:
        # Setup args for predict.main based on selected model
        checkpoint_path = MODELS[model_id]['checkpoint']
        args = Namespace(
            instruments=['clean'],
            cuda=False,
            features=32,
            load_model=checkpoint_path,
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
            output=output_path,
            patience=10
        )

        # Run prediction
        main(args)

        # Output file is saved as: Outputs/<original_filename>_clean.wav
        cleaned_filename = f"{input_filename}_clean.wav"
        cleaned_path = os.path.join(app.config['OUTPUT_FOLDER'], cleaned_filename)

        if not os.path.exists(cleaned_path):
            return redirect(url_for('index', error="Output file not found"))

        # Redirect back to the index page with the processed filename
        return redirect(url_for('index', 
                              processed_file=cleaned_filename,
                              original_file=input_filename,
                              trimmed=str(was_trimmed).lower()))

    except Exception as e:
        return redirect(url_for('index', error=f"Error during processing: {str(e)}"))

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve the audio file (either original or cleaned)"""
    is_original = request.args.get('original', 'false') == 'true'
    folder = app.config['UPLOAD_FOLDER'] if is_original else app.config['OUTPUT_FOLDER']
    
    filepath = os.path.join(folder, secure_filename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404
        
    return send_file(filepath)

@app.route('/download/<filename>')
def download_file(filename):
    """Download the cleaned audio file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return "File not found", 404
        
    return send_file(
        filepath,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='cleaned_audio.wav'
    )

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))