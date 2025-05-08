import os
import io
import time
import torch
import librosa
import requests
import tempfile
import threading
import numpy as np
import soundfile as sf
import gradio as gr
from transformers import AutoModel, logging as trf_logging
from huggingface_hub import login, hf_hub_download, scan_cache_dir

# Increase timeout for transformers HTTP requests
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes timeout

# Enable verbose logging for transformers
trf_logging.set_verbosity_info()

# Get Hugging Face token from environment variable
# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN", "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")
if hf_token:
    print("Logging in to Hugging Face...")
    try:
        login(token=hf_token)
        print("Login successful")
    except Exception as e:
        print(f"Login failed: {e}")
else:
    print("⚠️ No Hugging Face token provided. Anonymous access may be limited.")

# Load model with GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Initialize model variable
model = None

# Define the repository ID
repo_id = "ai4bharat/IndicF5"

# Improved model loading with error handling and cache checking
def load_model_with_retry(max_retries=3, retry_delay=5):
    global model

    # First, check if model is already in cache
    print("Checking if model is in cache...")
    try:
        cache_info = scan_cache_dir()
        model_in_cache = any(repo_id in repo.repo_id for repo in cache_info.repos)
        if model_in_cache:
            print(f"Model {repo_id} found in cache, loading locally...")
            model = AutoModel.from_pretrained(
                repo_id,
                trust_remote_code=True,
                local_files_only=True
            ).to(device)
            print("Model loaded from cache successfully!")
            return
    except Exception as e:
        print(f"Cache check failed: {e}")

    # If not in cache or cache check failed, try loading with retries
    for attempt in range(max_retries):
        try:
            print(f"Loading {repo_id} model (attempt {attempt+1}/{max_retries})...")
            model = AutoModel.from_pretrained(
                repo_id,
                trust_remote_code=True,
                revision="main",
                use_auth_token=hf_token,  # Use token if available
                low_cpu_mem_usage=True    # Reduce memory usage
            ).to(device)

            print(f"Model loaded successfully! Type: {type(model)}")

            # Check model attributes
            model_methods = [method for method in dir(model) if not method.startswith('_') and callable(getattr(model, method))]
            print(f"Available model methods: {model_methods[:10]}...")

            return  # Success, exit function

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff

    # If all attempts failed, try one last time with fallback options
    try:
        print("Trying with fallback options...")
        model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
            revision="main",
            local_files_only=False,
            use_auth_token=hf_token,
            force_download=False,
            resume_download=True
        ).to(device)
        print("Model loaded with fallback options!")
    except Exception as e2:
        print(f"❌ All attempts to load model failed: {e2}")
        print("Will continue without model loaded.")

# Call the improved loading function
load_model_with_retry()

# Advanced audio processing functions
def remove_noise(audio_data, threshold=0.01):
    """Apply simple noise gate to remove low-level noise"""
    if audio_data is None:
        return np.zeros(1000)

    # Convert to numpy if needed
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    # Apply noise gate
    noise_mask = np.abs(audio_data) < threshold
    clean_audio = audio_data.copy()
    clean_audio[noise_mask] = 0

    return clean_audio

def apply_smoothing(audio_data, window_size=5):
    """Apply gentle smoothing to reduce artifacts"""
    if audio_data is None or len(audio_data) < window_size*2:
        return audio_data

    # Simple moving average filter
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(audio_data, kernel, mode='same')

    # Keep original at the edges
    smoothed[:window_size] = audio_data[:window_size]
    smoothed[-window_size:] = audio_data[-window_size:]

    return smoothed

def enhance_audio(audio_data):
    """Process audio to improve quality and reduce noise"""
    if audio_data is None:
        return np.zeros(1000)

    # Ensure numpy array
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    # Ensure correct shape and dtype
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Skip processing if audio is empty or too short
    if audio_data.size < 100:
        return audio_data

    # Check if the audio has reasonable amplitude
    rms = np.sqrt(np.mean(audio_data**2))
    print(f"Initial RMS: {rms}")

    # Apply gain if needed
    if rms < 0.05:  # Very quiet
        target_rms = 0.2
        gain = target_rms / max(rms, 0.0001)
        print(f"Applying gain factor: {gain}")
        audio_data = audio_data * gain

    # Remove DC offset
    audio_data = audio_data - np.mean(audio_data)

    # Apply noise gate to remove low-level noise
    audio_data = remove_noise(audio_data, threshold=0.01)

    # Apply gentle smoothing to reduce artifacts
    audio_data = apply_smoothing(audio_data, window_size=3)

    # Apply soft limiting to prevent clipping
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 0.95:
        audio_data = 0.95 * audio_data / max_amp

    # Apply subtle compression for better audibility
    audio_data = np.tanh(audio_data * 1.1) * 0.9

    return audio_data

# Load audio from URL with improved error handling and retries
def load_audio_from_url(url, max_retries=3):
    print(f"Downloading reference audio from {url}")

    for attempt in range(max_retries):
        try:
            # Use a longer timeout
            response = requests.get(url, timeout=60)  # 60 second timeout

            if response.status_code == 200:
                try:
                    # Save content to a temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file.write(response.content)
                    temp_file.close()
                    print(f"Saved reference audio to temp file: {temp_file.name}")

                    # Try different methods to read the audio file
                    audio_data = None
                    sample_rate = None

                    # Try SoundFile first
                    try:
                        audio_data, sample_rate = sf.read(temp_file.name)
                        print(f"Audio loaded with SoundFile: {sample_rate}Hz, {len(audio_data)} samples")
                    except Exception as sf_error:
                        print(f"SoundFile failed: {sf_error}")

                        # Try librosa as fallback
                        try:
                            audio_data, sample_rate = librosa.load(temp_file.name, sr=None)
                            print(f"Audio loaded with librosa: {sample_rate}Hz, shape={audio_data.shape}")
                        except Exception as lr_error:
                            print(f"Librosa also failed: {lr_error}")

                    # Clean up temp file
                    os.unlink(temp_file.name)

                    if audio_data is not None:
                        # Apply audio enhancement to the reference
                        audio_data = enhance_audio(audio_data)
                        return sample_rate, audio_data

                except Exception as e:
                    print(f"Failed to process audio data: {e}")
            else:
                print(f"Failed to download audio: status code {response.status_code}")

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"Request timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed due to timeout.")
        except Exception as e:
            print(f"Error downloading audio: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    # If we reach here, all attempts failed
    print("⚠️ Returning default silence as reference audio")

    # Try to load a local backup audio if provided
    backup_path = "backup_reference.wav"
    if os.path.exists(backup_path):
        try:
            audio_data, sample_rate = sf.read(backup_path)
            print(f"Loaded backup reference audio: {sample_rate}Hz")
            return sample_rate, audio_data
        except Exception as e:
            print(f"Failed to load backup audio: {e}")

    return 24000, np.zeros(int(24000))  # 1 second of silence at 24kHz

# Split text into chunks for streaming
def split_into_chunks(text, max_length=30):
    """Split text into smaller chunks based on punctuation and length"""
    # First split by sentences
    sentence_markers = ['.', '?', '!', ';', ':', '।', '॥']
    chunks = []
    current = ""

    # Initial coarse splitting by sentence markers
    for char in text:
        current += char
        if char in sentence_markers and current.strip():
            chunks.append(current.strip())
            current = ""

    if current.strip():
        chunks.append(current.strip())

    # Further break down long sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Try splitting by commas for long sentences
            comma_splits = chunk.split(',')
            current_part = ""

            for part in comma_splits:
                if len(current_part) + len(part) <= max_length:
                    if current_part:
                        current_part += ","
                    current_part += part
                else:
                    if current_part:
                        final_chunks.append(current_part.strip())
                    current_part = part

            if current_part:
                final_chunks.append(current_part.strip())

    print(f"Split text into {len(final_chunks)} chunks")
    return final_chunks

# Improved model wrapper with timeout handling
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        print(f"Model wrapper initialized with model type: {type(model)}")

        # Discover the appropriate generation method
        self.generation_method = self._find_generation_method()

    def _find_generation_method(self):
        """Find the appropriate method to generate speech"""
        if self.model is None:
            return None

        # Look for plausible generation methods
        candidates = [
            "generate_speech", "tts", "generate_audio", "synthesize",
            "generate", "forward", "__call__"
        ]

        # Check for methods containing these keywords
        for name in dir(self.model):
            if any(candidate in name.lower() for candidate in candidates):
                print(f"Found potential generation method: {name}")
                return name

        # If nothing specific found, default to __call__
        print("No specific generation method found, will use __call__")
        return "__call__"

    def generate(self, text, ref_audio_path, ref_text, **kwargs):
        """Generate speech with improved error handling and preprocessing"""
        print(f"\n==== MODEL INFERENCE ====")
        print(f"Text to generate: '{text}'")  # Make sure this is the text we want to generate
        print(f"Reference audio path: {ref_audio_path}")

        # Check if model is loaded
        if self.model is None:
            print("⚠️ Model is not loaded. Cannot generate speech.")
            return np.zeros(int(24000))  # Return silence

        # Check if files exist
        if not os.path.exists(ref_audio_path):
            print(f"⚠️ Reference audio file not found")
            return None

        # Try different calling approaches
        result = None
        method_name = self.generation_method if self.generation_method else "__call__"

        # Set up different parameter combinations to try
        param_combinations = [
            # First try: standard keyword parameters
            {"text": text, "ref_audio_path": ref_audio_path, "ref_text": ref_text},
            # Second try: alternative parameter names
            {"text": text, "reference_audio": ref_audio_path, "speaker_text": ref_text},
            # Third try: alternative parameter names 2
            {"text": text, "reference_audio": ref_audio_path, "reference_text": ref_text},
            # Fourth try: just text and audio
            {"text": text, "reference_audio": ref_audio_path},
            # Fifth try: just text
            {"text": text},
            # Sixth try: positional arguments
            {}  # Will use positional below
        ]

        # Try each parameter combination with timeout
        for i, params in enumerate(param_combinations):
            try:
                method = getattr(self.model, method_name)
                print(f"Attempt {i+1}: Calling model.{method_name} with {list(params.keys())} parameters")

                # Set a timeout for inference
                with torch.inference_mode():
                    # For the positional arguments case
                    if not params:
                        print(f"Using positional args with text='{text}'")
                        result = method(text, ref_audio_path, ref_text, **kwargs)
                    else:
                        print(f"Using keyword args with text='{params.get('text')}'")
                        result = method(**params, **kwargs)

                print(f"✓ Call succeeded with parameters: {list(params.keys())}")
                break  # Exit loop if successful

            except Exception as e:
                print(f"✗ Attempt {i+1} failed: {str(e)[:100]}...")
                continue

        # Process the result
        if result is not None:
            # Handle tuple results (might be audio, sample_rate)
            if isinstance(result, tuple):
                result = result[0]  # Extract first element, assuming it's audio

            # Convert torch tensor to numpy if needed
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()

            # Ensure array is 1D
            if hasattr(result, 'shape') and len(result.shape) > 1:
                result = result.flatten()

            # Apply advanced audio processing to improve quality
            result = enhance_audio(result)

            return result
        else:
            print("❌ All inference attempts failed")
            return np.zeros(int(24000))  # Return 1 second of silence as fallback

# Create model wrapper
model_wrapper = ModelWrapper(model) if model is not None else None

# Streaming TTS class with improved audio quality and error handling
class StreamingTTS:
    def __init__(self):
        self.is_generating = False
        self.should_stop = False
        self.temp_dir = None
        self.ref_audio_path = None
        self.output_file = None
        self.all_chunks = []
        self.sample_rate = 24000  # Default sample rate
        self.current_text = ""    # Track current text being processed

        # Create temp directory
        try:
            self.temp_dir = tempfile.mkdtemp()
            print(f"Created temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error creating temp directory: {e}")
            self.temp_dir = "."  # Use current directory as fallback

    def prepare_ref_audio(self, ref_audio, ref_sr):
        """Prepare reference audio with enhanced quality"""
        try:
            if self.ref_audio_path is None:
                self.ref_audio_path = os.path.join(self.temp_dir, "ref_audio.wav")

                # Process the reference audio to ensure clean quality
                ref_audio = enhance_audio(ref_audio)

                # Save the reference audio
                sf.write(self.ref_audio_path, ref_audio, ref_sr, format='WAV', subtype='FLOAT')
                print(f"Saved reference audio to: {self.ref_audio_path}")

                # Verify file was created
                if os.path.exists(self.ref_audio_path):
                    print(f"Reference audio saved successfully: {os.path.getsize(self.ref_audio_path)} bytes")
                else:
                    print("⚠️ Failed to create reference audio file!")

            # Create output file
            if self.output_file is None:
                self.output_file = os.path.join(self.temp_dir, "output.wav")
                print(f"Output will be saved to: {self.output_file}")
        except Exception as e:
            print(f"Error preparing reference audio: {e}")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir:
            try:
                if os.path.exists(self.ref_audio_path):
                    os.remove(self.ref_audio_path)
                if os.path.exists(self.output_file):
                    os.remove(self.output_file)
                os.rmdir(self.temp_dir)
                self.temp_dir = None
                print("Cleaned up temporary files")
            except Exception as e:
                print(f"Error cleaning up: {e}")

    def generate(self, text, ref_audio, ref_sr, ref_text):
        """Start generation in a new thread with validation"""
        if self.is_generating:
            print("Already generating speech, please wait")
            return

        # Store the text for verification
        self.current_text = text
        print(f"Setting current text to: '{self.current_text}'")

        # Check model is loaded
        if model_wrapper is None or model is None:
            print("⚠️ Model is not loaded. Cannot generate speech.")
            return

        self.is_generating = True
        self.should_stop = False
        self.all_chunks = []

        # Start in a new thread
        threading.Thread(
            target=self._process_streaming,
            args=(text, ref_audio, ref_sr, ref_text),
            daemon=True
        ).start()

    def _process_streaming(self, text, ref_audio, ref_sr, ref_text):
        """Process text in chunks with high-quality audio generation"""
        try:
            # Double check text matches what we expect
            if text != self.current_text:
                print(f"⚠️ Text mismatch detected! Expected: '{self.current_text}', Got: '{text}'")
                # Use the stored text to be safe
                text = self.current_text

            # Prepare reference audio
            self.prepare_ref_audio(ref_audio, ref_sr)

            # Print the text we're actually going to process
            print(f"Processing text: '{text}'")

            # Split text into smaller chunks for faster processing
            chunks = split_into_chunks(text)
            print(f"Processing {len(chunks)} chunks")

            combined_audio = None
            total_start_time = time.time()

            # Process each chunk
            for i, chunk in enumerate(chunks):
                if self.should_stop:
                    print("Stopping generation as requested")
                    break

                chunk_start = time.time()
                print(f"Processing chunk {i+1}/{len(chunks)}: '{chunk}'")

                # Generate speech for this chunk
                try:
                    # Set timeout for inference
                    chunk_timeout = 30  # 30 seconds timeout per chunk

                    with torch.inference_mode():
                        # Explicitly pass the chunk text
                        chunk_audio = model_wrapper.generate(
                            text=chunk,  # Make sure we're using the current chunk
                            ref_audio_path=self.ref_audio_path,
                            ref_text=ref_text
                        )

                        if chunk_audio is None or (hasattr(chunk_audio, 'size') and chunk_audio.size == 0):
                            print("⚠️ Empty audio returned for this chunk")
                            chunk_audio = np.zeros(int(24000 * 0.5))  # 0.5s silence

                    # Process the audio to improve quality
                    chunk_audio = enhance_audio(chunk_audio)

                    chunk_time = time.time() - chunk_start
                    print(f"✓ Chunk {i+1} processed in {chunk_time:.2f}s")

                    # Add small silence between chunks
                    silence = np.zeros(int(24000 * 0.1))  # 0.1s silence
                    chunk_audio = np.concatenate([chunk_audio, silence])

                    # Add to our collection
                    self.all_chunks.append(chunk_audio)

                    # Combine all chunks so far
                    if combined_audio is None:
                        combined_audio = chunk_audio
                    else:
                        combined_audio = np.concatenate([combined_audio, chunk_audio])

                    # Process combined audio for consistent quality
                    processed_audio = enhance_audio(combined_audio)

                    # Write intermediate output
                    sf.write(self.output_file, processed_audio, 24000, format='WAV', subtype='FLOAT')

                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)[:100]}")
                    continue

            total_time = time.time() - total_start_time
            print(f"Total generation time: {total_time:.2f}s")

        except Exception as e:
            print(f"Error in streaming TTS: {str(e)[:200]}")
            # Try to write whatever we have so far
            if len(self.all_chunks) > 0:
                try:
                    combined = np.concatenate(self.all_chunks)
                    sf.write(self.output_file, combined, 24000, format='WAV', subtype='FLOAT')
                    print("Saved partial output")
                except Exception as e2:
                    print(f"Failed to save partial output: {e2}")
        finally:
            self.is_generating = False
            print("Generation complete")

    def get_current_audio(self):
        """Get current audio file path for Gradio"""
        if self.output_file and os.path.exists(self.output_file):
            file_size = os.path.getsize(self.output_file)
            if file_size > 0:
                return self.output_file
        return None

    def stop(self):
        """Stop generation"""
        self.should_stop = True
        print("Stop request received")

# Load reference example (Malayalam)
EXAMPLES = [{
    "audio_url": "https://raw.githubusercontent.com/Aparna0112/voicerecording-_TTS/main/KC%20Voice.wav",
    "ref_text": "ഹലോ ഇത് അപരനെ അല്ലേ ഞാൻ ജഗദീപ് ആണ് വിളിക്കുന്നത് ഇപ്പോൾ ഫ്രീയാണോ സംസാരിക്കാമോ ",
    "synth_text": "ഞാൻ മലയാളം സംസാരിക്കാൻ കഴിയുന്നു."
}]

print("\nPreloading reference audio...")
ref_sr, ref_audio = load_audio_from_url(EXAMPLES[0]["audio_url"])

if ref_audio is None:
    print("⚠️ Failed to load reference audio. Using silence instead.")
    ref_audio = np.zeros(int(24000))
    ref_sr = 24000

# Initialize streaming TTS
streaming_tts = StreamingTTS()

# Add a stop button functionality
def stop_generation():
    streaming_tts.stop()
    return "Generation stopped"

# Gradio interface with offline mode
with gr.Blocks() as iface:
    gr.Markdown("## 🚀 IndicF5 Malayalam TTS")

    with gr.Row():
        gr.Markdown("### System Status:")
        system_status = gr.Markdown(f"- Device: {device}\n- Model loaded: {'Yes' if model is not None else 'No'}\n- Reference audio: {'Loaded' if ref_audio is not None else 'Not loaded'}")

    with gr.Row():
        text_input = gr.Textbox(
            label="Malayalam Text",
            placeholder="Enter text here...",
            lines=3,
            value=EXAMPLES[0]["synth_text"] if EXAMPLES else "ഹലോ, എന്തൊക്കെ ഉണ്ട് വിശേഷം?"
        )

    with gr.Row():
        generate_btn = gr.Button("🎤 Generate Speech", variant="primary")
        stop_btn = gr.Button("🛑 Stop Generation", variant="secondary")

    # Status indicator
    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)

    # Audio output
    output_audio = gr.Audio(
        label="Generated Speech",
        type="filepath",
        autoplay=True
    )

    # Debug information (hidden by default)
    with gr.Accordion("Advanced", open=False):
        debug_output = gr.Textbox(label="Debug Log", value="", lines=5)

    def start_generation(text):
        if not text.strip():
            return None, "Please enter some text", "Error: Empty text input"

        if model is None:
            return None, "⚠️ Model not loaded. Cannot generate speech.", "Error: Model not loaded"

        if ref_audio is None:
            return None, "⚠️ Reference audio not loaded. Cannot generate speech.", "Error: Reference audio not loaded"

        # Print the text being processed
        print(f"🔍 User input text: '{text}'")

        # Capture stdout for debug purposes
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # Make sure the text is explicitly passed as the first parameter
                streaming_tts.generate(
                    text=text,  # Explicitly name parameter
                    ref_audio=ref_audio,
                    ref_sr=ref_sr,
                    ref_text=EXAMPLES[0]["ref_text"] if EXAMPLES else ""
                )
            except Exception as e:
                print(f"Error starting generation: {e}")

        debug_log = f.getvalue()

        # Add a delay to ensure file is created
        time.sleep(2.0)

        audio_path = streaming_tts.get_current_audio()
        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path, f"Generated speech for: {text[:30]}...", debug_log
        else:
            return None, "Starting generation... please wait", debug_log

    generate_btn.click(start_generation, inputs=text_input, outputs=[output_audio, status_text, debug_output])
    stop_btn.click(stop_generation, inputs=None, outputs=status_text)

# Cleanup on exit
def exit_handler():
    streaming_tts.cleanup()

import atexit
atexit.register(exit_handler)

# Start the interface with flexible port selection
print("Starting Gradio interface...")
# Try a range of ports if 7860 is busy
iface.launch(share=True)

