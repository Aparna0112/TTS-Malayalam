import os
import io
import torch
import librosa
import requests
import tempfile
import numpy as np
import soundfile as sf
import gradio as gr
from transformers import AutoModel
import spaces
import time
from huggingface_hub import login

# Get Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN","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")  # Retrieve token from environment

# Ensure the token is set, otherwise raise an error
if hf_token:
    login(token=hf_token)  # Log in using the token from environment
else:
    raise ValueError("Hugging Face token not found in environment variables.")

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

# Function to check and use GPU
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# Resampling function to match sample rates
def resample_audio(audio, orig_sr, target_sr):
    if orig_sr != target_sr:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text):
    if ref_audio is None or ref_text.strip() == "":
        return "Error: Please provide a reference audio and its corresponding text."

    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."

    # Save reference audio directly without resampling
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

    # Profiling the model inference time
    start_time = time.time()

    # Run the inference
    audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")

    # Normalize output and save
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Resample the generated audio to match the reference audio's sample rate
    audio = resample_audio(audio, orig_sr=24000, target_sr=sample_rate)

    return sample_rate, audio

# Load TTS model and move it to the appropriate device (GPU/CPU)
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = get_device()  # Get device (GPU/CPU)
model = model.to(device)

# Example Malayalam Data
EXAMPLES = [
    {
        "audio_name": "Aparna Voice",
        "audio_url": "https://raw.githubusercontent.com/Aparna0112/voicerecording-_TTS/main/Aparna%20Voice.wav",  # Replace with actual Malayalam audio URL
        "ref_text": " ഞാൻ ഒരു ഫോണിന്‍റെ കവർ നോക്കുകയാണ്. എനിക്ക് സ്മാർട്ട് ഫോണിന് കവർ വേണം",
        "synth_text": "ഞാൻ മലയാളം സംസാരിക്കാൻ കഴിയുന്നു."
    },
    {
        "audio_name": "KC Voice",
        "audio_url": "https://raw.githubusercontent.com/Aparna0112/voicerecording-_TTS/main/KC%20Voice.wav",  # Replace with actual Malayalam audio URL
        "ref_text": "ഹലോ ഇത് അപരനെ അല്ലേ ഞാൻ ജഗദീപ് ആണ് വിളിക്കുന്നത് ഇപ്പോൾ ഫ്രീയാണോ സംസാരിക്കാമോ ",
        "synth_text": "ഞാൻ മലയാളം സംസാരിക്കാൻ കഴിയുന്നു."
    },
    # Add more examples here if needed
]

# Preload all example audios
for example in EXAMPLES:
    sample_rate, audio_data = load_audio_from_url(example["audio_url"])
    example["sample_rate"] = sample_rate
    example["audio_data"] = audio_data

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Text-to-Speech for Malayalam**
        [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)
        Use **IndicF5**, a **Text-to-Speech (TTS)** model, to generate Malayalam speech.
        """
    )

    with gr.Row():
        with gr.Column():
            # Text to synthesize
            text_input = gr.Textbox(label="Text to Synthesize (Malayalam)", placeholder="Enter Malayalam text...", lines=3)
            # Reference audio input
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
            # Reference text input
            ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio (Malayalam)", placeholder="Enter the transcript in Malayalam...", lines=2)
            # Submit button
            submit_btn = gr.Button("🎤 Generate Malayalam Speech", variant="primary")

        with gr.Column():
            # Output audio of generated speech
            output_audio = gr.Audio(label="Generated Speech (Malayalam)", type="numpy")

    # Dropdown to select audio name
    audio_name_input = gr.Dropdown(
        label="Select Reference Audio",
        choices=[ex["audio_name"] for ex in EXAMPLES],
        type="value"  # The value will be the audio name
    )

    # Function to update the reference audio and text based on selected audio name
    def update_reference_audio(selected_audio_name):
        # Find the selected example by audio name
        selected_example = next(ex for ex in EXAMPLES if ex["audio_name"] == selected_audio_name)
        ref_audio = (selected_example["sample_rate"], selected_example["audio_data"])
        ref_text = selected_example["ref_text"]
        return ref_audio, ref_text

    # Use `audio_name_input` to update `ref_audio_input` and `ref_text_input`
    audio_name_input.change(
        update_reference_audio,
        inputs=[audio_name_input],
        outputs=[ref_audio_input, ref_text_input]
    )

    # Set the click event for the button
    submit_btn.click(
        synthesize_speech,
        inputs=[text_input, ref_audio_input, ref_text_input],
        outputs=[output_audio]
    )

iface.launch(share=True)
