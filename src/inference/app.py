"""
Usage:
    streamlit run app.py
    
"""

import streamlit as st
import soundfile as sf
from transformers import pipeline
import sounddevice as sd
import numpy as np
import tempfile
import os
import time

@st.cache_resource
def load_pipeline():
    """
    Load and cache the Whisper pipeline for Nepali ASR.
    
    Returns:
        pipeline: A Hugging Face pipeline for automatic speech recognition.
    """
    model_name = r"src/inference/ASR_nepali_finetuned"
    whisper_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=0)  # Use device=0 for GPU
    return whisper_pipeline

whisper_pipeline = load_pipeline()

st.title("Voice-to-Text Converter (Nepali)")

is_recording = False

def record_audio(duration, samplerate=16000):
    """
    Record audio from the user's microphone.
    
    Args:
        duration (int): The duration of the recording in seconds.
        samplerate (int): The sample rate of the recording (default: 16000).
    
    Returns:
        numpy.ndarray: The recorded audio as a NumPy array.
    """
    global is_recording
    is_recording = True
    recorded_frames = []
    
    def callback(indata, frames, time, status):
        if is_recording:
            recorded_frames.append(indata.copy())
    
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(duration):
            if not is_recording:
                break
            status_text.text(f"Recording: {i+1} seconds")
            progress_bar.progress((i + 1) / duration)
            time.sleep(1)
    
    stream.stop()
    status_text.text("Recording complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    recording = np.concatenate(recorded_frames, axis=0)
    return recording.flatten()

def transcribe_audio(audio_input):
    """
    Transcribe the given audio input using the Whisper pipeline.
    
    Args:
        audio_input (numpy.ndarray): The audio input as a NumPy array.
    
    Returns:
        str: The transcribed text.
    """
    result = whisper_pipeline(audio_input)
    return result['text']

# Main application logic
input_method = st.radio("Choose input method:", ("Record Audio", "Upload Audio File"))

if input_method == "Record Audio":
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Recording")
    with col2:
        stop_button = st.button("Stop Recording")
    with col3:
        transcribe_button = st.button("Transcribe Recorded Audio", disabled=True)

    if start_button:
        st.session_state['recording'] = True
        with st.spinner("Recording..."):
            audio = record_audio(60)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio, 16000)
            st.session_state['recorded_audio_path'] = tmp_file.name
        st.audio(st.session_state['recorded_audio_path'], format='audio/wav')
        st.session_state['recording_complete'] = True
        transcribe_button = st.button("Transcribe Recorded Audio", disabled=False, key='transcribe_after_record')

    if stop_button:
        is_recording = False
        st.session_state['recording'] = False

    if transcribe_button or (st.session_state.get('recording_complete', False) and st.button("Transcribe Recorded Audio", key='transcribe_existing')):
        with st.spinner("Transcribing..."):
            audio_input, _ = sf.read(st.session_state['recorded_audio_path'])
            transcription = transcribe_audio(audio_input)
        st.write("Transcription:")
        st.write(transcription)
        os.unlink(st.session_state['recorded_audio_path'])
        st.session_state['recording_complete'] = False

else:
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe Uploaded Audio"):
            with st.spinner("Transcribing..."):
                audio_input, _ = sf.read(tmp_file_path)
                transcription = transcribe_audio(audio_input)
            st.write("Transcription:")
            st.write(transcription)
            os.unlink(tmp_file_path)