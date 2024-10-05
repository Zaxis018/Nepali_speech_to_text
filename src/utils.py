import random
import textwrap
import numpy as np
from tabulate import tabulate
from pydub import AudioSegment

def mp3_to_wav(input_file, output_file):
    """
    Converts an MP3 file to WAV format.

    Args:
    input_file (str): The path to the input MP3 file.
    output_file (str): The path to the output WAV file.
    """
    audio = AudioSegment.from_mp3(input_file)

    audio.export(output_file, format="wav")
    print(f"File converted and saved as: {output_file}")


def match_sample_rate(audio1, audio2=None, target_sample_rate=16000):
    """Resample the audio files to match the target sample rate."""
    if audio1.frame_rate != target_sample_rate:
        audio1 = audio1.set_frame_rate(target_sample_rate)
    if audio2 and audio2.frame_rate != target_sample_rate:
        audio2 = audio2.set_frame_rate(target_sample_rate)
    return audio1, audio2

def generate_random_noise(duration_ms, sample_rate=16000):
    """Generate random white noise."""
    samples = np.random.normal(0, 1, int(sample_rate * (duration_ms / 1000))).astype(np.int16)
    noise = AudioSegment(
        samples.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=samples.dtype.itemsize, 
        channels=1
    )
    return noise


def add_noise_to_audio(speech_file, noise_file=None, output_file="output_with_noise.wav", snr_dB=10):
    """
    Adds background noise to a speech audio file.
    
    Parameters:
    - speech_file: Path to the speech file (mp3 or wav).
    - noise_file: Path to the noise file (mp3 or wav) or None for random noise.
    - output_file: Path for saving the output audio file.
    - snr_dB: Signal-to-noise ratio in decibels. Default is 10dB.
    """
    # Load speech file
    speech = AudioSegment.from_file(speech_file)
    
    # Generate random noise if no noise file is provided
    if noise_file is None:
        print("No noise file provided. Generating random white noise.")
        noise = generate_random_noise(duration_ms=len(speech), sample_rate=speech.frame_rate)
    else:
        noise = AudioSegment.from_file(noise_file)

    speech, noise = match_sample_rate(speech, noise)

    if len(noise) < len(speech):
        noise = noise * (len(speech) // len(noise) + 1)  
    # Trim noise to the length of the speech
    noise = noise[:len(speech)]

    speech_samples = np.array(speech.get_array_of_samples())
    noise_samples = np.array(noise.get_array_of_samples())

    # Calculate the scaling factor based on desired SNR
    speech_power = np.mean(speech_samples**2)
    noise_power = np.mean(noise_samples**2)
    scaling_factor = np.sqrt(speech_power / (noise_power * 10**(snr_dB / 10)))
    
    # Scale and add noise to the speech
    scaled_noise = noise_samples * scaling_factor
    merged_samples = speech_samples + scaled_noise

    # Convert to AudioSegment and export
    merged_audio = speech._spawn(merged_samples.astype(np.int16).tobytes())
    merged_audio.export(output_file, format="wav")
    print(f"Output saved as {output_file}")


def transcribe_audio(audio_input, sampling_rate=16000):
    """
    Transcribe a single audio segment using the model.
    Args:
        audio_input: A numpy array of audio samples.
        sampling_rate: Sampling rate of the audio input.
    Returns:
        transcription: The transcribed text.
    """
    input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
    model.eval()

    # Perform transcription using generate()
    with torch.no_grad():
        predicted_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription



def transcribe_large_audio(file_path, chunk_duration_sec=20, overlap_duration_sec=0.5):
    """
    Transcribes a large audio file by dividing it into smaller chunks with overlap.

    Args:
        file_path (str): Path to the input audio file.
        chunk_duration_sec (int): Duration of each chunk in seconds (default is 20 seconds).
        overlap_duration_sec (float): Duration of overlap between chunks in seconds (default is 0.5 seconds).

    Returns:
        full_transcription (str): Final transcribed text from the entire audio file.
    """
    # Load the audio file using librosa for accurate sampling rate handling
    audio, sr = librosa.load(file_path, sr=16000)  
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_duration_sec * sr)
    full_transcription = ""

    start = 0
    while start < len(audio):
        end = start + chunk_samples
        chunk = audio[start:end]
        transcription = transcribe_audio(chunk, sampling_rate=sr)

        full_transcription += transcription + " "
        start += chunk_samples - overlap_samples

    return full_transcription.strip()


def compare_texts(transcribed_text, ground_truth, width=50):
    """
    Compares the transcribed text with the ground truth side-by-side in a table.

    Args:
        transcribed_text (str): The text produced by the model.
        ground_truth (str): The reference ground truth text.
        width (int): Maximum width for each column in the table.

    Returns:
        None. Prints a side-by-side comparison.
    """
    ground_truth_lines = textwrap.wrap(ground_truth, width)
    transcribed_lines = textwrap.wrap(transcribed_text, width)

    max_len = max(len(ground_truth_lines), len(transcribed_lines))
    ground_truth_lines += [''] * (max_len - len(ground_truth_lines)) 
    transcribed_lines += [''] * (max_len - len(transcribed_lines))
    data = [["Ground Truth", "Transcribed Text"]]
    for gt_line, trans_line in zip(ground_truth_lines, transcribed_lines):
        data.append([gt_line, trans_line])

    print(tabulate(data, headers="firstrow", tablefmt="fancy_grid", maxcolwidths=[width, width]))