
import textwrap
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