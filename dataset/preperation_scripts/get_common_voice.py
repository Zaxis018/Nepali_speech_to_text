import os
import csv
import shutil
from datasets import load_dataset

# Load the Common Voice Nepali dataset
dataset = load_dataset("fsicoli/common_voice_19_0", "ne-NP", trust_remote_code=True)

# Define directory to save audio files
audio_save_dir = "common-voice-np"
os.makedirs(audio_save_dir, exist_ok=True)

# Path for the CSV file
csv_file_path = "common-voice-np.csv"

# Open the CSV file to write
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header
    csv_writer.writerow(["audio_file", "transcription"])
    
    # Loop through the dataset and save each audio file and transcription for both train and test
    for split in ["train", "test"]:
        for i, sample in enumerate(dataset[split]):
            # Extract audio and transcription
            audio_path = sample["audio"]["path"]
            transcription = sample["sentence"]

            # Define new audio file name
            audio_filename = f"{split}_audio_{i}.wav"  # Include split in the filename
            new_audio_path = os.path.join(audio_save_dir, audio_filename)
            
            # Copy the audio file to the new directory
            shutil.copy(audio_path, new_audio_path)

            # Write audio file and transcription to the CSV
            csv_writer.writerow([new_audio_path, transcription])

print(f"Audio files saved to {audio_save_dir} and transcriptions saved to {csv_file_path}")
