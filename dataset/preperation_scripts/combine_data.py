import os
import pandas as pd
import shutil

# Define the paths
common_voice_np_csv = 'common-voice-np.csv'
openslr_tts_csv = 'openslr-tts.csv'
data_dir = './data'

# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Read the CSV files
common_voice_np_df = pd.read_csv(common_voice_np_csv, header=None)
openslr_tts_df = pd.read_csv(openslr_tts_csv, header=None)

# Combine the DataFrames
combined_df = pd.concat([common_voice_np_df, openslr_tts_df], ignore_index=True)

# Prepare a list to hold new paths and corresponding second column values
new_audio_files = []

# Copy audio files and store new paths with the second column
for index, row in combined_df.iterrows():
    original_path = row[0]
    second_column_value = row[1] if len(row) > 1 else ''
    
    # Get the filename to copy
    filename = os.path.basename(original_path)
    # Define the new path
    new_path = os.path.join(data_dir, filename)
    
    # Copy the file
    shutil.copy(original_path, new_path)
    # Store the new path and the second column value
    new_audio_files.append([new_path, second_column_value])

# Create a new DataFrame for the combined CSV
new_combined_df = pd.DataFrame(new_audio_files, columns=['audio_path', 'second_column'])

# Save the new combined CSV
new_combined_csv_path = 'combined_audio_paths.csv'
new_combined_df.to_csv(new_combined_csv_path, index=False)

print(f"Combined CSV created: {new_combined_csv_path}")
print(f"All audio files copied to: {data_dir}")
