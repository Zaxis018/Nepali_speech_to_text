import os
import pandas as pd
from datasets import Dataset, Audio
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset
from hf_token import hf_token

def prepare_dataset(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    
    data = {
        'audio': df['audio_path'].tolist(),
        'transcription': df['transcription'].tolist()
    }
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('audio', Audio())
    
    return dataset

def push_to_huggingface(dataset, repo_name, token):
    # Create the repository if it doesn't exist
    api = HfApi()
    # Push the dataset to Hugging Face
    dataset.push_to_hub(repo_name, token=token)

def push_large_dataset(csv_path, audio_dir, repo_name, token, batch_size=1000):
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        data = {
            'audio': batch_df['audio_path'].tolist(),
            'transcription': batch_df['transcription'].tolist()
        }
        
        dataset = Dataset.from_dict(data)
        dataset = dataset.cast_column('audio', Audio())
        
        # Push this batch to Hugging Face
        dataset.push_to_hub(repo_name, token=token, split=f"train.{i//batch_size}")
    
    print(f"Dataset successfully pushed to Hugging Face in {(total_samples-1)//batch_size + 1} batches!")

if __name__ == "__main__":
    # Set your parameters
    csv_path = "metadata.csv"
    audio_dir = "./data"  # Directory containing individual audio files
    repo_name = "amitpant7/nepali-speech-to-text"

    # For smaller datasets (less than 100 files)
    # dataset = prepare_dataset(csv_path, audio_dir)
    # push_to_huggingface(dataset, repo_name, hf_token)

    # For larger datasets
    push_large_dataset(csv_path, audio_dir, repo_name, hf_token)

    print("Dataset successfully prepared and pushed to Hugging Face!")