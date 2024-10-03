# Nepali Speech-to-Text Dataset

This dataset contains high-quality speech samples in Nepali, originally from [OpenSLR SLR43](https://www.openslr.org/43/) and Mozilla's [Common Voice dataset](https://discourse.mozilla.org/t/common-voice-19-0-dataset-release/135857/1). It has been cleaned and processed for Automatic Speech Recognition (ASR) tasks. The dataset consists of approximately 3,000 audio samples, each around 30 seconds long, compiled for use in training and testing ASR models.

## Dataset Details

- **Link:** https://huggingface.co/datasets/amitpant7/nepali-speech-to-text/
- **Number of samples:** approx. 3,000
- **Audio length per sample:** ~30 seconds
- **Language:** Nepali

## Usage

To load and work with the dataset, you can use the `datasets` library from Hugging Face. Below is the code to load, process, and split the data for training and testing.

### Load the dataset

```python
from datasets import load_dataset, concatenate_datasets

# Load dataset
data = load_dataset("amitpant7/nepali-speech-to-text")

# Check available splits
print("Available splits:", data.keys())

# Combine all splits into a single dataset
total_splits = []
for i in range(0, len(data)):
    split_name = f'train.{i}'
    if split_name in data:
        total_splits.append(data[split_name])
    else:
        break

combined_dataset = concatenate_datasets(total_splits)
print(f"Total dataset size: {len(combined_dataset)}")
```

### Split the dataset into training and testing sets

```python
# Split the dataset into 90% training and 10% testing
split_data = combined_dataset.train_test_split(test_size=0.1)

# Access the train and test sets
train_np = split_data['train']
val_np = split_data['test']

print(f"Train dataset size: {len(train_np)}")
print(f"Test dataset size: {len(val_np)}")
```

## Intended Use

This dataset is intended for training and evaluating speech-to-text (ASR) models in Nepali. You can use it to fine-tune models like [Whisper](https://huggingface.co/models?search=whisper) or other ASR architectures that support Nepali language datasets.

## License

Refer to the license details of the original datasets from [OpenSLR](https://www.openslr.org/resources/43/LICENSE) and [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) for licensing terms.

---
