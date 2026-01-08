# Nepali Speech to Text (ASR) System

<div align="center">

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Nepali%20ASR-yellow)](https://huggingface.co/amitpant7/Nepali-Automatic-Speech-Recognition)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/kshitizzzzzzz/NEPALI_ASR_Whisper_Small)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-green)](https://huggingface.co/datasets/amitpant7/nepali-speech-to-text)

**Fine-tuned Whisper Small model for Nepali speech recognition**

[Quick Start](#-quick-start) • [Model Usage](#-using-the-model) • [Training](#-training) • [Demo](#-demo) • [Repository Structure](#-repository-structure)

</div>

---

## 📋 Overview

An Automatic Speech Recognition (ASR) system for transcribing Nepali speech to text. Built on OpenAI's Whisper Small model and fine-tuned on a custom Nepali dataset, achieving a Word Error Rate (WER) of 32% on the validation set.

### Key Features
- 🎯 Fine-tuned specifically for Nepali language
- 🚀 Easy-to-use HuggingFace integration
- 💻 Both GUI and CLI inference options
- 📊 Trained on diverse Nepali speech data
- 🔓 Open-source model and dataset

### Performance
- **Current WER**: 32% on Common Voice and validation set
- **Base Model**: OpenAI Whisper Small
- **Training Data**: Custom Nepali speech dataset

---

### 🌐 Use Cases

- **Transcription Services**: Convert Nepali audio/video content to text
- **Subtitle Generation**: Auto-generate Nepali subtitles for videos
- **Voice Assistants**: Build Nepali voice-enabled applications
- **Accessibility**: Help hearing-impaired users access Nepali audio content
- **Research**: Academic research on Nepali language processing
- **Education**: Language learning and pronunciation tools

## 🚀 Quick Start

### Using the Model (HuggingFace - Recommended)

The easiest way to use the model is through HuggingFace:

```python
from transformers import pipeline

# Load the model
transcriber = pipeline(
    "automatic-speech-recognition",
    model="amitpant7/Nepali-Automatic-Speech-Recognition"
)

# Transcribe audio
result = transcriber("path/to/your/audio.mp3")
print(result["text"])
```

### Try Online Demo
**No installation required!** Try the model instantly:
👉 [**Live Demo on HuggingFace Spaces**](https://huggingface.co/spaces/kshitizzzzzzz/NEPALI_ASR_Whisper_Small)

---

## 💻 Using the Model

### Option 1: HuggingFace Pipeline (Easiest)

```python
from transformers import pipeline
import torch

# Initialize the pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    model="amitpant7/Nepali-Automatic-Speech-Recognition",
    device=device
)

# Transcribe audio file
result = pipe("audio.mp3")
print(result["text"])

# Transcribe with timestamps
result = pipe("audio.mp3", return_timestamps=True)
print(result)
```

### Option 2: Direct Model Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load model and processor
processor = WhisperProcessor.from_pretrained("amitpant7/Nepali-Automatic-Speech-Recognition")
model = WhisperForConditionalGeneration.from_pretrained("amitpant7/Nepali-Automatic-Speech-Recognition")

# Load and process audio
audio, sr = librosa.load("audio.mp3", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
```

### Option 3: Local Installation

For development or offline use:

```bash
# Clone the repository
git clone https://github.com/fuseai-fellowship/Nepali-Speech-to-Text-Translation.git
cd Nepali-Speech-to-Text-Translation

# Install dependencies
pip install -r requirements.txt
```

#### GUI Interface (Streamlit)

```bash
cd src/inference
streamlit run app.py
```

#### CLI Inference

```bash
python src/inference.py path/to/audio.mp3
```

---

## 📊 Training Dataset

### Access the Dataset
The training dataset is publicly available on HuggingFace:
- **Dataset Link**: [amitpant7/nepali-speech-to-text](https://huggingface.co/datasets/amitpant7/nepali-speech-to-text)
- **Format**: Audio files with Nepali transcriptions
- **Sources**: Multiple sources including SLR143, SLR43, and custom collections

### Using the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("amitpant7/nepali-speech-to-text")

# Access training data
print(dataset["train"][0])
```

For detailed information about the dataset, see [dataset/README.md](./dataset/README.md)

---

## 🔧 Training

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Training the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Start training
python src/train.py
```

### Training Configuration
The model was trained with the following setup:
- Base model: `openai/whisper-small`
- Epochs: 5
- Batch size: 16
- Learning rate: 1e-5
- Optimizer: AdamW

For detailed training notebooks, see the `notebook/` directory.

---

## 📁 Repository Structure

```
.
├── assets/                          # Images and visual assets
│   └── image.png                    # Training metrics visualization
│
├── dataset/                         # Dataset preparation and sources
│   ├── male-female-data/           # SLR143 dataset
│   ├── ne_np_female/               # SLR43 dataset
│   ├── preperation_scripts/        # Data preprocessing scripts
│   ├── scraping/                   # Web scraping tools
│   ├── synthetic_data_using_TTS/   # Synthetic data generation
│   └── README.md                   # Dataset documentation
│
├── docs/                           # Additional documentation
│
├── notebook/                       # Jupyter notebooks
│   ├── finetuning-whispher-on-Nepali-base_old_data.ipynb
│   ├── finetuning-whispher-on-Nepali-small_old_data.ipynb
│   ├── notebook_inference_and_push_hub.ipynb
│   ├── whisper_fine_tune_5_epoch.ipynb
│   └── whispher-finetune-on-small_NP_ASR_data.ipynb
│
├── src/                            # Source code
│   ├── inference/                  # Inference application
│   │   └── app.py                 # Streamlit GUI
│   ├── inference.py               # CLI inference script
│   ├── train.py                   # Training script
│   ├── utils.py                   # Utility functions
│   └── test.mp3                   # Sample audio file
│
├── tests/                          # Unit tests
│   └── test_template.py
│
├── Dockerfile                      # Docker configuration
├── Makefile                        # Build automation
├── pyproject.toml                  # Project metadata
├── requirements.in                 # Core dependencies
├── requirements.txt                # Full dependencies list
└── README.md                       # This file
```

### Key Directories

- **`src/`**: Main source code for training and inference
- **`dataset/`**: Data preparation scripts and dataset information
- **`notebook/`**: Experimental notebooks for model development
- **`src/inference/`**: Streamlit web application for easy testing

---

## 📈 Results

### Training Metrics

![Training Loss and WER vs Epochs](assets/image.png)

### Evaluation Metric: Word Error Rate (WER)

WER measures transcription accuracy:

$$\text{WER} = \frac{\text{Substitutions} + \text{Insertions} + \text{Deletions}}{\text{Total Words}}$$

**Current Performance**: WER of **32%** on validation set

Lower WER indicates better performance. Our model achieves competitive results for Nepali ASR.

---

## ⚠️ Known Limitations

Current challenges and areas for improvement:

1. **Background Noise**: Audio with significant background noise affects transcription quality
2. **Limited Training Data**: More diverse data needed for better generalization
3. **Multi-speaker Scenarios**: Multiple speakers or overlapping speech not handled well
4. **Domain Specificity**: Performance may vary across different audio domains

---

## 🗺️ Roadmap

### Next Steps

1. **Data Collection**: Gather more diverse, high-quality Nepali speech data
2. **Model Scaling**: Train larger Whisper variants (Medium, Large) with more GPU resources
3. **Noise Robustness**: Implement better noise handling and audio preprocessing
4. **Multi-speaker Support**: Improve performance on conversations with multiple speakers
5. **Domain Adaptation**: Fine-tune for specific domains (news, conversational, technical)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug or have a suggestion? [Open an issue](https://github.com/fuseai-fellowship/Nepali-Speech-to-Text-Translation/issues)
2. **Improve Dataset**: Share high-quality Nepali audio recordings
3. **Code Contributions**: Submit pull requests for improvements
4. **Documentation**: Help improve documentation and examples

---

## 📄 License

This project is open-source. Please check the repository for license details.

---

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- HuggingFace for model hosting and tools
- Contributors to the Nepali speech datasets (SLR143, SLR43)
- FuseAI Fellowship program

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/fuseai-fellowship/Nepali-Speech-to-Text-Translation/issues)
- **Model**: [HuggingFace Model Card](https://huggingface.co/amitpant7/Nepali-Automatic-Speech-Recognition)
- **Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/amitpant7/nepali-speech-to-text)

---

<div align="center">

**Made with ❤️ for the Nepali language community**

⭐ Star this repo if you find it useful!

</div>
