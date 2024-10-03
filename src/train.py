import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate



# Parameters
MODEL_ID = 'openai/whisper-small'
OUTPUT_DIR = 'whisper_nepali_model'
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-5


# Initialization of feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language='Nepali', task='transcribe')
processor = WhisperProcessor.from_pretrained(MODEL_ID, language='Nepali', task='transcribe')

def load_datasets():
    """Load and preprocess the datasets."""
    train_dataset = load_dataset("fsicoli/common_voice_19_0", "ne-NP", split="train", trust_remote_code=True)
    val_dataset = load_dataset("fsicoli/common_voice_19_0", "ne-NP", split="test", trust_remote_code=True)

    # Resample to 16kHz
    train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column('audio', Audio(sampling_rate=16000))

    # Remove unnecessary columns
    columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset


def prepare_dataset(batch):
    """Prepare a single batch of the dataset."""
    audio = batch['audio']
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch



def compute_metrics(pred):
    """Compute WER for model evaluation."""
    metric = evaluate.load('wer')
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}


def main():
    # Load and preprocess datasets
    train_dataset, val_dataset = load_datasets()
    
    train_dataset = train_dataset.map(prepare_dataset, num_proc=4)
    val_dataset = val_dataset.map(prepare_dataset, num_proc=4)

    # Initialize the model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Prepare data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=225,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        report_to=['tensorboard'],
        dataloader_num_workers=4,
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()