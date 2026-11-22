from dataclasses import dataclass
from typing import Any

from datasets import Audio, load_dataset
from transformers import WhisperProcessor

MODEL_ID = "openai/whisper-small"


def get_processor():
    return WhisperProcessor.from_pretrained(MODEL_ID, language="urdu", task="transcribe")


def build_dataset(processor):
    # Load FLEURS Urdu, resample to 16 kHz, then turn each clip into
    # a log-Mel spectrogram (input_features) and its text into token ids (labels).
    ds = load_dataset("google/fleurs", "ur_pk")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    return ds.map(prepare, remove_columns=ds["train"].column_names)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        # spectrograms are already the same size, so just stack them
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        # labels differ in length: pad them and mark the padding as -100 (ignored by the loss)
        labels_in = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels_in, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # the model adds <|startoftranscript|> itself, so drop it from the labels
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
