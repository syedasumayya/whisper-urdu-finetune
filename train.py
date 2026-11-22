import argparse

import evaluate
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data_prep import (
    MODEL_ID,
    DataCollatorSpeechSeq2SeqWithPadding,
    build_dataset,
    get_processor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    cli = parser.parse_args()

    processor = get_processor()
    ds = build_dataset(processor)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "urdu"
    model.generation_config.task = "transcribe"

    # LoRA: freeze the base model, train small extra matrices on the attention layers
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(normalizer(p), normalizer(l)) for p, l in zip(pred_str, label_str)]
        pairs = [(p, l) for p, l in pairs if len(l) > 0]
        wer = wer_metric.compute(
            predictions=[p for p, _ in pairs], references=[l for _, l in pairs]
        )
        return {"wer": wer}

    args = Seq2SeqTrainingArguments(
        output_dir=cli.out_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=cli.learning_rate,
        warmup_steps=50,
        max_steps=cli.max_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        logging_steps=25,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"].select(range(100)),
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    model.save_pretrained(cli.out_dir + "/final")
    processor.save_pretrained(cli.out_dir + "/final")
    print("Saved to", cli.out_dir + "/final")


if __name__ == "__main__":
    main()
