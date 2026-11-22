import argparse

import evaluate
import pandas as pd
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data_prep import MODEL_ID, get_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="syedasumayya1/whisper-small-urdu-lora",
        help="Hugging Face repo id or local folder of the LoRA adapter",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N test clips (for a quick check)")
    parser.add_argument("--out_csv", default="test_results.csv")
    cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = get_processor()
    test = load_dataset("google/fleurs", "ur_pk", split="test")
    test = test.cast_column("audio", Audio(sampling_rate=16000))
    if cli.limit:
        test = test.select(range(cli.limit))

    normalizer = BasicTextNormalizer()
    refs = [normalizer(t) for t in test["transcription"]]
    wer_m, cer_m = evaluate.load("wer"), evaluate.load("cer")

    def transcribe(model, batch_size=16):
        outs = []
        for i in range(0, len(test), batch_size):
            arrays = [test[j]["audio"]["array"] for j in range(i, min(i + batch_size, len(test)))]
            feats = processor.feature_extractor(
                arrays, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device, dtype=dtype)
            with torch.no_grad():
                ids = model.generate(
                    input_features=feats, language="ur", task="transcribe", max_new_tokens=225
                )
            outs += processor.batch_decode(ids, skip_special_tokens=True)
        return [normalizer(o) for o in outs]

    def load_baseline():
        return WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device, dtype=dtype).eval()

    def load_tuned():
        base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        tuned = PeftModel.from_pretrained(base, cli.adapter).merge_and_unload()
        return tuned.to(device, dtype=dtype).eval()

    preds = {}
    for name, loader in [("baseline", load_baseline), ("fine-tuned", load_tuned)]:
        preds[name] = transcribe(loader())
        w = wer_m.compute(predictions=preds[name], references=refs)
        c = cer_m.compute(predictions=preds[name], references=refs)
        print(f"{name:11s} WER = {w:.3f}   CER = {c:.3f}")

    pd.DataFrame(
        {"reference": refs, "baseline": preds["baseline"], "fine_tuned": preds["fine-tuned"]}
    ).to_csv(cli.out_csv, index=False, encoding="utf-8-sig")
    print("Saved", cli.out_csv)


if __name__ == "__main__":
    main()
