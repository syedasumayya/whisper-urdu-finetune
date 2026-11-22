# Whisper-small fine-tuned for Urdu speech recognition (LoRA)

Fine-tuning `openai/whisper-small` on Urdu with LoRA, measured against the original model on a held-out test set.

**Model:** https://huggingface.co/syedasumayya1/whisper-small-urdu-lora

## Why

Off-the-shelf Whisper-small makes many mistakes on Urdu. This project fine-tunes it on a small public Urdu dataset and measures the change honestly.

## Results

FLEURS Urdu (`ur_pk`) test set, 299 clips, text normalised (lowercased, punctuation removed) before scoring.

| Model | WER | CER |
|---|---|---|
| openai/whisper-small (baseline) | 0.415 | 0.169 |
| This project (LoRA fine-tune) | **0.316** | **0.112** |

About a 24% relative reduction in word errors and a 34% reduction in character errors.

**How much is real Urdu learning?** The baseline sometimes falls into runaway repetition loops (the same word repeated until the token limit). My simple loop detector found 2 such clips in the baseline and none in the fine-tuned model (it missed at least one more, a long run of digits). Excluding the clips it flagged, baseline WER is about 0.400 versus 0.316, so most of the gain comes from better Urdu recognition, with a smaller part from removing loops.

## Example transcriptions

| Reference | Original Whisper-small | Fine-tuned |
|---|---|---|
| 15اگست 1940 کو اتحادی افواج نے جنوبی فرانس پر حملہ کر دیا اس حملے کو آپریشن ڈریگن کا نام دیا گیا | پندرہ اگرس, then a long run of repeated digits (loop) | 15 اگست 1940 کو اتحادی افواجنے جنوبی فرانس پر حملہ کر دیا اس حملے کو آپریشن ڈرگین کا نام دیا گیا |
| ایک بم گورنر جنرل کے دفتر کے باہر پھٹا | ایک بم گوارنل جرنر کے دفتر کے باہر پھٹا | ایک بمگ کورنل جنرد کے دستر کے باہر پھٹا |

The first row shows a clear improvement (digits and spacing now match the dataset's style). The second shows a remaining weakness: names and foreign words, where both models make mistakes.

## Method

- **Data:** [FLEURS](https://arxiv.org/abs/2205.12446) Urdu (Pakistan): 2,109 train, 267 validation, 299 test clips (CC-BY 4.0)
- **Model:** `openai/whisper-small` (244M parameters), base weights frozen
- **LoRA:** rank 32, alpha 64, dropout 0.05 on `q_proj` and `v_proj`; 3.5M trainable parameters (1.44%)
- **Training:** 1,000 steps, effective batch size 16, learning rate 1e-3 with 50 warmup steps, fp16, one Colab T4 GPU, about 71 minutes

## Limitations

- **Overfitting:** validation loss rose after about step 250 and validation WER stayed flat (0.330 at step 250, 0.335 at step 1000) while training loss kept falling. A shorter run would probably match this result.
- The test set was never used for training or checkpoint selection, but validation WER came from only 100 clips, so small differences between checkpoints are noise.
- Only read speech was tested; noisy or conversational Urdu is untested.
- Names, numbers and foreign words are still often wrong. FLEURS references also contain broken word spacing, which inflates WER.

## Use the model

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "syedasumayya1/whisper-small-urdu-lora").merge_and_unload()
processor = WhisperProcessor.from_pretrained("syedasumayya1/whisper-small-urdu-lora")
```

## Run it yourself

```bash
pip install -r requirements.txt
python train.py --out_dir outputs          # trains and saves the adapter to outputs/final
python eval_model.py                        # evaluates baseline vs the published adapter
python eval_model.py --limit 20             # quick check on 20 clips
```

A GPU (for example a free Colab T4) is recommended.

## Files

- `data_prep.py`: loads FLEURS, builds spectrograms and token ids, defines the batch collator
- `train.py`: LoRA fine-tuning
- `eval_model.py`: WER/CER comparison of the baseline and the fine-tuned adapter

## Credits

Base model: OpenAI Whisper. Data: Google FLEURS (Conneau et al., 2022). Built with Hugging Face Transformers, PEFT and Datasets.
