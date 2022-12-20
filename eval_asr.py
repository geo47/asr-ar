import torch
import torchaudio
from datasets import load_dataset
from lang_trans.arabic import buckwalter
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

asr_model = "muzamil47/wav2vec2-large-xlsr-53-arabic-demo"

dataset = load_dataset("common_voice", "ar", split="test[:10]")

resamplers = {  # all three sampling rates exist in test split
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
}

def prepare_example(example):
    speech, sampling_rate = torchaudio.load(example["path"])
    example["speech"] = resamplers[sampling_rate](speech).squeeze().numpy()
    return example

dataset = dataset.map(prepare_example)
processor = Wav2Vec2Processor.from_pretrained(asr_model)
model = Wav2Vec2ForCTC.from_pretrained(asr_model).eval()

def predict(batch):
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        predicted = torch.argmax(model(inputs.input_values).logits, dim=-1)
    predicted[predicted == -100] = processor.tokenizer.pad_token_id  # see fine-tuning script
    batch["predicted"] = processor.tokenizer.batch_decode(predicted)
    return batch

dataset = dataset.map(predict, batched=True, batch_size=1, remove_columns=["speech"])

for reference, predicted in zip(dataset["sentence"], dataset["predicted"]):
    print("reference:", reference)
    print("predicted:", buckwalter.untrans(predicted))
    print("--")