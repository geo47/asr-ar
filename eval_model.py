import re

import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import set_seed, Wav2Vec2ForCTC, Wav2Vec2Processor

set_seed(42)

test_dataset = load_dataset("common_voice", "ar", split="test")

processor = Wav2Vec2Processor.from_pretrained("muzamil47/wav2vec2-large-xlsr-53-arabic-demo")
model = Wav2Vec2ForCTC.from_pretrained("muzamil47/wav2vec2-large-xlsr-53-arabic-demo")
model.to("cuda")

chars_to_ignore_regex = '[\,\؟\.\!\-\;\\:\'\"\☭\«\»\؛\—\ـ\_\،\“\%\‘\”\�]'

resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Preprocessing the datasets. We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    batch["sentence"] = re.sub('[a-z]','',batch["sentence"])
    batch["sentence"] = re.sub("[إأٱآا]", "ا", batch["sentence"])
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    batch["sentence"] = re.sub(noise, '', batch["sentence"])
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)


def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
         logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

wer = load_metric("wer")
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
