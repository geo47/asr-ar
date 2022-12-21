# asr-ar

# wav2vec2-large-xlsr-53-arabic-demo

This model is fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Arabic using the [Common Voice Corpus 6.1](https://commonvoice.mozilla.org/en/datasets).

The fine-tuned [muzamil47/wav2vec2-large-xlsr-53-arabic-demo](https://huggingface.co/muzamil47/wav2vec2-large-xlsr-53-arabic-demo) model is available on [HuggingFace](https://huggingface.co/).

## Training

The [Common Voice Corpus 6.1](https://commonvoice.mozilla.org/en/datasets) `train`, `validation` datasets were used for training.

The model is fine-tuned on [XLS-R: Self-supervised Corss-lingual Speech Representation Learning at Scale](https://arxiv.org/pdf/2111.09296.pdf). 
When using this model, make sure that your speech input is sampled at 16kHz.

Most of the code used in this work has been adotpted from [this repo](https://github.com/anashas/Fine-Tuning-of-XLSR-Wav2Vec2-on-Arabic).

Before training and using make sure to download the [Common Voice Corpus 6.1](https://commonvoice.mozilla.org/en/datasets) dataset and install the following dependencies.

```shell
pip install librosa
pip install torchaudio
pip install transformers
pip install datasets
pip install jiwer
pip install arabic_pronounce
```

Copy the dataset in `dataset/ar/` folder and run `python script/train_asr.py`. 

## Usage

The model demo can be used directly as follows `python demo.py`:

```python
import librosa
import torch
from lang_trans.arabic import buckwalter

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

asr_model = "muzamil47/wav2vec2-large-xlsr-53-arabic-demo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_file_to_data(file, srate=16_000):
    batch = {}
    speech, sampling_rate = librosa.load(file, sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


processor = Wav2Vec2Processor.from_pretrained(asr_model)
model = Wav2Vec2ForCTC.from_pretrained(asr_model).to(device)


def predict(data):
    features = processor(data["speech"], sampling_rate=data["sampling_rate"], return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    try:
        attention_mask = features.attention_mask.to(device)
    except:
        attention_mask = None
    with torch.no_grad():
        predicted = torch.argmax(model(input_values, attention_mask=attention_mask).logits, dim=-1)

    data["predicted"] = processor.tokenizer.decode(predicted[0])
    print(data["predicted"])
    print("predicted:", buckwalter.untrans(data["predicted"]))
    return data

predict(load_file_to_data("common_voice_ar_19058307.mp3"))
```
**Output Result**:
```shell
reference: هل يمكنني التحدث مع المسؤول هنا
predicted: هل يمكنني التحدث مع المسؤول هنا
```

## Evaluation

The model can be evaluated as follows on the Arabic test data of [Common Voice](https://huggingface.co/datasets/common_voice). `python eval_asr.py`

```python
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

```
**Output Results**:
```shell
reference: ما أطول عودك!
predicted: ما اطول عودك

reference: ماتت عمتي منذ سنتين.
predicted: ما تتعمتي منذو سنتين

reference: الألمانية ليست لغة سهلة.
predicted: الالمانية ليست لغة سهلة

reference: طلبت منه أن يبعث الكتاب إلينا.
predicted: طلبت منه ان يبعث الكتاب الينا

reference: .السيد إيتو رجل متعلم
predicted: السيد ايتو رجل متعلم

reference: الحمد لله.
predicted: الحمذ لللا

reference: في الوقت نفسه بدأت الرماح والسهام تقع بين الغزاة
predicted: في الوقت نفسه ابدات الرماح و السهام تقع بين الغزاء

reference: لا أريد أن أكون ثقيلَ الظِّل ، أريد أن أكون رائعًا! !
predicted: لا اريد ان اكون ثقيل الظل اريد ان اكون رائع

reference: خذ مظلة معك في حال أمطرت.
predicted: خذ مظلة معك في حال امطرت

reference: .ركب توم السيارة
predicted: ركب توم السيارة
```

The model evaluation **(WER)** on the Arabic test data of [Common Voice](https://huggingface.co/datasets/common_voice). `python eval_model.py`

```python
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

```

**Test Result**: 53.54
