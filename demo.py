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

predict(load_file_to_data("dataset/sample/common_voice_ar_19058307.mp3"))

