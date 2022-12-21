import re
import random
import pandas as pd
import numpy as np

import librosa
import torch
import torchaudio

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from datasets import load_dataset, load_metric

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

language_code = 'ar'
language_name = 'arabic'
dataset_dir = "dataset/ar/"
base_model = "facebook/wav2vec2-large-xlsr-53"
new_output_models_dir = f"output_models/{language_code}/wav2vec2-large-xlsr-{language_name}"


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df.head())


chars_to_ignore_regex = '[\,\؟\.\!\-\;\:\'\"\☭\«\»\؛\—\ـ\_\،\“\%\‘\”\�]'


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
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
    return batch


def create_dataset():

    # read the dataset & convert tsv to csv
    df_train = pd.read_csv(dataset_dir + 'train.tsv', sep='\t')
    df_dev = pd.read_csv(dataset_dir + 'dev.tsv', sep='\t')
    df_test = pd.read_csv(dataset_dir + 'test.tsv', sep='\t')

    df_train['path'] = dataset_dir + 'clips/' + df_train['path']
    df_dev['path'] = dataset_dir + 'clips/' + df_dev['path']
    df_test['path'] = dataset_dir + 'clips/' + df_test['path']

    df_train.to_csv(dataset_dir + 'train.csv', sep='\t', index=False)
    df_dev.to_csv(dataset_dir + 'dev.csv', sep='\t', index=False)
    df_test.to_csv(dataset_dir + 'test.csv', sep='\t', index=False)

    cv_train = load_dataset('csv', data_files=[dataset_dir + 'train.csv', dataset_dir + 'dev.csv'],
                                      delimiter='\t')
    cv_train = cv_train['train'].remove_columns(
        ['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent'])

    cv_test = load_dataset('csv', data_files=[dataset_dir + 'test.csv'], delimiter='\t')
    cv_test = cv_test['train'].remove_columns(
        ['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent'])

    cv_train = cv_train.map(remove_special_characters)
    cv_test = cv_test.map(remove_special_characters)
    show_random_elements(cv_train.remove_columns(["path"]), num_examples=20)

    create_vocab(cv_train, cv_test)

    print(cv_train[0])

    cv_train = cv_train.map(speech_file_to_array_fn, remove_columns=cv_train.column_names)
    cv_test = cv_test.map(speech_file_to_array_fn, remove_columns=cv_test.column_names)

    cv_train = cv_train.map(resample, num_proc=4)
    cv_test = cv_test.map(resample, num_proc=4)

    rand_int = random.randint(0, len(cv_train) - 1)

    print("Target text:", cv_train[rand_int]["target_text"])
    print("Input array shape:", np.asarray(cv_train[rand_int]["speech"]).shape)
    print("Sampling rate:", cv_train[rand_int]["sampling_rate"])

    return cv_train, cv_test


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def create_vocab(cv_train, cv_test):
    vocab_train = cv_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=cv_train.column_names)
    vocab_test = cv_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=cv_test.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print(vocab_dict)
    print("VOCAB_LEN: " + str(len(vocab_dict)))

    import json
    with open("vocab.json", 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class TrainingPipeline:

    def init(self):
        self.wer_metric = load_metric("wer")
        self.common_voice_train, self.common_voice_test = create_dataset()
        # show_random_elements(common_voice_train.remove_columns(["path"]), num_examples=20)

        # create_vocab(self.common_voice_train, self.common_voice_test)

        self.tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=True)

        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.processor.save_pretrained(new_output_models_dir)

        self.common_voice_train = self.common_voice_train.map(self.prepare_dataset,
                                                    remove_columns=self.common_voice_train.column_names,
                                                    batch_size=8, num_proc=4, batched=True)
        self.common_voice_test = self.common_voice_test.map(self.prepare_dataset,
                                                  remove_columns=self.common_voice_test.column_names,
                                                  batch_size=8, num_proc=4, batched=True)

    def prepare_dataset(self, batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["target_text"]).input_ids
        return batch

    def load(self):
        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        return data_collator


    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def get_datasets(self):
        return self.common_voice_train, self.common_voice_test

    def get_processor(self):
        return self.processor

