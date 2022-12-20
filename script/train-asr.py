from processing_util import TrainingPipeline

from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer


language_code = 'ar'
language_name = 'arabic'
base_model = "facebook/wav2vec2-large-xlsr-53"


if __name__ == '__main__':

    output_models_dir = f"output_models/{language_code}/wav2vec2-large-xlsr-{language_name}-demo"
    new_output_models_dir = f"output_models/{language_code}/wav2vec2-large-xlsr-{language_name}"

    data_loader = TrainingPipeline()
    data_loader.init()
    data_collator = data_loader.load()
    processor = data_loader.get_processor()
    common_voice_train, common_voice_test = data_loader.get_datasets()

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.05,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=new_output_models_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=28,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=data_loader.compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(new_output_models_dir)


