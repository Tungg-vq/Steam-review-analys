import argparse
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from transformers import EarlyStoppingCallback

def prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # train val test split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })

def main(config_path):
    # read yaml file
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
        
    print(cfg['run_name'])
    print(f"model: {cfg['model_name_or_path']}")
    print(f"data path: {cfg['data_path']}")

    # tokenizer and model initialization
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_name_or_path'])
    hf_datasets = prepare_data(cfg['data_path'])
    
    def tokenize_function(batch):
        inputs = tokenizer(batch["input_text"], max_length=256, truncation=True, padding="max_length")
        targets = tokenizer(batch["target_text"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs
    tokenized_datasets = hf_datasets.map(
        tokenize_function, 
        batched=True, 
        remove_columns=hf_datasets["train"].column_names
    )
    # config
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/{cfg['run_name']}",
        eval_strategy=cfg['eval_strategy'],
        save_strategy=cfg['save_strategy'],
        logging_strategy=cfg['logging_strategy'],
        learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        weight_decay=cfg['weight_decay'],
        num_train_epochs=cfg['num_train_epochs'],
        fp16=cfg['fp16'],
        predict_with_generate=cfg['predict_with_generate'],
        save_total_limit=cfg['save_total_limit'],
        load_best_model_at_end=cfg['load_best_model_at_end'],
    )
    
    # data collector
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # training
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    print("start training...")
    trainer.train()
    
    # checkpoint save
    final_save_path = f"models/{cfg['run_name']}_final"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"model saved at {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA training script")
    parser.add_argument("--config", type=str, required=True, help=" yaml config file path")
    args = parser.parse_args()
    
    main(args.config)