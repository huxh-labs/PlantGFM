import os
import pandas as pd
import torch
import numpy as np
import argparse
from datasets import Dataset, DatasetDict
from sklearn.metrics import r2_score, accuracy_score, f1_score, matthews_corrcoef
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast
from plantgfm.modeling_plantgfm import PlantGFMForSequenceClassification
from plantgfm.modeling_segmentgfm import  SegmentGFMConfig,SegmentGFMModel
from datasets import load_dataset
# Disable parallelism in tokenizers to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to tokenize the sequences
def tokenize_function_regression_or_classification(examples, tokenizer, max_length):
    return tokenizer(
        examples['sequence'], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
    )
    
def tokenize_function_segmentation(examples, tokenizer, max_length):
    tokenized_examples = {"input_ids": [], "labels": []}
    for column_name, column_value in examples.items():
        for value in column_value:
            value_list = value.split("\t")
            sequence = value_list[0]
            cds_labels = [eval(value_list[i]) for i in range(1, max_length - 2 + 1)]
            tokenized_sequence = tokenizer(
                sequence, 
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt", 
            )
            tokenized_examples["input_ids"].append(tokenized_sequence["input_ids"].flatten())
            tokenized_examples["labels"].append(torch.Tensor(cds_labels))

    return tokenized_examples

    
# Function to compute evaluation metrics for regression task
def compute_metrics_for_regression_task(eval_pred):
    predictions, labels = eval_pred
    rmse = r2_score(labels, predictions)
    return {"r2": rmse}

# Function to compute evaluation metrics for classification task
def compute_metrics_for_classification_task(eval_pred):
    predictions, labels = eval_pred
    pred_labels = predictions.argmax(axis=-1) if len(predictions.shape) > 1 else predictions
    acc = accuracy_score(labels, pred_labels)
    return {"accuracy": acc}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logits_to_labels(x):
    pred_labels = np.zeros_like(x)
    pred_labels[x >= 0.50] = 1
    return pred_labels

def compute_metrics_for_segmentation_task(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_labels = logits_to_labels(sigmoid(logits))
    return {
        "mcc": matthews_corrcoef(labels.ravel(), pred_labels.ravel())
    }
    
# Main function to train and test the model
def train_and_test(data_name, output_dir, model_name_or_path, tokenizer_path, max_length, batch_size, epochs, learning_rate, logging_strategy, evaluation_strategy, save_strategy, save_total_limit, weight_decay, task_type):
    # Load tokenizer and add special tokens
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    if(task_type == 'regression' or task_type == 'classification'):
        # Load the data
        train_data = pd.read_csv(f'{data_name}/train.csv')
        val_data = pd.read_csv(f'{data_name}/val.csv')
        test_data = pd.read_csv(f'{data_name}/test.csv')

        ds_train = Dataset.from_pandas(train_data)
        ds_valid = Dataset.from_pandas(val_data)
        ds_test = Dataset.from_pandas(test_data)

        # Create the dataset dictionary
        raw_datasets = DatasetDict({
            "train": ds_train,
            "valid": ds_valid,
            "test": ds_test, 
        })
        # Tokenize the datasets
        tokenized_datasets = raw_datasets.map(lambda examples: tokenize_function_regression_or_classification(examples, tokenizer, max_length), batched=True)
        model = PlantGFMForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        raw_datasets = load_dataset(
            "csv",
            data_files={
                "train":os.path.join(data_name,"train.tsv"),
                "valid":os.path.join(data_name,"val.tsv"),
                },
            split=["train","valid"],
            )
        tokenized_datasets = {}
        tokenized_datasets["train"] = raw_datasets[0].map(
            lambda examples: tokenize_function_segmentation(examples,tokenizer,max_length),
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            )
        tokenized_datasets["valid"] = raw_datasets[1].map(
            lambda examples:tokenize_function_segmentation(examples,tokenizer,max_length),
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            )

        config = SegmentGFMConfig(pre_trained_path=model_name_or_path)
        model = SegmentGFMModel(config=config)


    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/',
        logging_strategy=logging_strategy,
        evaluation_strategy=evaluation_strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy=save_strategy,
        num_train_epochs=epochs,
        save_total_limit=save_total_limit,
        save_safetensors=False,
        learning_rate=learning_rate,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        bf16=True,
        weight_decay=weight_decay,
        gradient_checkpointing=True
    )

    # Set metric_for_best_model based on task_type
    if task_type == 'regression':
        compute_metrics = compute_metrics_for_regression_task
        metric_for_best_model = 'r2'
    elif task_type == 'classification':
        compute_metrics = compute_metrics_for_classification_task
        metric_for_best_model = 'accuracy'
    elif task_type == 'segmentation':
        compute_metrics = compute_metrics_for_segmentation_task
        metric_for_best_model = 'mcc'
    else:
        raise ValueError("Invalid task type. Choose from 'regression', 'classification'")

    # Update training arguments with metric_for_best_model
    training_args.metric_for_best_model = metric_for_best_model

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    if(tokenized_datasets["test"] is not None):
        # Evaluate the model
        result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print(result)

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the model.")
    
    # Required arguments
    parser.add_argument('--data_name', type=str, required=True, help="The data name or identifier.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory to store model and logs.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer.")
    
    # Optional arguments with defaults
    parser.add_argument('--max_length', type=int, default=172, help="Maximum sequence length for tokenization.")
    parser.add_argument('--batch_size', type=int, default=96, help="Batch size for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for training.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument('--logging_strategy', type=str, default='epoch', choices=['steps', 'epoch'], help="Strategy for logging.")
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', choices=['steps', 'epoch'], help="Strategy for evaluation.")
    parser.add_argument('--save_strategy', type=str, default='epoch', choices=['steps', 'epoch'], help="Strategy for saving checkpoints.")
    parser.add_argument('--save_total_limit', type=int, default=1, help="The maximum number of checkpoints to save.")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay for optimization.")
    
    # Task type (regression, classification)
    parser.add_argument('--task_type', type=str, required=True, choices=['regression', 'classification','segmentation'], help="Type of task (regression, classification).")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Train and test the model with the provided arguments
    train_and_test(
        data_name=args.data_name, 
        output_dir=args.output_dir, 
        model_name_or_path=args.model_name_or_path, 
        tokenizer_path=args.tokenizer_path, 
        max_length=args.max_length, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate, 
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        task_type=args.task_type
    )
