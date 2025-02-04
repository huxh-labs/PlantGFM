import os
import pandas as pd
import torch
import numpy as np
import argparse
from datasets import Dataset, DatasetDict
from sklearn.metrics import r2_score, accuracy_score, f1_score, matthews_corrcoef
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast
from plantglm.modeling_plantglm import PlantGLMForSequenceClassification
from plantglm.configuration_plantglm import PlantGLMConfig

# Disable parallelism in tokenizers to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to tokenize the sequences
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples['sequence'], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
    )

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

# Sigmoid function for segmentation
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Convert logits to labels for segmentation task
def logits_to_labels(x):
    pred_labels = np.zeros_like(x)
    pred_labels[x >= 0.50] = 1
    return pred_labels

# Function to compute evaluation metrics for segmentation task
def compute_metrics_for_segmentation_task(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_labels = logits_to_labels(sigmoid(logits))
    return {
        "gene_mcc": matthews_corrcoef(labels.ravel(), pred_labels.ravel())
    }

# Main function to train and test the model
def train_and_test(data_name, output_dir, model_name_or_path, tokenizer_path, max_length, batch_size, epochs, learning_rate, logging_strategy, evaluation_strategy, save_strategy, save_total_limit, weight_decay, metric_for_best_model, task_type):
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

    # Load tokenizer and add special tokens
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize the datasets
    tokenized_datasets = raw_datasets.map(lambda examples: tokenize_function(examples, tokenizer, max_length), batched=True)

    # Load model and configure it for sequence classification
    model = PlantGLMForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

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
        metric_for_best_model=metric_for_best_model,
        load_best_model_at_end=True,
        bf16=True,
        weight_decay=weight_decay,
    )

    # Select the appropriate metrics function based on the task type
    if task_type == 'regression':
        compute_metrics = compute_metrics_for_regression_task
    elif task_type == 'classification':
        compute_metrics = compute_metrics_for_classification_task
    elif task_type == 'segmentation':
        compute_metrics = compute_metrics_for_segmentation_task
    else:
        raise ValueError("Invalid task type. Choose from 'regression', 'classification', or 'segmentation'.")

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
    parser.add_argument('--metric_for_best_model', type=str, default='r2', help="Metric to determine the best model.")
    
    # Task type (regression, classification, segmentation)
    parser.add_argument('--task_type', type=str, required=True, choices=['regression', 'classification', 'segmentation'], help="Type of task (regression, classification, segmentation).")
    
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
        metric_for_best_model=args.metric_for_best_model,
        task_type=args.task_type
    )
