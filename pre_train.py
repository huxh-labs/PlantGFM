import os
import torch
import warnings
import argparse
from datasets import load_dataset, config
from plantglm.modeling_plantglm import PlantGLMForCausalLM
from plantglm.configuration_plantglm import PlantGLMConfig
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
config.HF_DATASETS_CACHE = "./datasets"

#  argparse 
def parse_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to training data")
    parser.add_argument('--dev_data_path', type=str, required=True, help="Path to validation data")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer")
    parser.add_argument('--max_length', type=int, default=65536, help="Maximum sequence length")
    parser.add_argument('--init_model_path', type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument('--per_device_train_batch_size', type=int, default=3, help="Train batch size per device")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=3, help="Eval batch size per device")
    parser.add_argument('--max_steps', type=int, default=30000, help="Maximum number of training steps")
    parser.add_argument('--logging_steps', type=int, default=1250, help="Number of steps between logs")
    parser.add_argument('--save_steps', type=int, default=1250, help="Number of steps between saving checkpoints")
    parser.add_argument('--eval_steps', type=int, default=1250, help="Number of steps between evaluations")
    parser.add_argument('--learning_rate', type=float, default=6e-4, help="Learning rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Adam beta1")
    parser.add_argument('--adam_beta2', type=float, default=0.95, help="Adam beta2")
    
    return parser.parse_args()


args = parse_args()


max_length = args.max_length
model_name_or_path = args.init_model_path
tokenizer_path = args.tokenizer_path


config = PlantGLMConfig.from_pretrained(model_name_or_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

def tokenize_function(examples):
    tokenized_sequence = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt", 
    )
    return {"input_ids": tokenized_sequence["input_ids"],"labels":tokenized_sequence["input_ids"]}

datasets = load_dataset("text", data_files={"train": args.train_data_path,
                                            "valid": args.dev_data_path})
column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
)

model = PlantGLMForCausalLM.from_pretrained(model_name_or_path, config=config)


training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=args.max_steps,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=args.learning_rate,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    lr_scheduler_type="cosine",
    save_steps=args.save_steps,
    save_total_limit=24,
    save_safetensors=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    bf16=True,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
)

trainer.train()
