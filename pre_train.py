import os
import torch
import warnings
import argparse
from datasets import load_dataset, config
from plantgfm.modeling_plantgfm import PlantGFMForCausalLM
from plantgfm.configuration_plantgfm import PlantGFMConfig
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
config.HF_DATASETS_CACHE = "./datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_data_path', type=str, default='./sample_data/pre-train/train.txt', help="Path to training data")
    parser.add_argument('--dev_data_path', type=str, default='./sample_data/pre-train/dev.txt', help="Path to validation data")
    parser.add_argument('--tokenizer_path', type=str, default='/path/to/model', help="Path to the tokenizer")
    parser.add_argument('--init_model_path', type=str, default='/path/to/initial/model', help="Path to the initial model")
    parser.add_argument('--max_length', type=int, default=65538, help="Maximum sequence length")
    parser.add_argument('--output_dir', type=str, default='./output', help="Output directory for model checkpoints")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Train batch size per device")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Eval batch size per device")
    parser.add_argument('--max_steps', type=int, default=30000, help="Maximum number of training steps")
    parser.add_argument('--logging_steps', type=int, default=1250, help="Number of steps between logs")
    parser.add_argument('--save_steps', type=int, default=1250, help="Number of steps between saving checkpoints")
    parser.add_argument('--eval_steps', type=int, default=1250, help="Number of steps between evaluations")
    parser.add_argument('--learning_rate', type=float, default=6e-4, help="Learning rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=24, help="Gradient accumulation steps")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Adam beta1")
    parser.add_argument('--adam_beta2', type=float, default=0.95, help="Adam beta2")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="Warmup steps")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "constant"], help="LR scheduler type")
    parser.add_argument('--save_total_limit', type=int, default=24, help="Total number of saved checkpoints")
    parser.add_argument('--save_safetensors', type=bool, default=False, help="Whether to save safetensors")
    parser.add_argument('--ddp_find_unused_parameters', type=bool, default=False, help="Whether to find unused parameters in DDP")
    parser.add_argument('--gradient_checkpointing', type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument('--bf16', type=bool, default=True, help="Use bf16 precision")

    return parser.parse_args()


args = parse_args()

max_length = args.max_length
model_name_or_path = args.init_model_path  
tokenizer_path = args.tokenizer_path



config = PlantGFMConfig.from_pretrained(model_name_or_path)
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

model = PlantGFMForCausalLM.from_pretrained(model_name_or_path, config=config)



training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=args.max_steps,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    lr_scheduler_type=args.lr_scheduler_type,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    save_safetensors=args.save_safetensors,
    ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    gradient_checkpointing=args.gradient_checkpointing,
    bf16=args.bf16,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
)


trainer.train()
