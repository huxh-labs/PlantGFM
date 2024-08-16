import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,  
    AutoConfig, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments, 
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class PathAndControlArguments:
    train_data_path: str = field(default="./train.txt")
    dev_data_path: str = field(default="./dev.txt")
    tokenizer_path: str = field(default="./gpt2")
    max_length: int = field(default=1024)
    init_model_path: str = field(default="./gpt2")
    checkpoint_path: str = field(default=None)

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default=None)
    optim: str = field(default="adamw_hf") 
    per_device_train_batch_size: int = field(default=10)
    per_device_eval_batch_size: int = field(default=10)
    max_steps: int = field(default=10000)
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=1000)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=1000)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    learning_rate: float = field(default=5e-4)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    weight_decay: float = field(default=0.01)
    gradient_accumulation_steps: int = field(default=4)
    save_total_limit: int = field(default=1)


def pre_training():
    parser = HfArgumentParser((PathAndControlArguments, TrainingArguments))
    path_and_control_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(path_and_control_args.tokenizer_path)
    datasets = load_dataset("text", data_files={"train":path_and_control_args.train_data_path,
                                                "validation":path_and_control_args.dev_data_path})
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = datasets.map(
            lambda examples: tokenizer(
                    [line for line in examples["text"] if len(line) > 0 and not line.isspace()], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=path_and_control_args.max_length, 
                    return_tensors="pt",
            ),
            batched=True,
            num_proc=None,
            remove_columns=[text_column_name],
            load_from_cache_file=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    config = AutoConfig.from_pretrained(path_and_control_args.init_model_path,
                                        vocab_size=len(tokenizer),
                                        n_ctx=path_and_control_args.max_length,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=path_and_control_args.init_model_path,
                                                 config=config)
    model.resize_token_embeddings(len(tokenizer))
    

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
    )
    if path_and_control_args.checkpoint_path is not None:
        trainer.train(resume_from_checkpoint=path_and_control_args.checkpoint_path)
    else:
        trainer.train()

if __name__ == "__main__":
    pre_training()
