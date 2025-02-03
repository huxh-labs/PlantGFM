import os
import torch
import warnings
from datasets import load_dataset,config
from plantglm.modeling_plantglm import PlantGLMForCausalLM
from plantglm.configuration_plantglm import PlantGLMConfig
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
config.HF_DATASETS_CACHE = "./datasets"
max_length = 65538
model_name_or_path = "./output/8192/checkpoint-5000"
tokenizer_path = "./tokenizer.json"

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

datasets = load_dataset("text", data_files={"train":"./65k/65536cut/16sp_train.txt",
                                            "valid":"./65k/65536cut/16sp_dev.txt"})
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
# model = PlantGLMForCausalLM(config=config)

training_args = TrainingArguments(
    output_dir="./output/"+str(max_length-2)+"/",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    eval_strategy="steps",
    eval_steps=1250,
    logging_steps=1250,
    gradient_accumulation_steps=8,
    # num_train_epochs=30,
    max_steps=30000,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=6e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler_type="cosine",
    save_steps=1250,
    save_total_limit=24,
    save_safetensors=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    bf16=True,
    # push_to_hub=True,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        # data_collator=data_collator,
)
trainer.train()