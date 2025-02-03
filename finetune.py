import os
import pandas as pd
import torch
from datasets import load_dataset,Dataset,DatasetDict,config
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast
from plantglm.modeling_plantglm import PlantGLMForSequenceClassification
from plantglm.configuration_plantglm import PlantGLMConfig
from transformers import EarlyStoppingCallback
from sklearn.metrics import r2_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config.HF_DATASETS_CACHE = "/scratch/huxuehai/qzzhang/finetune_hyena/datasets"
max_length = 172
model_name_or_path = "/scratch/huxuehai/lch/core_promoter/checkpoint-15000"
tokenizer_path = "/scratch/huxuehai/lch/core_promoter/tokenizer.json"

config = PlantGLMConfig.from_pretrained(model_name_or_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
 
def tokenize_function(examples):
    return tokenizer(
        examples['sequence'], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = r2_score(labels, predictions)
    return {"r2": rmse}

def train_and_test(species, output_dir):
    train_data = pd.read_csv('/scratch/huxuehai/lch/core_promoter/datas/' + species + '/train.csv')
    val_data = pd.read_csv('/scratch/huxuehai/lch/core_promoter/datas/' + species + '/val.csv')
    test_data = pd.read_csv('/scratch/huxuehai/lch/core_promoter/datas/' + species + '/test.csv')

    ds_train = Dataset.from_pandas(train_data)
    ds_valid = Dataset.from_pandas(val_data)
    ds_test = Dataset.from_pandas(test_data)

    raw_datasets = DatasetDict(
        {
            "train": ds_train,
            "valid": ds_valid,
            "test": ds_test, 
        }
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # 将模型改为二分类模型
    model = PlantGLMForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    training_args = TrainingArguments(
        output_dir='/scratch/huxuehai/lch/core_promoter/output/' + output_dir + '/',
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        save_strategy='epoch',
        num_train_epochs=50,
        save_total_limit=1,
        save_safetensors=False,
        learning_rate=1e-4,
        ddp_find_unused_parameters=False,
        metric_for_best_model='r2',  # 改为 accuracy
        load_best_model_at_end=True,
        bf16=True,
        weight_decay=0.001,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,

    )
    trainer.train()

    result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(result)

    return


if __name__ == "__main__":
    train_and_test(species='proto_with_enhancer', output_dir='proto_with_enhancer') 
    train_and_test(species='proto_no_enhancer', output_dir='proto_no_enhancer') 
    train_and_test(species='light_with_enhancer', output_dir='light_with_enhancer') 
    train_and_test(species='light_no_enhancer', output_dir='light_no_enhancer') 