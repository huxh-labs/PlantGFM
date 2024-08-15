import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import sklearn
import scipy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class PathAndControlArguments:
    data_path: str = field(default=None)
    model_path: str = field(default=None)
    checkpoint_path: str = field(default=None)
    problem_type: str = field(default="regression")
    max_length: int = field(default=1000)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    run_name: str = field(default="run")
    output_dir: str = field(default="./output")
    optim: str = field(default="adamw_hf")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    logging_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(default=100)
    learning_rate: float = field(default=5e-5)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    weight_decay: float = field(default=0.01)
    gradient_accumulation_steps: int = field(default=1)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="r2")
    

class SupervisedDataset(Dataset):

    def __init__(self, 
            data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_length: int = 100,
        ):

        super(SupervisedDataset, self).__init__()

        self.max_length = max_length
        data = pd.read_csv(data_path)
        sequences = list(data.iloc[:,0].values)
        labels = [list(data.iloc[i,1:].values) for i in data.index]

        tokenized_data = tokenizer(
            sequences,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
        )

        self.input_ids = tokenized_data['input_ids']
        self.labels = labels
        self.num_labels = len(labels[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def compute_metrics_for_regression_task(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    r2 = sklearn.metrics.r2_score(labels, predictions)
    pcc, _ = scipy.stats.pearsonr(labels.flatten(),predictions.flatten())

    return {"r2": r2, "pcc":pcc}

def compute_metrics_for_classification_task(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    auc = sklearn.metrics.roc_auc_score(labels,logits,average="micro")
    auprc = sklearn.metrics.average_precision_score(labels,logits,average="micro")
    
    return {"auc": auc,"auprc": auprc}

metrics_func = {
    "regression":compute_metrics_for_regression_task,
    "classification":compute_metrics_for_classification_task,
}

def fine_tuning():
    parser = transformers.HfArgumentParser((PathAndControlArguments, TrainingArguments))
    path_and_control_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path_and_control_args.model_path,
        padding_side="right",
        use_fast=True,
    )

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(path_and_control_args.data_path, "train.csv"),
        max_length=path_and_control_args.max_length
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(path_and_control_args.data_path, "dev.csv"),
        max_length=path_and_control_args.max_length
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(path_and_control_args.data_path, "test.csv"),
        max_length=path_and_control_args.max_length
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        path_and_control_args.model_path,
        num_labels=train_dataset.num_labels,
    )
    model.config.problem_type = path_and_control_args.problem_type

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=metrics_func[path_and_control_args.problem_type],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    if path_and_control_args.checkpoint_path is not None:
        trainer.train(resume_from_checkpoint=path_and_control_args.checkpoint_path)
    else:
        trainer.train()
    result = trainer.evaluate(eval_dataset=test_dataset)
    print(result)

if __name__ == "__main__":
    fine_tuning()
