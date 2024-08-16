import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import sklearn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class PathAndControlArguments:
    data_path: str = field(default=None)
    model_path: str = field(default=None)
    checkpoint_path: str = field(default=None)
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
    metric_for_best_model: str = field(default="auprc")
    

class SegmentDataset(Dataset):

    def __init__(self, 
            data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_length: int = 100,
        ):

        super(SegmentDataset, self).__init__()

        self.max_length = max_length
        data = pd.read_csv(data_path, delimiter="\t")
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

class GPT2Embd(torch.nn.Module):
    def __init__(self, pre_trained_path, max_len):
        super(GPT2Embd, self).__init__()
        self.len = max_len
        self.gpt2 = transformers.AutoModel.from_pretrained(pre_trained_path)

    def forward(self, input_ids):
        embd = self.gpt2(input_ids)["last_hidden_state"]

        return torch.reshape(embd,(-1,768,self.len))

class UNetHead(nn.Module):
    def __init__(self, max_len):
        super(UNetHead, self).__init__()
        self.len = max_len
        self.down_conv1 = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=3, padding=1)
        self.down_conv2 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        
        self.up_trans1 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        self.up_conv1 = nn.Conv1d(in_channels=3072, out_channels=1024, kernel_size=3, padding=1)
        self.up_trans2 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv1d(in_channels=1536, out_channels=8, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.down_conv1(x))
        x2 = F.max_pool1d(x1, kernel_size=2, stride=2)
        x3 = F.relu(self.down_conv2(x2))
        x4 = F.max_pool1d(x3, kernel_size=2, stride=2)

        x5 = self.up_trans1(x4)
        x6 = torch.cat([x5, x3], 1)
        x7 = F.relu(self.up_conv1(x6))
        x8 = self.up_trans2(x7)
        x9 = torch.cat([x8, x1], 1)
        x10 = self.up_conv2(x9)
        
        return torch.reshape(x10,(-1,8*self.len))


class SegmentationModel(torch.nn.Module):
    def __init__(self, path, max_len):
        super(SegmentationModel,self).__init__()
        self.pre_trained_path = path
        self.max_len = max_len
        self.gpt2embd = GPT2Embd(self.pre_trained_path, self.max_len)
        self.unethead = UNetHead(self.max_len)
        self.cdshead = nn.Sequential(nn.Linear(8*self.max_len,self.max_len), nn.Dropout(p=0.5, inplace=False))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, labels,**kwargs):
        x = self.gpt2embd(input_ids)
        x = self.unethead(x)
        x = self.cdshead(x)
        return {
            "loss":self.loss(x, labels),
            "predictions":x
        }

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logits_to_labels(x):
    pred_labels = np.zeros_like(x)
    pred_labels[x>=0.50] = 1

    return pred_labels

def compute_metrics_for_segmentation_task(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_logits, pred_labels = sigmoid(logits), logits_to_labels(logits)
    mcc = sklearn.metrics.matthews_corrcoef(labels.ravel(),pred_labels.ravel())
    auprc = sklearn.metrics.average_precision_score(pred_labels, pred_logits, average="micro")
    
    return {"mcc": mcc,"auprc": auprc}


def run_segmentation():
    parser = transformers.HfArgumentParser((PathAndControlArguments, TrainingArguments))
    path_and_control_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path_and_control_args.model_path,
        padding_side="right",
        use_fast=True,
    )

    train_dataset = SegmentDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(path_and_control_args.data_path, "train.tsv"),
        max_length=path_and_control_args.max_length
    )
    val_dataset = SegmentDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(path_and_control_args.data_path, "val.tsv"),
        max_length=path_and_control_args.max_length
    )

    model = SegmentationModel(path_and_control_args.model_path, path_and_control_args.max_length).to("cuda")

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_for_segmentation_task,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    run_segmentation()
