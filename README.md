# PlantGFM

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![PyTorch Version](https://img.shields.io/badge/torch-2.0-red.svg)](https://pytorch.org/get-started/locally/)
[![Transformers Version](https://img.shields.io/badge/transformers-4.44-orange.svg)](https://huggingface.co/transformers/)
[![Accelerate Version](https://img.shields.io/badge/accelerate-0.33-yellow.svg)](https://huggingface.co/docs/accelerate/)

Welcome to the official repository for the paper "PlantGFM: A Genetic Foundation Model for Discovery and Creation of Plant Genes".

In this repository, you will find the following:

- Comprehensive guidelines for both pre-training and fine-tuning the models, including preprocessing steps and handling special cases in your data.
- Resources and example scripts to assist you in preparing your data and running the models for various tasks.
- The related code is in `./plantgfm`:
  - **`configuration_plantgfm.py`**: This is the configuration file for the PlantGFM model.
  - **`modeling_plantgfm.py`**: This file contains the model architecture for PlantGFM, along with code for classification and regression tasks.
  - **`modeling_segmentgfm.py`**: This script is focused on gene prediction tasks.

## 1. Environment 🚀

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Create environment 

```bash
conda create -n gfms python=3.8
conda activate gfms
```

#### 1.3 Install dependencies

```bash
git clone --recursive https://github.com/hu-lab-PlantGFM/PlantGFM.git
cd PlantGFM
python3 -m pip install -r requirements.txt
```
## 2. Pre-train ✒️

If you want to retrain our model, you first need to download [PlantGFM](https://huggingface.co/hu-lab) locally from Hugging Face🤗.To ensure compatibility with our pre-training scripts, your data needs to be formatted according to the structure in the `/sample/pre-data` directory.

```bash
python pre_train.py \
    --train_data_path './sample_data/pre-train/train.txt' \
    --dev_data_path './sample_data/pre-train/dev.txt' \
    --tokenizer_path './tokenizer.json' \
    --max_length 65538 \
    --init_model_path '/path/to/model'
    --output_dir './output' \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --max_steps 30000 \
    --logging_steps 1250 \
    --save_steps 1250 \
    --eval_steps 1250 \
    --learning_rate 6e-4 \
    --gradient_accumulation_steps 24 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --save_total_limit 24 \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --bf16 True


```

In this script:  

1. **`train_data_path`**: default="./sample_data/pre-train/train.txt", Path to training data.
2. **`dev_data_path`**: default="./sample_data/pre-train/dev.txt", Path to validation data.
3. **`tokenizer_path`**: default="/path/to/model", Path to the tokenizer.
4. **`max_length`**: default=65538, Maximum length of input sequences, increased by 2 from the previous default value.
5. **`output_dir`**: default="./output", Output directory for model checkpoints.
6. **`per_device_train_batch_size`**: default=1, Train batch size per device.
7. **`per_device_eval_batch_size`**: default=1, Eval batch size per device.
8. **`max_steps`**: default=30000, Maximum number of training steps.
9. **`logging_steps`**: default=1250, Number of steps between logs.
10. **`save_steps`**: default=1250, Number of steps between saving checkpoints.
11. **`eval_steps`**: default=1250, Number of steps between evaluations.
12. **`learning_rate`**: default=6e-4, Learning rate.
13. **`gradient_accumulation_steps`**: default=24, Gradient accumulation steps.
14. **`adam_beta1`**: default=0.9, Adam beta1.
15. **`adam_beta2`**: default=0.95, Adam beta2.
16. **`weight_decay`**: default=0.1, Weight decay.
17. **`warmup_steps`**: default=1000, Warmup steps.
18. **`lr_scheduler_type`**: default="cosine", LR scheduler type (choices: ["linear", "cosine", "constant"]).
19. **`save_total_limit`**: default=24, Total number of saved checkpoints.
20. **`save_safetensors`**: default=False, Whether to save safetensors.
21. **`ddp_find_unused_parameters`**: default=False, Whether to find unused parameters in DDP.
22. **`gradient_checkpointing`**: default=True, Enable gradient checkpointing.
23. **`bf16`**: default=True, Use bf16 precision.
24. **`init_model_path`**: default="/path/to/model", Path to the pre-trained model .



## 3. Fine-tune ✏️
If you want to fine-tune our model, please take note of the following:🔍


-**`Sequence Preprocessing`**: The sequences need to be converted into individual nucleotides. For example, the sequence "ATCGACCT" should be processed into "A T C G A C C T". between single nucleotides.

-**`Handling  Other Bases`** :  Although our model was pre-trained on the bases 'A', 'T', 'C', 'G', and 'N', it can also handle a small amount of other characters.

####  Classification and Regression

For both classification and regression tasks,your dataset should be formatted as a CSV file with the following structure:
 ```csv
sequence,labels
```

#### Segmentation

For segmentation tasks, your dataset should be formatted as a TSV file with the following structure:
 ```tsv
sequence    gene_0    gene_1    ...    gene_65536
```
Ensure that your data follows this structure, similar to the examples provided in `/sample_data/segmentation`, before proceeding with fine-tuning the model using the provided scripts.

Ensure that your data follows this structure, similar to the examples provided in `/sample_data/classification` and `/sample_data/regression`, before proceeding with fine-tuning the model using the provided scripts.

```bash
python fine_tune.py
  --data_name '/sample_data/regression'
  --output_dir '/output' \
  --model_name_or_path 'path/to/model' \
  --tokenizer_path './tokenizer.json' \
  --max_length 172 \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 1e-4 \
  --logging_strategy 'epoch' \
  --evaluation_strategy 'epoch' \
  --save_strategy 'epoch' \
  --save_total_limit 3 \
  --weight_decay 0.01 \
  --metric_for_best_model 'r2' \
  --task_type 'regression'

```

In this script:  

1. **`data_name`**: default=None, The name or path of the dataset for training, validation, and testing.
2. **`output_dir`**: default=None, Directory to save model checkpoints and logs.
3. **`model_name_or_path`**: default=None, Path to the pre-trained model or model name from Hugging Face Model Hub.
4. **`tokenizer_path`**: default=None, Path to the tokenizer used for text preprocessing.
5. **`max_length`**: default=172, Maximum sequence length for tokenization. Sequences longer than this are truncated.
6. **`batch_size`**: default=96, Batch size for training and evaluation.
7. **`epochs`**: default=20, Number of training epochs (full passes through the dataset).
8. **`learning_rate`**: default=1e-4, Learning rate for training.
9. **`logging_strategy`**: default='epoch', choices=['steps', 'epoch'], How frequently to log training progress ('steps' or 'epoch').
10. **`evaluation_strategy`**: default='epoch', choices=['steps', 'epoch'], How frequently to evaluate the model ('steps' or 'epoch').
11. **`save_strategy`**: default='epoch', choices=['steps', 'epoch'], How frequently to save model checkpoints ('steps' or 'epoch').
12. **`save_total_limit`**: default=1, Maximum number of checkpoints to save.
13. **`weight_decay`**: default=0.001, Weight decay used to prevent overfitting.
14. **`task_type`**: default=None, Type of task ('segmentation'、'regression' or 'classification').








Feel free to contact us if you have any questions or suggestions regarding the code and models.
