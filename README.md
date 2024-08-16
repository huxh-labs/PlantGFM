# PlantGLM

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![PyTorch Version](https://img.shields.io/badge/torch-2.0-red.svg)](https://pytorch.org/get-started/locally/)
[![Transformers Version](https://img.shields.io/badge/transformers-4.44-orange.svg)](https://huggingface.co/transformers/)
[![Accelerate Version](https://img.shields.io/badge/accelerate-0.33-yellow.svg)](https://huggingface.co/docs/accelerate/)

Welcome to the official repository for the paper "PlantGLM: A Genetic Language Model for Plant Genomics".

In this repository, you will find the following:

- Inference code for our models
- Pre-trained weights for all 9 NT models and 2 SegmentNT models
- Instructions for using the code and pre-trained models

## 1. Environment 🚀

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Create environment 

```bash
conda create -n glms python=3.8
conda activate glms
```

#### 1.3 Install dependencies

```bash
git clone --recursive https://github.com/hu-lab-PlantGLM/PlantGLM.git
cd PlantGLM
python3 -m pip install -r requirements.txt
```
## 2. Pre-train ✒️

If you want to retrain our model, you first need to download [PlantGLM](https://huggingface.co/hu-lab) locally from Hugging Face🤗.To ensure compatibility with our pre-training scripts, your data needs to be formatted according to the structure in the `/sample/pre-data` directory.

```bash
python pre_train.py \
    --train_data_path './sample_data/pre-train/train.txt' \
    --dev_data_path './sample_data/pre-train/dev.txt' \
    --tokenizer_path '/path/to/model'\
    --max_length 1024 \
    --init_model_path '/path/to/model' \
    --output_dir './output' \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --max_steps 10000 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \

```

In this script:  



1. **`train_data_path`**: default="./train.txt", Path to the training dataset.
2. **`dev_data_path`**: default="./dev.txt", Path to the validation dataset.
3. **`tokenizer_path`**: default="./gpt2", Path to the pre-trained tokenizer.
4. **`max_length`**: default=1024, Maximum length of the input sequences; inputs longer than this will be truncated.
5. **`init_model_path`**: default="./gpt2", Path to the pre-trained model that will be fine-tuned.
6. **`checkpoint_path`**: default=None, Path to a saved checkpoint, if available, to resume training from a previous state.
7. **`output_dir`**: default=None, Directory where the output, including the trained model and logs, will be saved.
8. **`optim`**: default="adamw_hf", Optimization algorithm to be used for training.
9. **`per_device_train_batch_size`**: default=10, Batch size per device during training.
10. **`per_device_eval_batch_size`**: default=10, Batch size per device during evaluation.
11. **`max_steps`**: default=10000, Maximum number of training steps.
12. **`fp16`**: default=True, Whether to use 16-bit (mixed) precision training instead of 32-bit.
13. **`bf16`**: default=False, Whether to use bfloat16 precision for training.
14. **`logging_strategy`**: default="steps", The strategy for logging; can be "steps" or "epoch".
15. **`logging_steps`**: default=1000, Number of update steps between logging events.
16. **`save_strategy`**: default="steps", The strategy for saving the model; can be "steps" or "epoch".
17. **`save_steps`**: default=1000, Number of update steps between model saves.
18. **`evaluation_strategy`**: default="steps", The strategy for evaluation; can be "steps" or "epoch".
19. **`eval_steps`**: default=1000, Number of update steps between evaluations.
20. **`lr_scheduler_type`**: default="cosine", The learning rate scheduler type; can be "linear", "cosine", "polynomial", etc.
21. **`warmup_steps`**: default=1000, Number of steps used for a linear warmup from 0 to the learning rate.
22. **`learning_rate`**: default=5e-4, The initial learning rate for training.
23. **`adam_beta1`**: default=0.9, Beta1 parameter for the AdamW optimizer.
24. **`adam_beta2`**: default=0.999, Beta2 parameter for the AdamW optimizer.
25. **`weight_decay`**: default=0.01, Weight decay to apply to the optimizer.
26. **`gradient_accumulation_steps`**: default=4, Number of steps to accumulate gradients before performing a backward/update pass.
27. **`save_total_limit`**: default=1, The maximum number of checkpoints to keep; older ones will be deleted.



## 3. Fine-tune ✏️
If you want to retrain our model, you first need to download PlantGLM locally from Hugging Face🤗.We recommend prioritizing the use of [PlantGLM-A](https://huggingface.co/hu-lab/PlantGLM-A) for sequence prediction tasks in coding regions, and [PlantGLM-AOZ](https://huggingface.co/hu-lab/PlantGLM-AOZ) for sequence prediction tasks in non-coding regions to achieve optimal performance.
#### 3.1 Classification and Regression

your dataset should be formatted as a CSV file with the following structure:
```csv
sequence,label
```
Ensure that your data follows this structure, similar to the examples provided in `/sample_data/classification` and `/sample_data/regression`, before proceeding with fine-tuning the model using the provided scripts.

```bash
python fine_tune.py \
    --data_path './sample_data/classification' \
    --model_path /path/to/model \
    --problem_type 'classification' or 'regression'\
    --eval_data /path_to_the_data/dev.csv \
    --max_length 170 \
    --output_dir './output' \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --bf16 True \
    --gradient_accumulation_steps 16 \
    --gradient_accumulation_steps 'acc' \
    --save_strategy epoch \
    --eval_strategy epoch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \

```

In this script:  

1. **`data_path`**: default=None,Path to the fine-tuning dataset.
2. **`model_path`**: default=None,Path to the pre-trained model.
3. **`checkpoint_path`**: default=None, Path to a saved checkpoint, if available, to resume training from a previous state.
4. **`max_length`**: default=1000, Maximum length of the input sequences. Inputs longer than this will be truncated.
6. **`run_name`**: default="run", Name of the training run, useful for organizing and distinguishing different experiments.
7. **`output_dir`**: default="./output", Directory where the output, including the trained model and logs, will be saved.
8. **`optim`**: default="adamw_hf",default is AdamW as implemented by Hugging Face.
9. **`per_device_train_batch_size`**: default=1, Batch size to use per device (e.g., per GPU) during training.
10. **`per_device_eval_batch_size`**: default=1, Batch size to use per device during evaluation.
11. **`num_train_epochs`**: default=1, Number of epochs to train the model.
12. **`fp16`**: default=False, Whether to use 16-bit floating point precision (FP16) for training to save memory and speed up computation.
13. **`bf16`**: default=False, Whether to use BFloat16 precision for training, similar to FP16 but with a larger dynamic range.
14. **`logging_strategy`**: default="epoch", Strategy for logging training information; options include `epoch` and `steps`.
15. **`save_strategy`**: default="epoch", Strategy for saving the model checkpoints; can be `epoch` or `steps`.
16. **`eval_strategy`**: default="epoch", Strategy for evaluating the model; options include `epoch` and `steps`.
17. **`lr_scheduler_type`**: default="linear", Type of learning rate scheduler to use.
18. **`warmup_steps`**: default=100, Number of steps for the learning rate warmup phase.
19. **`learning_rate`**: default=5e-5, Initial learning rate for the optimizer.
20. **`adam_beta1`**: default=0.9, The beta1 parameter for the Adam optimizer, affecting the first moment estimate.
21. **`adam_beta2`**: default=0.999, The beta2 parameter for the Adam optimizer, affecting the second moment estimate.
22. **`weight_decay`**: default=0.01, Weight decay rate for regularization to prevent overfitting.
23. **`gradient_accumulation_steps`**: default=1, Number of steps to accumulate gradients before updating the model parameters.
24. **`save_total_limit`**: default=1, Maximum number of checkpoints to keep; older ones will be deleted.
25. **`load_best_model_at_end`**: default=True, Whether to load the model with the best evaluation performance at the end of training.
26. **`metric_for_best_model`**: default="r2", Metric used to determine the best model; for regression tasks, this could be `r2`, and for classification, options include `acc`, etc.


#### 3.2 Segmentation

```csv
sequence,label
```

```bash
python segment.py \
    --data_path './sample_data/pre-train' \
    --model_path /path/to/model \
    --max_length 1000 \
    --output_dir './output' \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --bf16 True \
    --metric_for_best_model 'auprc' \
    --save_strategy epoch \
    --eval_strategy epoch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \

```

In this script:  

1. **`data_path`**: default=None,Path to the fine-tuning dataset.
2. **`model_path`**: default=None,Path to the pre-trained model.
3. **`checkpoint_path`**: default=None, Path to a saved checkpoint, if available, to resume training from a previous state.
4. **`problem_type`**: default="regression", Determines the type of task; it can be `regression`, `classification`.
5. **`max_length`**: default=1000, Maximum length of the input sequences. Inputs longer than this will be truncated.
6. **`run_name`**: default="run", Name of the training run, useful for organizing and distinguishing different experiments.
7. **`output_dir`**: default="./output", Directory where the output, including the trained model and logs, will be saved.
8. **`optim`**: default="adamw_hf",default is AdamW as implemented by Hugging Face.
9. **`per_device_train_batch_size`**: default=1, Batch size to use per device (e.g., per GPU) during training.
10. **`per_device_eval_batch_size`**: default=1, Batch size to use per device during evaluation.
11. **`num_train_epochs`**: default=1, Number of epochs to train the model.
12. **`fp16`**: default=False, Whether to use 16-bit floating point precision (FP16) for training to save memory and speed up computation.
13. **`bf16`**: default=False, Whether to use BFloat16 precision for training, similar to FP16 but with a larger dynamic range.
14. **`logging_strategy`**: default="epoch", Strategy for logging training information; options include `epoch` and `steps`.
15. **`save_strategy`**: default="epoch", Strategy for saving the model checkpoints; can be `epoch` or `steps`.
16. **`eval_strategy`**: default="epoch", Strategy for evaluating the model; options include `epoch` and `steps`.
17. **`lr_scheduler_type`**: default="linear", Type of learning rate scheduler to use.
18. **`warmup_steps`**: default=100, Number of steps for the learning rate warmup phase.
19. **`learning_rate`**: default=5e-5, Initial learning rate for the optimizer.
20. **`adam_beta1`**: default=0.9, The beta1 parameter for the Adam optimizer, affecting the first moment estimate.
21. **`adam_beta2`**: default=0.999, The beta2 parameter for the Adam optimizer, affecting the second moment estimate.
22. **`weight_decay`**: default=0.01, Weight decay rate for regularization to prevent overfitting.
23. **`gradient_accumulation_steps`**: default=1, Number of steps to accumulate gradients before updating the model parameters.
24. **`save_total_limit`**: default=1, Maximum number of checkpoints to keep; older ones will be deleted.
25. **`load_best_model_at_end`**: default=True, Whether to load the model with the best evaluation performance at the end of training.
26. **`metric_for_best_model`**: default="auprc",Metric used to determine the best model, you can choose between `auprc` for precision-recall evaluation or `mcc` for mean accuracy





Feel free to contact us if you have any questions or suggestions regarding the code and models.