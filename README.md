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

## 1. Environment

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
## 2. Pre-train

## 3. Fine-tune

#### 3.1 Classification and Regression

```bash
python fine_tune.py \
    --data_path './sample_data/classification' \
    --model_path /model \
    --problem_type 'classification' or 'egression'\
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

1. data_path: default=None,Path to the fine-tuning dataset.
2. model_path: default=None,Path to the pre-trained model.
3. checkpoint_path: default=None, Path to a saved checkpoint, if available, to resume training from a previous state.
4. problem_type: default="regression", Determines the type of task; it can be `regression`, `classification`.
5. max_length: default=1000, Maximum length of the input sequences. Inputs longer than this will be truncated.
6. run_name: default="run", Name of the training run, useful for organizing and distinguishing different experiments.
7. output_dir: default="./output", Directory where the output, including the trained model and logs, will be saved.
8. optim: default="adamw_hf",default is AdamW as implemented by Hugging Face.
9. per_device_train_batch_size: default=1, Batch size to use per device (e.g., per GPU) during training.
10. per_device_eval_batch_size: default=1, Batch size to use per device during evaluation.
11. num_train_epochs: default=1, Number of epochs to train the model.
12. fp16: default=False, Whether to use 16-bit floating point precision (FP16) for training to save memory and speed up computation.
13. bf16: default=False, Whether to use BFloat16 precision for training, similar to FP16 but with a larger dynamic range.
14. logging_strategy: default="epoch", Strategy for logging training information; options include `epoch` and `steps`.
15. save_strategy: default="epoch", Strategy for saving the model checkpoints; can be `epoch` or `steps`.
16. eval_strategy: default="epoch", Strategy for evaluating the model; options include `epoch` and `steps`.
17. lr_scheduler_type: default="linear", Type of learning rate scheduler to use.
18. warmup_steps: default=100, Number of steps for the learning rate warmup phase.
19. learning_rate: default=5e-5, Initial learning rate for the optimizer.
20. adam_beta1: default=0.9, The beta1 parameter for the Adam optimizer, affecting the first moment estimate.
21. adam_beta2: default=0.999, The beta2 parameter for the Adam optimizer, affecting the second moment estimate.
22. weight_decay: default=0.01, Weight decay rate for regularization to prevent overfitting.
23. gradient_accumulation_steps: default=1, Number of steps to accumulate gradients before updating the model parameters.
24. save_total_limit: default=1, Maximum number of checkpoints to keep; older ones will be deleted.
25. load_best_model_at_end: default=True, Whether to load the model with the best evaluation performance at the end of training.
26. metric_for_best_model: default="r2", Metric used to determine the best model; for regression tasks, this could be `r2`, and for classification, options include `accuracy`, `f1`, etc.


#### 3.2 Segmentation
To fine-tune the plant DNA LLMs, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_finetune.py` script to fine-tune a model for downstream task.  

Our script accepts `.csv` format data (separated by `,`) as input, when preparing the training data, please make sure the data contain a header and at least these two columns:
```csv
sequence,label
```
Where `sequence` is the input sequence, and `label` is the corresponding label for the sequence.

We also provide several plant genomic datasets for fine-tuning on the [HuggingFace](https://huggingface.co/zhangtaolab) and [ModelScope](https://www.modelscope.cn/organization/zhangtaolab).

With the appropriate supervised datasets, we can use the script to fine-tune a model for predicting promoters, for example:
```bash
python model_finetune.py \
    --model_name_or_path /path_to_the_model/plant-dnagpt \
    --train_data /path_to_the_data/train.csv \
    --test_data /path_to_the_data/test.csv \
    --eval_data /path_to_the_data/dev.csv \
    --train_task classification \
    --labels 'No;Yes' \
    --run_name plant_dnagpt_promoters \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --load_best_model_at_end \
    --metric_for_best_model 'f1' \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir finetune/plant-dnagpt-promoter
```

In this script:  
1. `--model_name_or_path`: Path to the foundation model you downloaded
2. `--train_data`: Path to the train dataset
3. `--test_data`: Path to the test dataset, omit it if no test data available
4. `--dev_data`: Path to the validation dataset, omit it if no validation data available
5. `--train_task`: Determine the task type, should be classification, multi-classification or regression
6. `--labels`: Set the labels for classification task, separated by `;`
7. `--run_name`: Name of the fine-tuned model
8. `--per_device_train_batch_size`: Batch size for training model
9. `--per_device_eval_batch_size`: Batch size for evaluating model
10. `--learning_rate`: Learning rate for training model
11. `--num_train_epochs`: Epoch for training model (also you can train model with steps, then you should change the strategies for save, logging and evaluation)
12. `--load_best_model_at_end`: Whether to load the model with the best performance on the evaluated data, default is `True`
13. `--metric_for_best_model`: Use which metric to determine the best model, default is `loss`, can be `accuracy`, `precison`, `recall`, `f1` or `matthews_correlation` for classification task, and `r2` or `spearmanr` for regression task
14. `--save_strategy`: Strategy for saving model, can be `epoch` or `steps`
15. `--logging_strategy`: Strategy for logging training information, can be `epoch` or `steps`
16. `--evaluation_strategy`: Strategy for evaluating model, can be `epoch` or `steps`
17. `--output_dir`: Where to save the fine-tuned model

Detailed descriptions of the arguments can be referred [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

## 3. Inference

To use a fine-tuned model for inference, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local or provide a model trained by yourself.  

We also provide a script named `model_inference.py` for model inference.  
Here is an example that use the script to predict histone modification:
```bash
# Directly input a sequence
python model_inference.py -m /path_to_the_model/plant-dnagpt-H3K27ac -s sequence
# Provide a file contains multiple sequences to predict
python model_inference.py -m /path_to_the_model/plant-dnagpt-H3K27ac -f /path_to_the_data/data.txt -o results/H3K27ac.txt
```

In this script:
1. `-m`: Path to the fine-tuned model that is used for inference
2. `-s`: Input DNA sequence, only nucleotide A, C, G, T, N are acceptable
3. `-f`: Input file that contain multiple sequences, one line for each sequence. If you want to keep more information, file with `,` of `\t` separator is acceptable, but a header contains `sequence` column must be specified.

Output results contains the original sequence, input sequence length, predicted label and probability of each label (for regression task, will show a predicted score).


## Docker implementation for model inference

Environment deployment for LLMs may be an arduous job. To simplify this process, we also provide a docker version of our model inference code.

The images of the docker version are [here](https://hub.docker.com/r/zhangtaolab/plant_llms_inference), and the usage of docker implementation is shown below.  
For GPU inference (with Nvidia GPU), please pull the image with `gpu` tag, and make sure your computer has install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```bash
docker pull zhangtaolab/plant_llms_inference:gpu
docker run --runtime=nvidia --gpus=all -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:gpu -h
```
```bash
usage: inference.py [-h] [-v] -m MODEL [-f FILE] [-s SEQUENCE] [-t THRESHOLD]
                    [-l MAX_LENGTH] [-bs BATCH_SIZE] [-p SAMPLE] [-seed SEED]
                    [-d {cpu,gpu,mps,auto}] [-o OUTFILE] [-n]

Script for Plant DNA Large Language Models (LLMs) inference

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m MODEL              Model path (should contain both model and tokenizer)
  -f FILE               File contains sequences that need to be classified
  -s SEQUENCE           One sequence that need to be classified
  -t THRESHOLD          Threshold for defining as True class (Default: 0.5)
  -l MAX_LENGTH         Max length of tokenized sequence (Default: 512)
  -bs BATCH_SIZE        Batch size for classification (Default: 1)
  -p SAMPLE             Subsampling for testing (Default: 1e7)
  -seed SEED            Random seed for subsampling (Default: None)
  -d {cpu,gpu,mps,auto}
                        Choose CPU or GPU to do inference (require specific
                        drivers) (Default: auto)
  -o OUTFILE            Prediction results (Default: stdout)
  -n                    Whether or not save the runtime locally (Default:
                        False)

Example:
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -f seqfile.csv -o output.txt
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt
```

For CPU inference,  please pull the image with `cpu` tag, this image support computer without NVIDIA GPU, such as cpu-only or Apple M-series Silicon. (Note that Inference of DNAMamba model is not supported in CPU mode)
```bash
docker pull zhangtaolab/plant_llms_inference:cpu
docker run -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:cpu -h
```

The detailed usage is the same as the section [Inference](#3-inference).

### Demo for open chormtain prediction
we also provide demo server for open chormtain prediction by using Plant DNAMamba model.  
The web application is accessible at https://bioinfor.yzu.edu.cn/llms/open-chromatin/ or http://llms.zhangtaolab.org/llms/open-chromatin.

Preview:

![gradio](imgs/gradio.jpeg)

