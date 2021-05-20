# Required packaged


```python
!pip install datasets
```


```python
!pip install transformers
```


```python
!pip install sentencepiece
```

# Input format


```python
toy_dataset = {'President Obama was born in 1961': 'When was Obama born ?',
                  'The french revolution began in 1789':'When did the French revolution begin ?',
                  'Astatine is the rarest naturally occurring element on Earth':'What is the rarest material in the world ?'              
               }
```

Questions can also be a list of questions


```python
import json
with open('toy_dataset2.json', 'w') as f:
    json.dump(toy_dataset,f)
```

# Preparing Squad Data

La partie ci-dessous ne concerne que le fine-tuning sur le dataset squad


```python
!mkdir squad
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad/dev-v2.0.json
```

    --2021-05-20 14:52:32--  https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    Resolving rajpurkar.github.io (rajpurkar.github.io)... 185.199.108.153, 185.199.109.153, 185.199.110.153, ...
    Connecting to rajpurkar.github.io (rajpurkar.github.io)|185.199.108.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 42123633 (40M) [application/json]
    Saving to: ‘squad/train-v2.0.json’
    
    squad/train-v2.0.js 100%[===================>]  40.17M   153MB/s    in 0.3s    
    
    2021-05-20 14:52:35 (153 MB/s) - ‘squad/train-v2.0.json’ saved [42123633/42123633]
    
    --2021-05-20 14:52:35--  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    Resolving rajpurkar.github.io (rajpurkar.github.io)... 185.199.108.153, 185.199.109.153, 185.199.110.153, ...
    Connecting to rajpurkar.github.io (rajpurkar.github.io)|185.199.108.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4370528 (4.2M) [application/json]
    Saving to: ‘squad/dev-v2.0.json’
    
    squad/dev-v2.0.json 100%[===================>]   4.17M  --.-KB/s    in 0.08s   
    
    2021-05-20 14:52:35 (53.4 MB/s) - ‘squad/dev-v2.0.json’ saved [4370528/4370528]
    
    


```python
import json
from pathlib import Path
import numpy as np

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    qid = []
    dataset = {}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            dataset[context] = [qa['question'] for qa in passage['qas']]




    return dataset

train_dataset = read_squad('squad/train-v2.0.json')
val_dataset = read_squad('squad/dev-v2.0.json')
```


```python
with open('train_dataset.json','w') as f:
  json.dump(train_dataset,f)
with open('val_dataset.json','w') as f:
  json.dump(val_dataset,f)
```


```python
from prepare_data import prepare_data
from run import run_qg
```

# Preparing the data


```python
args_dict = {
    'model_type': 'QG',
    'train_file': 'train_dataset.json',
    'valid_file': 'val_dataset.json'
    }

prepare_data(args_dict)
```
    
    

# Training script


```python
from run import run_qg

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "QG_qg_tokenizer",
    "output_dir": "t5-base-qg-hl",
    "train_file_path": "train_data_QG_T5.pt",
    "valid_file_path": "valid_data_QG_T5.pt",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 20,
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}

# start training
run_qg(args_dict)
```

    05/19/2021 13:54:22 - WARNING - drive.MyDrive.T5_Model.run -   Process rank: -1, device: cuda:0, n_gpu: 1, distributed training: False, 16-bits training: False
    05/19/2021 13:54:22 - INFO - drive.MyDrive.T5_Model.run -   Training/evaluation parameters TrainingArguments(output_dir=t5-base-qg-hl, overwrite_output_dir=False, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=True, per_device_train_batch_size=32, per_device_eval_batch_size=32, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=0.0001, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=20, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/May19_13-54-22_0f8bb17faeca, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=100, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=[], dataloader_drop_last=False, eval_steps=100, dataloader_num_workers=0, past_index=-1, run_name=t5-base-qg-hl, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, use_legacy_prediction_loop=False, push_to_hub=False, resume_from_checkpoint=None, _n_gpu=1, mp_parameters=)
    05/19/2021 13:54:23 - INFO - drive.MyDrive.T5_Model.run -   loading dataset
    05/19/2021 13:54:24 - INFO - drive.MyDrive.T5_Model.run -   finished loading dataset
    



# Testing the model


```python
from pipeline import pipeline

nlp = pipeline('valhalla/t5-small-e2e-qg')
```


```python
nlp("Python is a programming language. Created by Guido van Rossum and first released in 1991.")
```




    ['What is a programming language?',
     'Who created Python?',
     'When was Python first released?']


