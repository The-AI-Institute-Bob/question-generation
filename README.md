# question-generation

T5-based answer-aware question generation model. 
- Prepare the squad datasets (2.0)
- Train the question-generation model
- Use trained model to generatew questions

# Further actions:
- Integrate Google API for entity extraction
- Integrate summarization algorithm

# Required packaged

The required packages can be found in requirements.txt

# Input format

The input dataset format should be as follow:

```python
data = {'When did Beyonce start becoming popular?': {'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
  'answer': 'in the late 1990s'},
 'What areas did Beyonce compete in when she was growing up?': {'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
  'answer': 'singing and dancing'}
```

Any datasets sharing this format work.


# Preparing Squad Data
- Create a sub data folder
- Donwload squad datasets 2.0

# Download the data 

```python
!mkdir squad
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

    --2021-09-20 13:51:58--  https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    Resolving rajpurkar.github.io (rajpurkar.github.io)... 185.199.109.153, 185.199.110.153, 185.199.111.153, ...
    Connecting to rajpurkar.github.io (rajpurkar.github.io)|185.199.109.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 42123633 (40M) [application/json]
    Saving to: ‘squad/train-v2.0.json’

    squad/train-v2.0.js 100%[===================>]  40.17M   203MB/s    in 0.2s    

    2021-09-20 13:52:00 (203 MB/s) - ‘squad/train-v2.0.json’ saved [42123633/42123633]

    --2021-09-20 13:52:01--  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    Resolving rajpurkar.github.io (rajpurkar.github.io)... 185.199.108.153, 185.199.111.153, 185.199.110.153, ...
    Connecting to rajpurkar.github.io (rajpurkar.github.io)|185.199.108.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4370528 (4.2M) [application/json]
    Saving to: ‘dev-v2.0.json.1’

    dev-v2.0.json.1     100%[===================>]   4.17M  --.-KB/s    in 0.01s   

    2021-09-20 13:52:01 (283 MB/s) - ‘dev-v2.0.json.1’ saved [4370528/4370528]
    
    

# Format the data
```python
from data_formatter import format_data
from prepare_data import prepare_data
from run import run_qg

format_data('train-v2.0.json','dev-v2.0.json')

```
    09/21/2021 16:43:14 - INFO - data_fomatter -   Fomatted data has been saved into /shared/shawn/q_g/question-generations/T5_Model


# Prepare the data


```python
args_dict = {
    'model_type': 'QAG',
    'train_file': 'squad_train.json',
    'valid_file': 'squad_test.json',

    }

prepare_data(args_dict)
```
    
    

# Training script


```python
from run import run_qg

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "QAG_qg_tokenizer",
    "output_dir": "t5-base-qag-hl-batch20",
    "train_file_path": "train_data_QAG_T5.pt",
    "valid_file_path": "valid_data_QAG_T5.pt",
    "per_device_train_batch_size": 20, # 32
    "per_device_eval_batch_size": 20, # 32
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

    09/01/2021 13:30:16 - WARNING - run -   Process rank: -1, device: cuda:0, n_gpu: 1, distributed training: False, 16-bits training: False
    09/01/2021 13:30:16 - INFO - run -   Training/evaluation parameters TrainingArguments(
    _n_gpu=1,
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=None,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    do_eval=True,
    do_predict=False,
    do_train=True,
    eval_accumulation_steps=None,
    eval_steps=None,
    evaluation_strategy=IntervalStrategy.NO,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    gradient_accumulation_steps=8,
    greater_is_better=None,
    group_by_length=False,
    ignore_data_skip=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=0.0001,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=-1,
    log_level=-1,
    log_level_replica=-1,
    log_on_each_node=True,
    logging_dir=t5-base-qag-hl-batch20/runs/Sep01_13-30-16_ip-172-31-10-76,
    logging_first_step=False,
    logging_steps=100,
    logging_strategy=IntervalStrategy.STEPS,
    lr_scheduler_type=SchedulerType.LINEAR,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    no_cuda=False,
    num_train_epochs=20,
    output_dir=t5-base-qag-hl-batch20,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=20,
    per_device_train_batch_size=20,
    prediction_loss_only=True,
    push_to_hub=False,
    push_to_hub_model_id=t5-base-qag-hl-batch20,
    push_to_hub_organization=None,
    push_to_hub_token=None,
    remove_unused_columns=True,
    report_to=['tensorboard'],
    resume_from_checkpoint=None,
    run_name=t5-base-qag-hl-batch20,
    save_on_each_node=False,
    save_steps=500,
    save_strategy=IntervalStrategy.STEPS,
    save_total_limit=None,
    seed=42,
    sharded_ddp=[],
    skip_memory_metrics=True,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_legacy_prediction_loop=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
    )
    09/01/2021 13:30:19 - INFO - run -   loading dataset
    09/01/2021 13:30:21 - INFO - run -   finished loading dataset
    



# Testing the model

## Generate questions based on keyword-based answer
```python
from pipeline import pipeline

model = pipeline('t5-base-qag-hl-batch20','QAG')
```


```python
contexts = '''
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.
'''
sents = ['Normans ', 'France']
for sent in sents:
    print(model(contexts, sent))
```

    What were the people who gave their name to Normandy in the 10th and 11th centuries?
    Where is Normandy located?
    
    
## Generate questions based on sentences that contains answers
```python
from pipeline import pipeline

model = pipeline('t5-base-qag-hl-batch20','QAG')

context = '''
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
'''

sent = 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.'

model(context, sent)
```
    'What were the Normans in the 10th and 11th centuries?'
