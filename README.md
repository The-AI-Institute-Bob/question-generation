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

# Download the squad data 

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
 
# Download the quac data
```python
!wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json
!wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json
```

    --2021-09-27 13:30:24--  https://s3.amazonaws.com/my89public/quac/train_v0.2.json
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.67.94
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.67.94|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 68114819 (65M) [application/json]
    Saving to: ‘train_v0.2.json’

    train_v0.2.json     100%[===================>]  64.96M  17.8MB/s    in 4.2s    

    2021-09-27 13:30:29 (15.3 MB/s) - ‘train_v0.2.json’ saved [68114819/68114819]

    --2021-09-27 13:30:29--  https://s3.amazonaws.com/my89public/quac/val_v0.2.json
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.98.246
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.98.246|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 8929167 (8.5M) [application/json]
    Saving to: ‘val_v0.2.json’

    val_v0.2.json       100%[===================>]   8.51M  8.63MB/s    in 1.0s    

    2021-09-27 13:30:31 (8.63 MB/s) - ‘val_v0.2.json’ saved [8929167/8929167]


# Merge & format the data
```python
from data_formatter import format_data
from prepare_data import prepare_data
from run import run_qg

# example paths
training_data_quac = '/shared/shawn/q_g/question-generations/T5_Model/quac/train_v0.2.json'
test_data_quac = '/shared/shawn/q_g/question-generations/T5_Model/quac/val_v0.2.json'

training_data_squad = '/shared/shawn/q_g/question-generations/T5_Model/squad/train-v2.0.json'
test_data_squad = '/shared/shawn/q_g/question-generations/T5_Model/squad/dev-v2.0.json'

merged_training_data, merged_test_data = format_data(training_data_squad, test_data_squad,training_data_quac,test_data_quac)

```
    10/12/2021 18:06:19 - INFO - __main__ -   Fomatted data has been saved into /shared/shawn/q_g/question-generations/T5_Model


# Prepare the data


```python
args_dict = {
    'model_type': 'QAG',
    'train_file': 'merged_train.json',
    'valid_file': 'merged_test.json',

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
    "output_dir": "t5-base-qag-hl-merged",
    "train_file_path": "merged_train_data_QAG_T5.pt",
    "valid_file_path": "merged_valid_data_QAG_T5.pt",
    "per_device_train_batch_size": 32, 
    "per_device_eval_batch_size": 32, 
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 25,
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}

run_qg(args_dict)
```


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

model = pipeline('your_model_directory','QAG')

context = '''
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
'''

sent = 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.'

model(context, sent)
```
    'What were the Normans in the 10th and 11th centuries?'
    
# Benchmark the model
 - BLEU (BLEU-4 by default)
 - ROUGE
 
## Create datasets with generate questions:
```python
with open('squad_test.json','r') as f:
    data = json.load(f)
    
questions = list(data.keys())
contexts = list(data.values())

results = []
for q, a in zip(questions, contexts):
    results.append({'original_question':q,
                    'generated_question': model_(a['context'], a['answer']),
                    'contexts': a['context']})
    
    
with open('t5_results_from_merged_squad.json','w') as f:
    json.dump(results,f)
    

```

## Load the data
```python
with open('t5_results_from_merged_squad.json','r') as f:
    data = json.load(f)
```

## BLEU
```python
from datasets import load_metric
# metric = load_metric("rouge")
metric = load_metric("bleu")

model_predictions = [q[1].split() for q in questions]
references = [[q[0].split()]for q in questions]

merged_results = metric.compute(predictions=model_predictions, references=references)

merged_results['bleu']
```
    0.18645062269843635
    
## ROGUE
```python
from datasets import load_metric
metric = load_metric("rouge")

model_predictions = [q[1] for q in questions]
references = [q[0] for q in questions]

merged_results = metric.compute(predictions=model_predictions, references=references)

merged_results['rougeL'].mid.fmeasure
```
    0.4599304131354284




