import json
import os
import logging
logger = logging.getLogger(__name__)
def format_data(training_data: str, test_data: str):
    '''
    Arguments:
        - training data: relative path to the training data. ('../squad/train-v2.0.json')
        - training data: relative path to the validation data. ('../squad/dev-v2.0.json')
    '''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    with open(training_data,'r') as f:
        training_data = json.load(f)
    
    with open(test_data,'r') as f_:
        data_test = json.load(f_)
    
    
    train_data = {}
    for group in training_data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for a in qa['answers']:
                    answer = a['text']
                    train_data[question] = {'context': context, 'answer':answer}
                    
    val_data = {}
    for group in data_test['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for a in qa['answers']:
                    answer = a['text']
                    val_data[question] = {'context': context, 'answer':answer}
                    
                    
                                    
    with open('squad/squad_train.json','w') as f:
        json.dump(train_data,f)
        
    with open('squad/squad_test.json','w') as f:
        json.dump(val_data,f)
        
    logger.info(f'Fomatted data has been saved into {os.getcwd()}')
    
    
    