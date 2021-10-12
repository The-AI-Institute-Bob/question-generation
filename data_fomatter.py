import json
import os
import logging
logger = logging.getLogger(__name__)
def foramt_training_data(training_data):
    train_data = {}
    for group in training_data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for a in qa['answers']:
                    answer = a['text']
                    train_data[question] = {'context': context, 'answer':answer}               
    return train_data

def foramt_test_data(test_data):
    val_data = {}
    for group in test_data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for a in qa['answers']:
                    answer = a['text']
                    val_data[question] = {'context': context, 'answer':answer}             
    return val_data


def merge_two_data(data1, data2):
    merged = data1.copy()
    merged.update(data2)
    return merged

def format_data(sqaud_training_data: str, 
                squad_test_data: str,
                quac_training_data: str,
                quac_test_data: str):
    '''
    Arguments:
        - sqaud_training_data: relative path to the squad training data. ('../squad/train-v2.0.json')
        - sqaud_training_data: relative path to the squad validation data. ('../squad/dev-v2.0.json')
        - quac_training_data: relative path to the quac training data. ('../quac/train.json')
        - quac_test_data: relative path to the quac validation data. ('../quac/dev.json')
    '''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # squad
    with open(sqaud_training_data,'r') as f:
        training_data_sqaud = json.load(f)
    with open(squad_test_data,'r') as f_:
        test_data_squad = json.load(f_)
        
    train_data_squad = foramt_training_data(training_data_sqaud)
    val_data_squad = foramt_test_data(test_data_squad)
    
    # quac
    with open(quac_training_data,'r') as f_:
        training_data_quac = json.load(f_)
    with open(quac_test_data,'r') as f_:
        test_data_quac = json.load(f_)
        
    train_data_quac = foramt_training_data(training_data_quac)
    val_data_quac = foramt_test_data(test_data_quac)
    
    # merge datasets
    merged_training_data = merge_two_data(train_data_squad, train_data_quac)
    merged_test_data = merge_two_data(val_data_squad, val_data_quac)

                                                    
    with open('merged_train.json','w') as f:
        json.dump(merged_training_data,f)
        
    with open('merged_test.json','w') as f:
        json.dump(merged_test_data,f)
        
    logger.info(f'Fomatted data has been saved into {os.getcwd()}')
    
    return merged_training_data, merged_test_data
    
    
