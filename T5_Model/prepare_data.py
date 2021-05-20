import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import torch
from transformers import T5Tokenizer, HfArgumentParser
from datasets import Dataset
import sys

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_type: Optional[str] = field(
        default='QG',
        metadata={"help": "if the model is trained for generation from text (QG) or generation from answer and text (QAG)"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for train dataset. Must be a .json file"},
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for valid dataset. Must be a .json file"},
    )

    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
        
    tokenizer: Optional[str] = field(
        default="t5-small",
        metadata={"help": "tokenizer used for the data"},
    )


class DataProcessor:
    def __init__(self, tokenizer, model_type = "QG", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.hl_token = "<hl>"
        self.sep_token = "<sep>"
        self.model_type = model_type

            
    def process(self, dataset):


        if self.model_type == "QG":
            dataset = dataset.map(self._add_special_tokens_QG)
        else:
            dataset = dataset.map(self._add_special_tokens_QAG)
        
        
        dataset = dataset.map(self._convert_to_features, batched= True)

        return dataset
  

    def _add_special_tokens_QG(self, ex):
        ex['source_text'] = f"generate questions: {ex['source_text']}"
        sub = ''
        if type(ex['target_text']) is list: #Preparing data with multiple questions per context
            for k in ex['target_text']:
                sub = sub + f"{k} <sep> "
            ex['target_text'] = sub 
        else: #Preparing data with one question per context
            ex['target_text'] = f"{ex['target_text']}"
        return ex

    def _add_special_tokens_QAG(self,ex): #This preprocessing is going to be modified later depending on the results I got
        ex['source_text'] = f"answer: {ex['answers']}  context: {ex['source_text']}"
        if type(ex['target_text']) is list:
            for k in ex['target_text']:
                sub = sub + f"{k} <sep> "
            ex['target_text'] = sub  
        else:
            ex['target_text'] = f"{ex['target_text']} </s>"
        return ex
    
    # encode the input
    def _convert_to_features(self, data):
        source_encoding = self.tokenizer.batch_encode_plus(
             data['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
             data['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings




def main(args_file=None):
    parser = HfArgumentParser((DataTrainingArguments,))

    
    #Parsing the arguments
    
    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        data_args = parser.parse_json_file(json_file=args_file_path)[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    #Loading the torch files

    with open(data_args.train_file) as f:
        train_dataset = json.load(f)

    if data_args.valid_file is not None:
        with open(data_args.valid_file) as f:
            valid_dataset = json.load(f)

    #Defining the tokenizer and adding the special tokens
    

    #If the input is a dict with keys being questions and item context:
    if 'source_text' not in train_dataset.keys():
        source_text = []
        target_text = []
        for source, target in train_dataset.items():
            source_text.append(source)
            target_text.append(target)
        train_dataset = {"source_text":source_text , "target_text":target_text}
        if data_args.valid_file is not None:
            source_text = []
            target_text = []
            for source, target in valid_dataset.items():
                source_text.append(source)
                target_text.append(target)
            valid_dataset = {"source_text":source_text , "target_text":target_text}
    
    
    tokenizer = T5Tokenizer.from_pretrained(data_args.tokenizer)

    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    # Processing the data
    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )



    train_dataset = Dataset.from_dict(train_dataset)
    train_dataset = processor.process(train_dataset)
    


    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    


    train_file = f"train_data_{data_args.model_type}_T5.pt"

    
    

    
    torch.save(train_dataset, train_file)
    logger.info(f"saved train dataset at {train_file}")
    
    if data_args.valid_file is not None:
        valid_dataset = Dataset.from_dict(valid_dataset)
        valid_dataset = processor.process(valid_dataset)        
        valid_dataset.set_format(type='torch', columns=columns)
        valid_file = f"valid_data_{data_args.model_type}_T5.pt"
        torch.save(valid_dataset, valid_file)
        logger.info(f"saved validation dataset at {valid_file}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")

def prepare_data(args_dict):
    with open("args_temp.json", 'w') as f:
        json.dump(args_dict, f)
    
    main(args_file="args_temp.json")
    os.remove("args_temp.json")

if __name__ == "__main__":
    main()