import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pickle
import torch
import nlp
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default="data/fquad",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    tokenizer: Optional[str] = field(
        default="airKlizz/t5-base-multi-fr-wiki-news",
        metadata={"help": "train dataset to cache"},
    )
    train_source: Optional[str] = field(
        default=None,
        metadata={"help": "train dataset to cache"},
    )
    valid_source: Optional[str] = field(
        default=None,
        metadata={"help": "valid dataset to cache"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"       
        self.sep_token = "<sep>"

  
    def process(self, dataset):
        dataset = dataset.map(self._add_eos_examples)        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
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


def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'




TASK_TO_FILTER_FN = {
    'e2e_qg': filter_e2e_qg,
}


def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    tokenizer = T5Tokenizer.from_pretrained(data_args.tokenizer)

    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    pickle.dump(data_args.train_source,open("train.p", 'wb'))
    pickle.dump(data_args.valid_source,open("valid.p", 'wb'))
    
    train_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.TRAIN)
    valid_dataset = nlp.load_dataset(data_args.dataset_path, name=data_args.qg_format, split=nlp.Split.VALIDATION)

    processor = DataProcessor(
        tokenizer,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = train_dataset.filter(filter_e2e_qg)
    valid_dataset = valid_dataset.filter(filter_e2e_qg)

    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    pickle.dump( valid_dataset, open( "valid.pkl", "wb" ) )
    pickle.dump( train_dataset, open( "train.pkl", "wb" ) )
    
    tokenizer_path = f"t5_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")
    
    os.remove('train.p')
    os.remove('valid.p')

if __name__ == "__main__":
    main()

