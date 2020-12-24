import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers import HfArgumentParser
from .prepare_data import prepare
import sys
import json
logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    gradient_accumulation_steps: int= field(
        default = 8,
        metadata={"help": "number of steps for gradient accumulation"},
    )
    
    learning_rate: int= field(
        default=1e-4,
        metadata={'help': 'learning rate'},
    )
        
    num_train_epochs: int=field(
        default=10,
        metadata={'help': 'number of epochs'}
    )
    
    seed: int=field(
        default = 42,
        metadata={'help': 'for reproductible results'},
    )
    
    do_train: bool=field(
        default= True,
        metadata={},
    )
    do_eval: bool=field(
        default= True,
        metadata={}
    )
    
    evaluate_during_training: bool=field(
        default= True,
        metadata={},
    )
    
    logging_steps: int=field(
        default = 100,
        metadata={},
    )
    per_device_train_batch: int= field(
        default = 32,
        metadata={"help": "size of the training batch"},
    )
        
    per_device_eval_batch: int= field(
        default = 32,
        metadata={"help": "size of the evaluation batch"},
    )
    model_type: str = field(
        default= "t5",
        metadata={"help": "type of the model, only t5 available for now"},
    )
        
    tokenizer_name_or_path: str=field(
        default= "t5_qg_tokenizer",
        metadata={"help": "path to the saved tokenizer when caching the dataset or the HF tokenizer"},
    )
    output_dir: str=field(
        default= None,
        metadata={"help": "location of the saved model"}
    )
    train_file_path: str = field(
        default= os.path.dirname(__file__)+"/data/train_data.pt",
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        default = os.path.dirname(__file__)+"/data/valid_data.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"}, 
    )
    model_name_or_path: str = field(
        default = "t5-base",
        metadata={'help' : 'Model used for finetuning'},
    )
                  
    task: Optional[str] = field(
        default=None,
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )


    def to_dict(self):
        return {
               'gradient_accumulation_steps': self.gradient_accumulation_steps,
               'learning_rate': self.learning_rate,
               'num_train_epochs': self.num_train_epochs,
               'seed': self.seed,
               'do_train': self.do_train,
               'do_eval': self.do_eval,
               'evaluate_during_training' : self.evaluate_during_training,
               'logging_steps' : self.logging_steps,
               'per_device_train_batch' : self.per_device_train_batch,
               'per_device_eval_batch' : self.per_device_eval_batch,
               'model_type': self.model_type,
               'tokenizer_name_or_path': self.tokenizer_name_or_path,
               'output_dir': self.output_dir,
               'train_file_path': self.train_file_path,
               'valid_file_path': self.valid_file_path,
               'data_dir': self.data_dir,
               'task': self.task,
               'model_name_or_path': self.model_name_or_path,
        }
               
                

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=os.path.dirname(__file__)+ "/data/data_processing",
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



def main(args_file=None):
    
    parser = HfArgumentParser((DataTrainingArguments, TrainingArgs))
    
    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()
    

    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    
    prepare(data_args)
        
    
    from .run_qg import run_qg
    run_qg(training_args.to_dict())

def run(args_dict):
    with open("args.json", 'w') as f:
        json.dump(args_dict, f)
        
    main(args_file="args.json")

if __name__ == "__main__":
    main()

