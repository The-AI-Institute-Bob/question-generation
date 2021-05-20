import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from torch import nn
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    set_seed,
)
from transformers import get_scheduler
from .data_collator import T2TDataCollator


def freeze_embeds(model: nn.Module):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)

def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={"help": "label smoothing rate, set to > 0 if you want to enable lable smoothing"}
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"}, 
    )
    model_type: Optional[str] = field(
        default='QG',
        metadata={"help": "QG for genration without answer, QAG if answer is provided"}, 
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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    torch.cuda.empty_cache()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.prediction_loss_only = True
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_embeds:
        logger.info("freezing embeddings of the model")
        freeze_embeds(model)
        assert_not_all_frozen(model)

    # Get datasets
    logger.info('loading dataset')
    
    train_dataset = torch.load(data_args.train_file_path) if training_args.do_train else None
    valid_dataset = torch.load(data_args.valid_file_path) if training_args.do_eval else None
    
    logger.info('finished loading dataset')

    # Initialize data_loader
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None
    )

    train_data_loader = DataLoader(
    train_dataset,
    batch_size=training_args.per_device_train_batch_size,
    collate_fn=data_collator,              
    )
    
    eval_data_loader = DataLoader(
    valid_dataset,
    batch_size=training_args.per_device_eval_batch_size,
    collate_fn=data_collator,              
    )




    # Training loop
    optimizer = AdamW(model.parameters(), lr=0.0001) #To change
    num_training_steps = training_args.num_train_epochs * len(train_data_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
        )

    progress_bar = tqdm(range(num_training_steps))
    model.to('cuda')
    model.train()
    
    for epoch in range(training_args.num_train_epochs):
        for (i,batch) in enumerate(train_data_loader):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            if model_args.label_smoothing == 0:
                outputs = model(**batch)
                loss = outputs[0]
            else:
                labels = batch.pop("labels")
                labels[labels == -100] = model.config.pad_token_id
                outputs = model(**batch)
                lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, labels, model_args.label_smoothing, ignore_index=model.config.pad_token_id
                    )

            loss.backward()
            if (i+1)%training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
    
    #Saving the models
    model.save_pretrained(training_args.output_dir)       
    tokenizer.save_pretrained(training_args.output_dir)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def run_qg(args_dict):
    with open("args.json", 'w') as f:
        json.dump(args_dict, f)
    
    main(args_file="args.json")

if __name__ == "__main__":
    main()