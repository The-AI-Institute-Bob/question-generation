import itertools
import logging
from typing import Optional, Dict, Union

from nltk import sent_tokenize

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    T5Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class QGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool,
        model_type: str
    ) :

        if model_type not in ['QAG','QG']:
            raise Exception("Expected model type to be one of 'QAG', 'QG'")
            
        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        

        self.model_type = model_type

        
        self.default_generate_kwargs = {
            "max_length": 32,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
        }
    
    def __call__(self, context: str, answer: Optional[str] = None, **generate_kwargs):
        '''Main pipeline function, will take as an input a context and an answer within the context (None by default) and will return the generated question'''
        if self.model_type == "QG":
            inputs = self._prepare_inputs_for_qg(context)
        else:
            inputs = self._prepare_inputs_for_qag(context,answer)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        if "<sep>" in prediction: #In the case of end to end QG, there often are multiple questions predicted
            questions = prediction.split("<sep>")
            questions = [question.strip() for question in questions[:-1]]
        else:
            questions = prediction
        return questions
    
    def _prepare_inputs_for_qg(self, context):
        '''Prepare and tokenize the input for question generation without answer'''
        source_text = f"generate questions: {context}"

        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _prepare_inputs_for_qag(self, context, answer):
        '''Prepare and tokenize the input for question generation with answer '''
        start_pos  = context.index(answer)
        end_pos = start_pos + len(answer)
        source_text = f"answer: {answer} context: {context[:start_pos]} <hl> {answer} <hl> {context[end_pos:]}"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
                                  
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs



def pipeline(
    model: Optional = None,
    model_type: Optional[str] = "QG",
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = 'valhalla/t5-small-e2e-qg'
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = T5Tokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = T5Tokenizer.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    

        return QGPipeline(model=model, tokenizer=tokenizer, use_cuda=use_cuda, model_type = model_type)
