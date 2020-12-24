# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3


from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pickle
import nltk
nltk.download('punkt')

import nlp





QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


class PrepareConfig(nlp.BuilderConfig):
    """BuilderConfig for data."""

    def __init__(self, qg_format="highlight", **kwargs):
        """BuilderConfig for data.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(PrepareConfig, self).__init__(**kwargs)
        self.qg_format = qg_format


class Prepare(nlp.GeneratorBasedBuilder):

    _URL = os.getcwd()
    _DEV_FILE = pickle.load(open("valid.p",'rb'))
    _TRAINING_FILE = pickle.load(open('train.p','rb'))

    BUILDER_CONFIGS = [
        PrepareConfig(
            name=f"{format_}_qg_format",
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text",
            qg_format=format_
        )
        for format_ in QG_FORMATS
    ]

    def _info(self):
        return nlp.DatasetInfo(
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL, f"{self._TRAINING_FILE}.json"),
            "dev": os.path.join(self._URL, f"{self._DEV_FILE}.json"),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
    
    

    def process_e2e_qg(self, paragraph):
        source_text = f"generate questions: {paragraph['context'].strip()}"
        questions = paragraph['questions']
        target_text = " {sep_token} ".join(questions)
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}



    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        ida = 0
        with open(filepath) as f:
            data = json.load(f)
            for k in range(len(data['context'])):
                paragraph = {'context' : data['context'][k], 'questions': data['questions'][k]}
                context = paragraph['context'].strip()
                yield count, self.process_e2e_qg(paragraph)
                count += 1
                    
                for q in paragraph['questions']:
                    question = q.strip()
                    id_ = str(ida)
                    ida += 1

