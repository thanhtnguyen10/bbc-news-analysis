# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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
"""Custom dataset loading script for Library Carpentries"""

import os

import datasets

logger = datasets.logging.get_logger(__name__)

##Carpentries Note- This is where I got this dataset.
_CITATION = """\
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "T-NER: An All-Round Python Library for Transformer-based Named Entity Recognition",
    author = "Asahi Ushio and Jose Camacho-Collados",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    year = "2021",
    url = "https://aclanthology.org/2021.eacl-demos.7",
    pages = "53--62",
}
"""

_DESCRIPTION = """\
A set of labels for BBC news title. For library carpentries.
"""

##Carpentries Note- We're changing the file names to match our dataset. The path lead to the zip file store all dataset file.
_URL = "bbc_news_label_bert.zip"
_TRAINING_FILE = "bbc_train.txt"
_DEV_FILE = "bbc_valid.txt"
_TEST_FILE = "bbc_test.txt"

##Carpentries Note- Find and replace to match the name of our python file.
class bbc_news_for_modelConfig(datasets.BuilderConfig):
    """BuilderConfig for bbc_news_for_model"""

    def __init__(self, **kwargs):
        """BuilderConfig bbc_news_for_model.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(bbc_news_for_modelConfig, self).__init__(**kwargs)


class bbc_news_for_model(datasets.GeneratorBasedBuilder):
    """bbc_news_for_model dataset."""

    BUILDER_CONFIGS = [
        ##Carpentries Note- Using bbc_news_for_modelConfig class that write the config function
        bbc_news_for_modelConfig(name="bbc_news_for_model", version=datasets.Version("1.0.0"), description="BBC news title dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                    ##Carpentries Note- Change these labels based on our project.
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-ACTIVITY",
                                "B-OBJECTIVITY"                     
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        data_files = {
            "train": os.path.join(downloaded_file, _TRAINING_FILE),
            "dev": os.path.join(downloaded_file, _DEV_FILE),
            "test": os.path.join(downloaded_file, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    ##Carpentries Note- Our BBC label file has ' -X- _ ' delimitation
                    splits = line.split(" -X- _ ")
                    ##Carpentries Note- We change the tokens and ner_tags to match with the position in BBC label file
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    ##Carpentries- Notice we took out the other tags.
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
