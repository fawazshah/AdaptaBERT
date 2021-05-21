# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Code adapted from the examples in pytorch-pretrained-bert library"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from io import open
import logging
import numpy as np
import os
import pandas as pd
import re
import random
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from transformers import BertTokenizer, BertForNextSentencePrediction

from common import CDL


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
nsp_cvg_hack = 10


class DataProcessor(object):
    """Processor for the Reddit cross-domain dataset."""

    def get_src_train_examples(self, data_dir):
        src_train_df = pd.read_csv(os.path.join(data_dir, CDL['src']['train_data']), sep='\t')
        return self._create_examples(list(src_train_df[CDL['src']['column']]), CDL['src']['train_data_name'])

    def get_src_test_examples(self, data_dir):
        src_test_df = pd.read_csv(os.path.join(data_dir, CDL['src']['test_data']), sep='\t')
        return self._create_examples(list(src_test_df[CDL['src']['column']]), CDL['src']['test_data_name'])

    def get_trg_train_examples(self, data_dir):
        trg_train_df = pd.read_csv(os.path.join(data_dir, CDL['trg']['train_data']), sep='\t')
        return self._create_examples(list(trg_train_df[CDL['trg']['column']]), CDL['trg']['train_data_name'])

    def get_trg_test_examples(self, data_dir):
        trg_test_df = pd.read_csv(os.path.join(data_dir, CDL['trg']['test_data']), sep='\t')
        return self._create_examples(list(trg_test_df[CDL['trg']['column']]), CDL['trg']['test_data_name'])

    def _create_examples(self, data, set_type):
        """Creates NSP examples for the training and test sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = elem
            # Split text into sentences
            split_regex = re.compile(r'[.|!|?|...]')
            sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
            if len(sentences) < 2:
                continue
            # Choose two sentence-pairs where the second sentence is actually the next sentence
            for i in range(2):
                j = random.randrange(len(sentences) - 1) # don't select last sentence since there is no next sentence from there
                text1 = sentences[j]
                text2 = sentences[j+1]
                for k in range(nsp_cvg_hack): # ??
                    examples.append(InputExample(guid=guid, text1=text1, text2=text2, label=1)) # 1 == IsNext
            # Choose two sentence pairs where the second sentence is instead a random sentence
            for i in range(2):
                j = random.randrange(len(sentences))
                text1 = sentences[j]
                k = random.randrange(len(sentences))
                text2 = sentences[k]
                for l in range(nsp_cvg_hack): # ??
                    examples.append(InputExample(guid=guid, text1=text1, text2=text2, label=0)) # 0 == NotNext
            # No labels needed since this is for unsupervised NSP domain-tuning
        return examples


class BERTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, seq_len):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sample_counter = 0

        processor = DataProcessor()

        trg_domain_examples = processor.get_trg_train_examples(data_dir)
        # use test examples in unsupervised domain tuning
        trg_domain_examples.extend(processor.get_trg_test_examples(data_dir))

        self.examples = trg_domain_examples

        # Include equal amount of src domain data for NSP, or all src data if not enough

        src_domain_examples = processor.get_src_train_examples(data_dir)
        src_domain_examples.extend(processor.get_src_test_examples(data_dir))

        num_trg = len(trg_domain_examples)
        if len(src_domain_examples) > num_trg:
            self.examples.extend(random.sample(src_domain_examples, k=num_trg))
        else:
            self.examples.extend(src_domain_examples)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        self.sample_counter += 1

        # combine to one sample
        cur_example = self.examples[item]

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        # Unpack 1 dimension from input ids, attention masks and token type ids
        cur_tensors = (torch.tensor(cur_features.input_ids)[0],
                       torch.tensor(cur_features.attention_mask)[0],
                       torch.tensor(cur_features.token_type_ids)[0],
                       torch.tensor(cur_features.next_sentence_label))

        return cur_tensors


class InputExample(object):
    """A single training/test example for NSP."""

    def __init__(self, guid, text1, text2, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text1: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text2: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text1 = text1
        self.text2 = text2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, next_sentence_label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.next_sentence_label = next_sentence_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert an NSP sample into a proper training sample with IDs, attention masks, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    encoding_dict = tokenizer(example.text1,
                         example.text2,
                         padding='max_length',
                         truncation=True,
                         max_length=max_seq_length,
                         return_token_type_ids=True,
                         return_attention_mask=True,
                         return_tensors='pt')

    features = InputFeatures(input_ids=encoding_dict['input_ids'],
                             attention_mask=encoding_dict['attention_mask'],
                             token_type_ids=encoding_dict['token_type_ids'],
                             next_sentence_label=example.label)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        print("WARNING: Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_dataset = BERTDataset(args.data_dir, tokenizer, seq_len=args.max_seq_length)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.trained_model_dir:
        if os.path.exists(os.path.join(args.output_dir, WEIGHTS_NAME)):
            previous_state_dict = torch.load(os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            from collections import OrderedDict
            previous_state_dict = OrderedDict()
        distant_state_dict = torch.load(os.path.join(args.trained_model_dir, WEIGHTS_NAME))
        previous_state_dict.update(distant_state_dict) # note that the final layers of previous model and distant model must have different attribute names!
        model = BertForNextSentencePrediction.from_pretrained(args.trained_model_dir, state_dict=previous_state_dict)
    else:
        model = BertForNextSentencePrediction.from_pretrained(args.bert_model)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_masks, token_type_ids, next_sentence_labels = batch
                output = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=next_sentence_labels)
                loss = output['loss']
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
