#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Make data for Cot-MAE

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import random
import nltk
import argparse
from typing import List
from math import floor
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

print("Downloading NLTK files...")
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--file',
                    required=False,
                    help="Path to txt data file. One line per article format.",
                    default="crosswoz/train.txt")
parser.add_argument('--save_to',
                    required=False,
                    default="pretrain/pretrain.json")
parser.add_argument('--maxlen',
                    default=128,
                    type=int,
                    required=False)
parser.add_argument('--short_sentence_prob',
                    default=0.1,
                    type=float,
                    required=False,
                    help="Keep some short length sentences, for better performance")
parser.add_argument('--tokenizer',
                    default="hfl/chinese-bert-wwm-ext",
                    required=False)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
rng = random.Random()

def _base_encode_one_span(line: str, maxlen=args.maxlen) -> List[List[List[int]]]:
    # 'spans' of this article:
    #           [   [sentence1: List[int], sentence2, sentence3 ...] → span1,
    #               [sentence1, sentence2, sentence3 ...] → span2
    #           ]
    # 
    # sentences = nltk.sent_tokenize(line.strip())
    sentence=line.strip().split("\t")[1]
    sentence=sentence.split("|")[0]
    sentences = sentence.split("$")
    tokenized = [
        tokenizer(
            s,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for s in sentences
    ]

    all_spans = []
    tokenized_span = []

    if rng.random() <= args.short_sentence_prob:
        target_seq_len = rng.randint(2, maxlen)
    else:
        target_seq_len = maxlen
    
    curr_len = 0

    for sent in tokenized:
        if len(sent) == 0:
            continue
        tokenized_span.append(sent)
        curr_len += len(sent)
        if curr_len > target_seq_len:
            all_spans.append(tokenized_span)
            curr_len = 0
            tokenized_span = []
            if rng.random() <= args.short_sentence_prob:
                target_seq_len = rng.randint(2, maxlen)
            else:
                target_seq_len = maxlen
            
    if len(tokenized_span) > 0:
        all_spans.append(tokenized_span)

    if len(all_spans) < 2:
        return None
    
    return all_spans

def encode_one(line):
    # { 'spans':
    #           [   [token1:int, token2, token3 ...] → span1,
    #               [token1, token2, token3 ...] → span2
    #           ]
    # }
    spans = _base_encode_one_span(line)
    if spans is None:
        return None
    return json.dumps({'spans': [sum(i, []) for i in spans]})

def encode_three_corpus_aware_type(line, maxlen=args.maxlen):
    # Article:
    # { 'spans':
    #           [   {'anchor': span1, 
    #                'random_sampled': random_sampled_span,     # Random sampled spans
    #                'nearby': nearby_span,     # ICT like, if short sentence, it will 50% prob keep this 
    #                                           # short sentence in the nearby_span
    #                'overlap': overlaped_span  # partial overlap with `anchor`, Contriver use this method
    #               },
    #               ……
    #           ]
    # }
    spans = _base_encode_one_span(line)
    if spans is None:
        return None
    
    # Concat anchors
    anchors = [sum(i, []) for i in spans]

    final_spans = [{'text': anch} for anch in anchors]
        
    return final_spans

with open(args.save_to, 'w') as f:
    # Multiprocess is highly recommended
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            encode_three_corpus_aware_type,
            tqdm(open(args.file), total=wc_count(args.file)),
            chunksize=1000,
        )
        for blocks in all_tokenized:
            if blocks is None:
                continue
            for block in blocks:
                f.write(json.dumps(block) + '\n')
    
    # For debug
    # for line in tqdm(open(args.file), total=wc_count(args.file)):
    #     encoded_line = encode_three_corpus_aware_type(line)
