from config import semeval_path

import os
import json
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from transformers import RobertaTokenizer

from data.Data_setting import template_Simple, template_Ent
from data.Data_setting import semeval_relation_type_dict, semeval_relation_label_dict


"""
Original Item:
{
    "token": ["the", "original", "play", "was", "filled", "with", "very", "topical", "humor", ",", "so", "the", "director", "felt", "free", "to", "add", "current", "topical", "humor", "to", "the", "script", "."],
    "h": {
        "name": "play",
        "pos": [2, 3]
        },
    "t": {
        "name": "humor",
        "pos": [8, 9]
        },
    "relation": "Component-Whole(e2,e1)"
}
----------
Dataset Item:
{
    'input_ids': [0, 96, 42, 3645, 2156, 5, 9355, 227, 310, 8, 12073, 16, 1086, 7681, 3645, 4832, 5, 1461, 1039, 310, 1039, 21, 3820, 19, 182, 33469, 10431, 12073, 10431, 2156, 98, 5, 736, 1299, 481, 7, 1606, 595, 33469, 12073, 7, 5, 8543, 479, 2],
    'MLM_input_ids': [0, 96, 42, 3645, 2156, 5, 9355, 227, 310, 8, 12073, 16, 50264, 50264, 3645, 4832, 5, 1461, 1039, 310, 1039, 21, 3820, 19, 182, 33469, 10431, 12073, 10431, 2156, 98, 5, 736, 1299, 481, 7, 1606, 595, 33469, 12073, 7, 5, 8543, 479, 2],
    'subj_start': 18,
    'obj_start': 26,
    'MLM_labels': [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1086, 7681, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
    'cls_label': 1
}
"""


file_name_dict = {
    "train": "semeval_train.txt",
    "dev": "semeval_val.txt",
    "test": "semeval_test.txt"
}


class Dataset_semeval(Dataset):
    def __init__(self, args, split):
        self.args = args
        assert split in file_name_dict
        self.split = split

        file_name = file_name_dict[self.split]
        self.file_path = os.path.join(semeval_path, file_name)

        self.items = []
        with open(self.file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                item = json.loads(line)
                self.items.append(item)

        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True)
        self.pad_id = self.tokenizer.pad_token_id

        self.relation_type_dict = self.process_type_dict(semeval_relation_type_dict)
        self.relation_label_dict = semeval_relation_label_dict

        if self.args.template == "Simple":
            self.relation_start_tokens = template_Simple
        elif self.args.template == "Ent":
            self.relation_start_tokens = template_Ent

        self.sentence_start_tokens = ["sentence", ":"]

    def __len__(self):
        return len(self.items)

    @property
    def label_num(self):
        return len(self.relation_label_dict)

    def process_type_dict(self, type_dict):
        type_dict_cp = deepcopy(type_dict)

        max_type_len = 0
        for key, value in type_dict_cp.items():
            tokenized_type = []
            for token in value:
                tokenized_type.extend(self.tokenizer.tokenize(token))

            if max_type_len < len(tokenized_type):
                max_type_len = len(tokenized_type)

        for key, value in type_dict_cp.items():
            tokenized_type = []
            for token in value:
                tokenized_type.extend(self.tokenizer.tokenize(token))
            type_len = len(tokenized_type)
            pad_token_num = max_type_len - type_len

            new_value = tokenized_type + ["-"] * pad_token_num
            type_dict_cp[key] = new_value

        return type_dict_cp

    def tokenize(self, tokens, relation, head_start, head_end, tail_start, tail_end):
        tokenized_relation_type = self.relation_type_dict[relation]

        head_tokens = tokens[head_start: head_end + 1]
        tail_tokens = tokens[tail_start: tail_end + 1]
        relation_start_tokens = deepcopy(self.relation_start_tokens)

        if self.args.template == "Ent":
            Ent1_pos = relation_start_tokens.index("Ent1")
            relation_start_tokens[Ent1_pos: Ent1_pos + 1] = head_tokens
            Ent2_pos = relation_start_tokens.index("Ent2")
            relation_start_tokens[Ent2_pos: Ent2_pos + 1] = tail_tokens

        tokenized_relation_start = []
        for token in relation_start_tokens:
            tokenized_relation_start.extend(self.tokenizer.tokenize(token))

        tokenized_sentence_start = []
        for token in self.sentence_start_tokens:
            tokenized_sentence_start.extend(self.tokenizer.tokenize(token))

        tokenized_tokens = []
        tokenized_tokens.extend(tokenized_relation_start)
        tokenized_tokens.extend(tokenized_relation_type)
        tokenized_tokens.extend(tokenized_sentence_start)

        input_marks = []
        input_marks.extend([0] * len(tokenized_relation_start))
        input_marks.extend([1] * len(tokenized_relation_type))
        input_marks.extend([0] * len(tokenized_sentence_start))

        for idx, token in enumerate(tokens):
            tokenized_token = self.tokenizer.tokenize(token)
            token_mark = [0] * len(tokenized_token)

            if idx == head_start:
                new_head_start = len(tokenized_tokens)
                tokenized_token = ['@'] + tokenized_token
                token_mark = [0] + token_mark
            if idx == head_end:
                tokenized_token = tokenized_token + ['@']
                token_mark = token_mark + [0]
            if idx == tail_start:
                new_tail_start = len(tokenized_tokens)
                tokenized_token = ['#'] + tokenized_token
                token_mark = [0] + token_mark
            if idx == tail_end:
                tokenized_token = tokenized_token + ['#']
                token_mark = token_mark + [0]

            tokenized_tokens.extend(tokenized_token)
            input_marks.extend(token_mark)

        usable_len = self.args.max_seq_len - 2
        tokenized_tokens = tokenized_tokens[:usable_len]
        input_marks = input_marks[:usable_len]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        input_marks = [0] + input_marks + [0]

        if new_head_start >= usable_len:
            new_head_start = self.args.max_seq_len - 1
        else:
            new_head_start = new_head_start + 1

        if new_tail_start >= usable_len:
            new_tail_start = self.args.max_seq_len - 1
        else:
            new_tail_start = new_tail_start + 1

        return input_ids, input_marks, new_head_start, new_tail_start

    def mask_relation_type(self, input_ids, input_marks):
        MLM_input_ids = deepcopy(input_ids)

        MLM_labels = [-100] * len(MLM_input_ids)

        for idx, mark in enumerate(input_marks):
            if mark == 1:
                MLM_labels[idx] = MLM_input_ids[idx]
                MLM_input_ids[idx] = self.tokenizer.mask_token_id

        return MLM_input_ids, MLM_labels

    def __getitem__(self, index):
        item = deepcopy(self.items[index])

        tokens = item['token']

        head_dict = item['h']
        head_start, head_end = head_dict['pos']

        tail_dict = item['t']
        tail_start, tail_end = tail_dict['pos']

        head_end = head_end - 1
        tail_end = tail_end - 1

        relation = item['relation']
        cls_label = self.relation_label_dict[relation]

        input_ids, input_marks, head_start, tail_start = self.tokenize(tokens, relation, head_start, head_end, tail_start, tail_end)

        MLM_input_ids, MLM_labels = self.mask_relation_type(input_ids, input_marks)

        output = {
            "input_ids": input_ids,
            "MLM_input_ids": MLM_input_ids,
            "subj_start": head_start,
            "obj_start": tail_start,
            "MLM_labels": MLM_labels,
            "cls_label": cls_label,
        }

        return output

    def collate_fn(self, feature_batch):
        max_len = max([len(feature["input_ids"]) for feature in feature_batch])

        input_ids = [feature["input_ids"] + [self.pad_id] * (max_len - len(feature["input_ids"])) for feature in feature_batch]
        attention_mask = [[1] * len(feature["input_ids"]) + [0] * (max_len - len(feature["input_ids"])) for feature in feature_batch]
        subj_start = [feature["subj_start"] for feature in feature_batch]
        obj_start = [feature["obj_start"] for feature in feature_batch]
        cls_label = [feature["cls_label"] for feature in feature_batch]

        MLM_input_ids = [feature["MLM_input_ids"] + [self.pad_id] * (max_len - len(feature["MLM_input_ids"])) for feature in feature_batch]
        MLM_labels = [feature["MLM_labels"] + [-100] * (max_len - len(feature["MLM_labels"])) for feature in feature_batch]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.FloatTensor(attention_mask)
        subj_start = torch.LongTensor(subj_start)
        obj_start = torch.LongTensor(obj_start)
        cls_label = torch.LongTensor(cls_label)

        MLM_input_ids = torch.LongTensor(MLM_input_ids)
        MLM_labels = torch.LongTensor(MLM_labels)

        outputs = {
            "input_ids": input_ids,
            "MLM_input_ids": MLM_input_ids,
            "attention_mask": attention_mask,
            "subj_start": subj_start,
            "obj_start": obj_start,
            "MLM_labels": MLM_labels,
            "cls_label": cls_label,
        }

        return outputs


"""
from config import roberta_large_path
class args_test():
    def __init__(self, dataset_name, template, max_seq_len):
        self.model_name_or_path = roberta_large_path
        self.dataset_name = dataset_name
        self.template = template
        self.max_seq_len = max_seq_len
args = args_test("", "Ent", 512)

dataset = Dataset_semeval(args, "train")
print(dataset.relation_type_dict)

item = dataset[0]
print(item)

for k, v in item.items():
    print(k)
    print(v)
"""
