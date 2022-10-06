from config import tacred_path, tacrev_path, retacred_path

import os
import json
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from transformers import RobertaTokenizer

from data.Data_setting import template_Simple, template_Ent, template_Typ, template_EntTyp
from data.Data_setting import tac_entity_type_dict
from data.Data_setting import tacred_tacrev_relation_type_dict, tacred_tacrev_relation_label_dict
from data.Data_setting import retacred_relation_type_dict, retacred_relation_label_dict
from utils.Tools import convert_token


"""
Original Item:
{
    "id": "61b3a5c8c9a882dcfcd2",
    "docid": "AFP_ENG_20070218.0019.LDC2009T13",
    "relation": "org:founded_by",
    "token": ["Tom", "Thabane", "resigned", "in", "October", "last", "year", "to", "form", "the", "All", "Basotho", "Convention", "-LRB-", "ABC", "-RRB-", ",", "crossing", "the", "floor", "with", "17", "members", "of", "parliament", ",", "causing", "constitutional", "monarch", "King", "Letsie", "III", "to", "dissolve", "parliament", "and", "call", "the", "snap", "election", "."],
    "subj_start": 10,
    "subj_end": 12,
    "obj_start": 0,
    "obj_end": 1,
    "subj_type": "ORGANIZATION",
    "obj_type": "PERSON",
    "stanford_pos": ["NNP", "NNP", "VBD", "IN", "NNP", "JJ", "NN", "TO", "VB", "DT", "DT", "NNP", "NNP", "-LRB-", "NNP", "-RRB-", ",", "VBG", "DT", "NN", "IN", "CD", "NNS", "IN", "NN", ",", "VBG", "JJ", "NN", "NNP", "NNP", "NNP", "TO", "VB", "NN", "CC", "VB", "DT", "NN", "NN", "."],
    "stanford_ner": ["PERSON", "PERSON", "O", "O", "DATE", "DATE", "DATE", "O", "O", "O", "O", "O", "O", "O", "ORGANIZATION", "O", "O", "O", "O", "O", "O", "NUMBER", "O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    "stanford_head": [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3],
    "stanford_deprel": ["compound", "nsubj", "ROOT", "case", "nmod", "amod", "nmod:tmod", "mark", "xcomp", "det", "compound", "compound", "dobj", "punct", "appos", "punct", "punct", "xcomp", "det", "dobj", "case", "nummod", "nmod", "case", "nmod", "punct", "xcomp", "amod", "compound", "compound", "compound", "dobj", "mark", "xcomp", "dobj", "cc", "conj", "det", "compound", "dobj", "punct"]
}
----------
Dataset Item:
{
    'input_ids': [0, 96, 42, 3645, 2156, 5, 9355, 227, 404, 7093, 6157, 139, 9127, 36, 1651, 4839, 8, 1560, 2032, 873, 1728, 36, 621, 4839, 16, 1651, 4790, 30, 12, 12, 12, 3645, 4832, 10431, 1560, 2032, 873, 1728, 10431, 6490, 11, 779, 94, 76, 7, 1026, 5, 1039, 404, 7093, 6157, 139, 9127, 1039, 36, 3943, 4839, 2156, 6724, 5, 1929, 19, 601, 453, 9, 3589, 2156, 3735, 6100, 20303, 1745, 40702, 324, 6395, 7, 30887, 3589, 8, 486, 5, 6788, 729, 479, 2],
    'MLM_input_ids': [0, 96, 42, 3645, 2156, 5, 9355, 227, 404, 7093, 6157, 139, 9127, 36, 1651, 4839, 8, 1560, 2032, 873, 1728, 36, 621, 4839, 16, 50264, 50264, 50264, 50264, 50264, 50264, 3645, 4832, 10431, 1560, 2032, 873, 1728, 10431, 6490, 11, 779, 94, 76, 7, 1026, 5, 1039, 404, 7093, 6157, 139, 9127, 1039, 36, 3943, 4839, 2156, 6724, 5, 1929, 19, 601, 453, 9, 3589, 2156, 3735, 6100, 20303, 1745, 40702, 324, 6395, 7, 30887, 3589, 8, 486, 5, 6788, 729, 479, 2],
    'subj_start': 47,
    'obj_start': 33,
    'MLM_labels': [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1651, 4790, 30, 12, 12, 12, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
    'cls_label': 6
}
"""


file_name_dict = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json"
}


class Dataset_tac(Dataset):
    def __init__(self, args, split):
        self.args = args
        assert split in file_name_dict
        self.split = split

        entity_type_dict = tac_entity_type_dict
        if self.args.dataset_name == "tacred":
            data_path = tacred_path
            relation_type_dict = tacred_tacrev_relation_type_dict
            relation_label_dict = tacred_tacrev_relation_label_dict
        elif self.args.dataset_name == "tacrev":
            data_path = tacrev_path
            relation_type_dict = tacred_tacrev_relation_type_dict
            relation_label_dict = tacred_tacrev_relation_label_dict
        elif self.args.dataset_name == "retacred":
            data_path = retacred_path
            relation_type_dict = retacred_relation_type_dict
            relation_label_dict = retacred_relation_label_dict

        file_name = file_name_dict[self.split]
        self.file_path = os.path.join(data_path, file_name)

        with open(self.file_path, "r", encoding="utf-8") as file:
            self.items = json.load(file)

        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True)
        self.pad_id = self.tokenizer.pad_token_id

        self.entity_type_dict = entity_type_dict
        self.relation_type_dict = self.process_type_dict(relation_type_dict)
        self.relation_label_dict = relation_label_dict

        if self.args.template == "Simple":
            self.relation_start_tokens = template_Simple
        elif self.args.template == "Ent":
            self.relation_start_tokens = template_Ent
        elif self.args.template == "Typ":
            self.relation_start_tokens = template_Typ
        elif self.args.template == "EntTyp":
            self.relation_start_tokens = template_EntTyp

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

    def tokenize(self, tokens, relation, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type):
        tokenized_relation_type = self.relation_type_dict[relation]

        subj_tokens = tokens[subj_start: subj_end + 1]
        obj_tokens = tokens[obj_start: obj_end + 1]
        relation_start_tokens = deepcopy(self.relation_start_tokens)

        if self.args.template == "Ent":
            Ent1_pos = relation_start_tokens.index("Ent1")
            relation_start_tokens[Ent1_pos: Ent1_pos + 1] = subj_tokens
            Ent2_pos = relation_start_tokens.index("Ent2")
            relation_start_tokens[Ent2_pos: Ent2_pos + 1] = obj_tokens
        elif self.args.template == "Typ":
            Typ1_pos = relation_start_tokens.index("Typ1")
            relation_start_tokens[Typ1_pos: Typ1_pos + 1] = subj_type
            Typ2_pos = relation_start_tokens.index("Typ2")
            relation_start_tokens[Typ2_pos: Typ2_pos + 1] = obj_type
        elif self.args.template == "EntTyp":
            Ent1_pos = relation_start_tokens.index("Ent1")
            relation_start_tokens[Ent1_pos: Ent1_pos + 1] = subj_tokens
            Typ1_pos = relation_start_tokens.index("Typ1")
            relation_start_tokens[Typ1_pos: Typ1_pos + 1] = subj_type
            Ent2_pos = relation_start_tokens.index("Ent2")
            relation_start_tokens[Ent2_pos: Ent2_pos + 1] = obj_tokens
            Typ2_pos = relation_start_tokens.index("Typ2")
            relation_start_tokens[Typ2_pos: Typ2_pos + 1] = obj_type

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

            if idx == subj_start:
                new_subj_start = len(tokenized_tokens)
                tokenized_token = ['@'] + tokenized_token
                token_mark = [0] + token_mark
            if idx == subj_end:
                tokenized_token = tokenized_token + ['@']
                token_mark = token_mark + [0]
            if idx == obj_start:
                new_obj_start = len(tokenized_tokens)
                tokenized_token = ['#'] + tokenized_token
                token_mark = [0] + token_mark
            if idx == obj_end:
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

        if new_subj_start >= usable_len:
            new_subj_start = self.args.max_seq_len - 1
        else:
            new_subj_start = new_subj_start + 1

        if new_obj_start >= usable_len:
            new_obj_start = self.args.max_seq_len - 1
        else:
            new_obj_start = new_obj_start + 1

        return input_ids, input_marks, new_subj_start, new_obj_start

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

        relation = item['relation']
        cls_label = self.relation_label_dict[relation]

        tokens = item['token']
        tokens = [convert_token(token) for token in tokens]

        subj_start = item['subj_start']
        subj_end = item['subj_end']

        obj_start = item['obj_start']
        obj_end = item['obj_end']

        subj_type = item['subj_type']
        obj_type = item['obj_type']
        subj_type = self.entity_type_dict[subj_type]
        obj_type = self.entity_type_dict[obj_type]

        input_ids, input_marks, subj_start, obj_start = self.tokenize(tokens, relation, subj_start, subj_end, subj_type, obj_start, obj_end, obj_type)

        MLM_input_ids, MLM_labels = self.mask_relation_type(input_ids, input_marks)

        output = {
            "input_ids": input_ids,
            "MLM_input_ids": MLM_input_ids,
            "subj_start": subj_start,
            "obj_start": obj_start,
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
args = args_test("retacred", "EntTyp", 512)

dataset = Dataset_tac(args, "train")
print(dataset.relation_type_dict)

item = dataset[0]
print(item)

for k, v in item.items():
    print(k)
    print(v)
"""
