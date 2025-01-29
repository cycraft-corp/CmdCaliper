import collections
import os
from typing import Dict

import torch
import torch.distributed as dist

from .utils import load_json

class ContrastDataset:
    """
    Data format:
    ```
    [
        [sentence_1, similar_sentence_1],
        [sentence_2, similar_sentence_2],
        [sentence_3, similar_sentence_3],
    ]
    ```
    or 
    ```
    [
        [sentence_1, similar_sentence_1, hard_negative_sentence_1],
        [sentence_2, similar_sentence_2, hard_negative_sentence_2],
        [sentence_3, similar_sentence_3, hard_negative_sentence_3],
    ]
    ```
    """
    def __init__(self, raw_data, tokenizer, device, tokenize_on_the_fly=False):
        self.tokenizer = tokenizer

        self.raw_data_length = len(raw_data)
        self.device = device

        self.has_negative_sample = len(raw_data[0]) == 3 if len(raw_data) > 0 else False
        self.tokenize_on_the_fly = tokenize_on_the_fly

        self.processed_data, self.total_sentences_map = self.preprocess(raw_data)

    @classmethod
    def initialize_dataset(cls, tokenizer, data_args, device="cuda"):
        train_dataset = None
        eval_dataset = None

        if data_args.train_percentage == 1:
            train_dataset = cls(
                load_json(os.path.join(data_args.path_to_train_data_dir, "data.json")),
                tokenizer, device, data_args.tokenize_on_the_fly
            )
            if data_args.path_to_eval_data_dir is not None:
                eval_dataset = cls(
                    load_json(os.path.join(data_args.path_to_eval_data_dir, "data.json")),
                    tokenizer, device, data_args.tokenize_on_the_fly
                )
        else:
            data = load_json(os.path.join(data_args.path_to_train_data_dir, "data.json"))
            
            perm = torch.randperm(len(data)).tolist()
            split = int(len(perm) * data_args.train_percentage)
            train_indices = perm[:split]
            eval_indices = perm[split:]
            
            train_data = [data[i] for i in train_indices]
            eval_data = [data[i] for i in eval_indices]
            
            train_dataset = cls(train_data, tokenizer, device, data_args.tokenize_on_the_fly)
            eval_dataset = cls(eval_data, tokenizer, device, data_args.tokenize_on_the_fly)
            
        return train_dataset, eval_dataset

    def preprocess(self, raw_data):
        total_sentences_map = collections.defaultdict(list)

        for d in raw_data:
            total_sentences_map["query_sentence_list"].append(d[0])
            total_sentences_map["positive_sentence_list"].append(d[1])
            if self.has_negative_sample:
                total_sentences_map["negative_sentence_list"].append(d[2])

        total_tokens_map = {}
        if not self.tokenize_on_the_fly:
            for k in total_sentences_map:
                k_tokens = self.tokenizer(
                    total_sentences_map[k], padding="max_length",
                    truncation=True, return_tensors="pt"
                )

                sentence_num = len(total_sentences_map[k])
                total_tokens_map[k] = k_tokens
        return total_tokens_map, total_sentences_map

    def __len__(self):
        return self.raw_data_length

    def __getitem__(self, idx):
        if self.tokenize_on_the_fly:
            return {k: self.total_sentences_map[k][idx] for k in self.total_sentences_map}
        return [{
            "input_ids": self.processed_data[k]["input_ids"][idx],
            "attention_mask": self.processed_data[k]["attention_mask"][idx]
        } for k in self.processed_data]

    def collate_fn(self, batch_pair_data):
        """
        Returns:
        {
            "input_ids": torch.tensor([
                [], query_sample
                [], positive_sample
                [], negative_sample if exist
                [], query_sample
                [], positive_sample
                [], negative_sample if exist
            ]),
            "attention_mask": torch.tensor([
                [], query_sample
                [], positive_sample
                [], negative_sample if exist
                [], query_sample
                [], positive_sample
                [], negative_sample if exist
            ]),
        }
        """
        if self.tokenize_on_the_fly:
            flatten_sentence_list = []
            for data in batch_pair_data:
                for k in data:
                    flatten_sentence_list.append(data[k])
            merged_batch_tokens = self.tokenizer(
                flatten_sentence_list, padding=True, max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )

            merged_batch_tokens = {
                "input_ids": merged_batch_tokens["input_ids"],
                "attention_mask": merged_batch_tokens["attention_mask"]
            }
        else:
            flatten_batch_pair_data = []
            for pd in batch_pair_data:
                flatten_batch_pair_data.extend(pd)
            merged_batch_tokens = dict(
                input_ids=torch.stack([d["input_ids"] for d in flatten_batch_pair_data], 0),
                attention_mask=torch.stack([d["attention_mask"] for d in flatten_batch_pair_data], 0),
            )

            merged_batch_tokens = self.truncate_redundant_tokens(merged_batch_tokens)
            return merged_batch_tokens, {"has_negative_sample": self.has_negative_sample}

    def truncate_redundant_tokens(self, batch_tokens: Dict[str, torch.tensor]):
        if dist.is_initialized() and dist.get_world_size() > 1:
            # If we use tensor parallelism, we must ensure the sequence lengths are the same for each process.
            # Therefore, we all reduce here to get the max value between all processes.
            max_non_zero_index = torch.max(torch.sum(batch_tokens["attention_mask"], 1)).to(self.device)
            dist.all_reduce(
            max_non_zero_index, 
            op=torch.distributed.ReduceOp.MAX
            )
            max_non_zero_index = max_non_zero_index.cpu()
        else:
            max_non_zero_index = torch.max(torch.sum(batch_tokens["attention_mask"], 1))

        # To compatible with flash attention
        max_non_zero_index += 4 - max_non_zero_index % 4

        for k, v in batch_tokens.items():
            v = v[:, :max_non_zero_index]
            batch_tokens[k] = v.to(self.device)
        return batch_tokens

