import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

User = str
Item = str
InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor
ItemIdxs = torch.Tensor


def get_positive2negatives(num_items: int, num_samples: int = 100) -> Dict[int, List[int]]:
    all_samples = np.arange(1, num_items + 1)
    positive2negatives = {}
    pbar = tqdm(iterable=all_samples,desc="Creating positive2negatives",total=all_samples.shape[0])
    for positive_sample in pbar:
        candidates = np.concatenate(
            (np.arange(positive_sample), np.arange(positive_sample + 1, num_items + 1)),
            axis=0,
        )
        negative_samples = np.random.choice(
            candidates, size=(num_samples,), replace=False
        )
        positive2negatives[positive_sample] = negative_samples.tolist()

    return positive2negatives


def get_negative_samples(
    positive2negatives: Dict[int, List[int]],
    positive_seqs: torch.Tensor,
    num_samples=1,
) -> torch.Tensor:
    negative_seqs = torch.zeros(size=positive_seqs.shape, dtype=torch.long)
    for row_idx in range(positive_seqs.shape[0]):
        for col_idx in range(positive_seqs[row_idx].shape[0]):
            positive_sample = positive_seqs[row_idx][col_idx].item()

            if positive_sample == 0:
                continue
            negative_samples = positive2negatives[positive_sample]
            negative_sample = np.random.choice(
                a=negative_samples, size=(num_samples,), replace=False
            )
            negative_seqs[row_idx][col_idx] = negative_sample[0]

    return negative_seqs


def pad_or_truncate_seq(
    sequence,
    max_seq_len: int,
) -> torch.Tensor:
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    if len(sequence) > max_seq_len:
        sequence = sequence[-max_seq_len:]
    else:
        diff = max_seq_len - len(sequence)
        sequence = F.pad(sequence, pad=(diff, 0))
    return sequence


class Dataset:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        data_filepath: str,
        debug: bool,
    ):
        self.debug = debug
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data_filepath = data_filepath
        self.data = self.load_data(data_filepath=self.data_filepath)

        self.user2items, self.item2users = self.create_mappings(data=self.data)
        self.num_users = len(self.user2items)
        self.num_items = len(self.item2users) if not self.debug else 57289

        self.positive2negatives = get_positive2negatives(self.num_items)
        splits = self.create_train_valid_test(user2items=self.user2items)
        self.user2items_train, self.user2items_valid, self.user2items_test = splits

    def load_data(self, data_filepath: str) -> List[List[int]]:
        with open(file=data_filepath) as f:
            user_item_pairs = f.readlines()
        user_item_pairs = [pair.strip().split() for pair in user_item_pairs]
        user_item_pairs = [list(map(int, pair)) for pair in user_item_pairs]

        return user_item_pairs

    def create_mappings(
        self, data: List[List[int]]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        user2items = {}
        item2users = {}
        pbar = tqdm(
            iterable=data,
            desc="Creating user2items",
            total=len(data),
        )
        for user, item in pbar:
            try:
                user2items[user].append(item)
            except KeyError:
                user2items[user] = [item]

            try:
                item2users[item].append(user)
            except KeyError:
                item2users[item] = [user]

        return user2items, item2users

    def create_train_valid_test(
        self, user2items: Dict[int, List[int]]
    ) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[List[int], int]], Dict[int, Tuple[List[int], int]]]:
        user2items_train = {}
        user2items_valid = {}
        user2items_test = {}

        pbar = tqdm(
            iterable=user2items.items(),
            desc="Getting train/valid/test splits",
            total=len(user2items),
        )
        for user, items in pbar:
            num_items = len(items)
            if num_items < 3:
                user2items_train[user] = items
                user2items_valid[user] = []
                user2items_test[user] = []
            else:
                user2items_train[user] = items[:-2]

                valid_input_seq = items[:-2]
                valid_label = items[-2]
                user2items_valid[user] = (valid_input_seq, valid_label)

                test_input_seq = valid_input_seq + [valid_label]
                test_label = items[-1]
                user2items_test[user] = (test_input_seq, test_label)

        return user2items_train, user2items_valid, user2items_test

    def collate_fn_train(self, batch: List[List[int]]) -> InputSequences:
        seq_tensors = []
        for seq in batch:
            seq = pad_or_truncate_seq(seq, max_seq_len=self.max_seq_len)
            seq_tensors.append(seq)

        input_seqs = torch.stack(seq_tensors)

        return input_seqs

    def collate_fn_eval(self, batch: List[Tuple[List[int], int]]) -> Tuple[InputSequences, ItemIdxs]:
        input_seqs = [x[0] for x in batch]
        seq_tensors = []
        for seq in input_seqs:
            seq = pad_or_truncate_seq(seq, max_seq_len=self.max_seq_len)
            seq_tensors.append(seq)

        input_seqs = torch.stack(seq_tensors)

        item_idxs = [x[1] for x in batch]
        item_idxs = torch.tensor(item_idxs, dtype=torch.long)

        return (input_seqs, item_idxs)

    def get_dataloader(
        self,
        data: Dict[int, List[int]],
        split: str = "train",
    ) -> DataLoader:
        dataset = list(data.values())
        if split in ["valid", "test"]:
            shuffle = False
            collate_fn = self.collate_fn_eval

            input_seqs = [x[0] for x in dataset if x != []]
            all_pred_item_idxs = []

            if split == "valid":
                for items in self.user2items_valid.values():
                    if items == []:
                        continue

                    positive_sample = items[1]
                    negative_samples = self.positive2negatives[positive_sample]
                    pred_item_idxs = [positive_sample] + negative_samples
                    all_pred_item_idxs.append(pred_item_idxs)
            elif split == "test":
                for items in self.user2items_test.values():
                    if items == []:
                        continue

                    positive_sample = items[1]
                    negative_samples = self.positive2negatives[positive_sample]
                    pred_item_idxs = [positive_sample] + negative_samples
                    all_pred_item_idxs.append(pred_item_idxs)

            assert len(input_seqs) == len(all_pred_item_idxs)
            dataset = list(zip(input_seqs, all_pred_item_idxs))
        else:
            shuffle = True
            collate_fn = self.collate_fn_train

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        return dataloader