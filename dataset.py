import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Optimizer

User = str
Item = str
InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor
ItemIdxs = torch.Tensor

def get_positive2negatives(num_items: int, num_samples: int = 100) -> list[int]:
    """
    Purpose: Creates a dictionary that maps each positive item to a list of negative 
      samples, used for generating negative samples during training.
    Preconditions: The function expects the total number of items and the number of 
      negative samples to generate per positive item.
    Returns: A dictionary mapping each positive item to an array of negative integers.
    Parameters:
        num_items (int): Total number of distinct items.
        num_samples (int): Number of negative samples per positive item.
    """
    # Create a list of all possible item indices.
    all_samples = np.arange(1, num_items + 1)
    positive2negatives = {}
    # Iterate over each positive sample.
    pbar = tqdm(iterable=all_samples,desc="Creating positive2negatives",total=all_samples.shape[0])
    for positive_sample in pbar:
         # Exclude the current item to generate candidates for negative sampling.
        candidates = np.concatenate(
            (np.arange(positive_sample), np.arange(positive_sample + 1, num_items + 1)),
            axis=0,
        )
        # Randomly select negative samples from the candidates.
        negative_samples = np.random.choice(
            candidates, size=(num_samples,), replace=False
        )
        # Map the positive item to its corresponding negative samples.
        positive2negatives[positive_sample] = negative_samples.tolist()

    return positive2negatives

def get_negative_samples(
    positive2negatives: dict[int, list[int]],
    positive_seqs: torch.Tensor,
    num_samples=1,
) -> torch.Tensor:
    """
    Purpose: Generates negative samples corresponding to positive samples for training.
    Preconditions: Requires a dictionary of positive-to-negative mappings and the input 
      sequence of positive samples.
    Returns: A tensor containing negative samples.
    Parameters:
        positive2negatives: Dictionary mapping positive items to lists of negative items.
        positive_seqs: Tensor of positive sequences.
    """
    # Initialize a tensor to store the negative samples.
    negative_seqs = torch.zeros(size=positive_seqs.shape, dtype=torch.long)
    # Iterate through the positive sequences.
    for row_idx in range(positive_seqs.shape[0]):
        for col_idx in range(positive_seqs[row_idx].shape[0]):
            positive_sample = positive_seqs[row_idx][col_idx].item()

            if positive_sample == 0:
                continue
            # Choose random negative samples for the current positive sample.
            negative_samples = positive2negatives[positive_sample]
            negative_sample = np.random.choice(
                a=negative_samples, size=(num_samples,), replace=False
            )
            negative_seqs[row_idx][col_idx] = negative_sample[0]

    return negative_seqs

def pad_or_truncate_seq(
    sequence: list[int],
    max_seq_len: int,
) -> InputSequences:
    """
    Purpose: Pads or truncates input sequences to a fixed length depending on max_seq_len.
    Preconditions: Input sequences should be in list form, and a maximum length should be specified.
    Returns: A tensor of input sequences padded or truncated to the specified length.
    Parameters:
        sequence: The sequence to be padded or truncated.
        max_seq_len: The maximum length of the sequence.
    """
    # Convert the input sequence to a tensor.
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    # Truncate or pad the sequence to match max_seq_len.
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
        # Initialize and load data.
        self.debug = debug
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data_filepath = data_filepath
        self.data = self.load_data(data_filepath=self.data_filepath)

         # Create mappings and perform debug checks.
        self.user2items, self.item2users = self.create_mappings(data=self.data)
        self.num_users = len(self.user2items)
        self.num_items = len(self.item2users) if not self.debug else 57289

        # Generate negative samples and train/valid/test splits.
        self.positive2negatives = get_positive2negatives(self.num_items)
        splits = self.create_train_valid_test(user2items=self.user2items)
        self.user2items_train, self.user2items_valid, self.user2items_test = splits

    def load_data(self, data_filepath: str) -> list[list[User, Item]]:
        """Purpose: Load and format data."""
        with open(file=data_filepath) as f:
            user_item_pairs = f.readlines()
        # Parse and format the data.
        user_item_pairs = [pair.strip().split() for pair in user_item_pairs]
        user_item_pairs = [list(map(int, pair)) for pair in user_item_pairs]

        return user_item_pairs

    def create_mappings(
        self, data: list[list[User, Item]]
    ) -> (dict[User, list[Item]], dict[Item, list[User]],):
        """
        Purpose: Convert the list of [user, item] pairs to a mapping where the users 
          are keys mapped to a list of items.
        """
        user2items = {}
        item2users = {}
        pbar = tqdm(
            iterable=data,
            desc="Creating user2items",
            total=len(data),
        )
        # Iterate over each user-item pair to populate mappings.
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
        self, user2items: dict[User, list[Item]]
    ) -> (dict[User, list[Item]], dict[User, list[Item]], dict[User, list[Item]],):
        """
        Purpose: Makes train/valid/test splits for users and items.
        """
        user2items_train = {}
        user2items_valid = {}
        user2items_test = {}

        pbar = tqdm(
            iterable=user2items.items(),
            desc="Getting train/valid/test splits",
            total=len(user2items),
        )
        # Iterate over each user and their item sequences.
        for user, items in pbar:
            num_items = len(items)
            # Use only training set if user has less than 3 interactions.
            if num_items < 3:
                user2items_train[user] = items
                user2items_valid[user] = []
                user2items_test[user] = []
            # Prepare training, validation, and test sets.
            else:
                user2items_train[user] = items[:-2]

                valid_input_seq = items[:-2]
                valid_label = items[-2]
                user2items_valid[user] = (valid_input_seq, valid_label)

                test_input_seq = valid_input_seq + [valid_label]
                test_label = items[-1]
                user2items_test[user] = (test_input_seq, test_label)

        return user2items_train, user2items_valid, user2items_test

    def collate_fn_train(self, batch: list[list[int]]) -> InputSequences:
        """
        Simple collate function for the DataLoader.
          1. Truncate input seqs that are longer than max_seq_len from the front.
          2. Pad input seqs that are shorter from the front.
          3. Slice the seqs so that the last element is used as the label.
        """
        seq_tensors = []
        # Prepare tensors by truncating or padding each sequence.
        for seq in batch:
            seq = pad_or_truncate_seq(seq, max_seq_len=self.max_seq_len)
            seq_tensors.append(seq)

        input_seqs = torch.stack(seq_tensors)

        return input_seqs

    def collate_fn_eval(self, batch: list[list[int]]) -> (InputSequences, ItemIdxs):
        """
        Essentially the same thing as collate_fn_train except for evaluation
          we have to take into consideration the positive and negative samples
          we'll be getting the logits for.

        The hidden representations of these samples are matrix multiplied with
          the hidden representations of the input sequence in order to get
          predictions.
        """
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
        data: dict[User, list[Item]],
        split: str = "train",
    ) -> DataLoader:
        """
        Create and return a DataLoader. Not considering users in this setting.

        1. If split == 'train':
             dataset -> list[list[int]]
        2. Elif split in ['valid', 'test']:
             dataset -> list[tuple[list[int], int]]
        """
        dataset = list(data.values())
        # Prepare dataset and collate function based on the split type.
        if split in ["valid", "test"]:
            shuffle = False
            collate_fn = self.collate_fn_eval

            input_seqs = [x[0] for x in dataset if x != []]
            all_pred_item_idxs = []

            # Get negative samples and append validation to
            #   input sequence for test phase.
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
