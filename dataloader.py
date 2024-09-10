import os
import torch
from torch.utils.data import Dataset, DataLoader
import random


class MovieReviewsDataset(Dataset):
    """
    Wrapper class over Dataset to process the aciimdb dataset with positive
    and negative movie reviews.
    """

    def __init__(self, pos_dir, neg_dir, tokenizer, max_length, pad_token):
        """
        Initialize a MovieReviewsDataset object.
        :param pos_dir: full path to positive reviews, e.g. acllmdb/train/pos
        :param neg_dir: full path to negative reviews
        :param tokenizer: tokenizer, e.g. BertTokenizer
        :param max_length: context window for Bert (max. 512 tokens)
        :param pad_token: padding token, e.g. -100
        """
        self.reviews = []
        self.labels = []

        # Load samples
        # 1 - positive review
        # 0 - negative review
        for file in os.listdir(pos_dir):
            with open(os.path.join(pos_dir, file), 'r', encoding='utf-8') as f:
                self.reviews.append(f.read().strip())
                self.labels.append(1)

        for file in os.listdir(neg_dir):
            with open(os.path.join(neg_dir, file), 'r', encoding='utf-8') as f:
                self.reviews.append(f.read().strip())
                self.labels.append(0)

        # Shuffle the positive and negative reviews order and unzip again
        data = list(zip(self.reviews, self.labels))
        random.shuffle(data)
        self.reviews, self.labels = zip(*data)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token

    def __len__(self):
        """
        Return the length of the dataset.
        :return: int
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """
        Return a review item from the dataset
        :param idx:
        :return: dataset item
        """

        review = self.reviews[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(review,
                                              add_special_tokens=True,
                                              max_length=self.max_length,
                                              return_token_type_ids=False,
                                              return_attention_mask=True,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt')

        return {
            'review': review,
            'label': torch.tensor(label, dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_dataloader(pos_dir, neg_dir, tokenizer, max_length, pad_token, batch_size):
    """
    Create a dataloader object for training and testing.
    :param pos_dir: full path to the positive reviews, e.g. acllmdb/train/pos
    :param neg_dir: full path to the negative reviews
    :param tokenizer: tokenizer, e.g. BertTokenizer
    :param max_length: context window for Bert (max. 512 tokens)
    :param pad_token: padding token, e.g. -100
    :param batch_size: size of the batch
    :return: Dataloader object
    """

    dataset = MovieReviewsDataset(pos_dir, neg_dir, tokenizer, max_length, pad_token)
    return DataLoader(dataset, batch_size=batch_size)