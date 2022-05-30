import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from transformers import BertJapaneseTokenizer

import pandas as pd


class FAQDataset(Dataset):
    def __init__(self, config, data, targets, tokenizer):
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.classes = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_row = self.data.iloc[index]
        question = data_row[self.config.data.question_column]
        answer = data_row[self.config.data.answer_column]
        labels = self.classes.index(answer)

        question_encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            questions=question,
            input_ids=question_encoding['input_ids'].flatten(),
            attention_mask=question_encoding['attention_mask'].flatten(),
            labels=torch.tensor(labels),
        )


class FAQDataModule(pl.LightningDataModule):
    def __init__(self, config, targets):
        super().__init__()
        self.config = config
        self.targets = targets

        self.train_df, self.test_df = self.load_dataset()

        self.batch_size = config.training.batch_size
        self.max_token_len = config.trainig.max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            self.config.model.pretrained_model)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.all_dataset = FAQDataset(
                self.config, self.train_df, self.targets, self.tokenizer, )
            train_size = int(len(self.all_dataset) *
                             self.config.train_valid_split)
            valid_size = len(self.all_dataset) - train_size
            self.train_dataset, self.valid_dataset = random_split(
                self.all_dataset, [train_size, valid_size])

        if stage == 'test' or stage is None:
            self.test_dataset = FAQDataset(
                self.config, self.test_df, self.targets, self.tokenizer, )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def load_dataset(self):
        train_data_dir = os.path.join(
            self.config.path.project_path, 'data', self.config.path.client_name, 'train')
        train_data_list = os.listdir(train_data_dir)
        df_list = []
        for data in train_data_list:
            if data.endswith('.csv'):
                df = pd.read_csv(os.path.join(train_data_dir, data))
                df_list.append(df)
        all_df = pd.concat(df_list)
        all_df = all_df.sample(frac=1)

        train_df = all_df.iloc[:-self.config.data.test_data_size, :]
        test_df = all_df.iloc[-self.config.data.test_data_size:, :]

        return train_df, test_df
