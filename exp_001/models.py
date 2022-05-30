from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
import os
import torch
from torch import nn


class FAQNet(nn.Module):
    def __init__(self, config, num_class):
        super().__init__()
        self.config = config
        self.num_class = num_class
        self.bert: AutoModel = AutoModel.from_pretrained(
            self.config.model.pretrained_model,
            return_dict=True,
            output_hidden_states=True).cuda()
        self.bert_config = AutoConfig.from_pretrained(
            self.config.model.pretrained_model)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

        self.dropout = nn.Dropout(p=self.config.training.dropout_rate)
        self.classifier = nn.Linear(
            self.bert_config.hidden_size, self.num_class)

        self.__init__weights()

    def __init__weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input, attention_mask):
        output = self.bert(input, attention_mask)
        hidden_states = output['hidden_states'][-1]
        weights = self.attention(hidden_states)
        context_vector = torch.sum(hidden_states*weights, dim=1)

        output = self.dropout(context_vector)
        output = self.classifier(output)

        return output


class QAModel(pl.LightningModule):
    def __init__(self, config, answer_list: list) -> None:
        super().__init__()
        self.config = config
        self.model = FAQNet(self.config, num_class=len(answer_list))
        self.criterion = nn.CrossEntropyLoss()
        self.n_epochs = self.config.training.num_epochs

    def forward(self, input, attention_mask, labels=None):
        preds = self.model(input, attention_mask)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)

        return loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)

        return [optimizer, ], [scheduler, ]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, mode: str):

        loss, preds = self.forward(
            input=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"])

        self.log(f"{mode}_loss", loss)

        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': batch["labels"]
        }

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, "test")

    def _epoch_end(self, outputs, mode: str):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)

        self.log(f"{mode}_epoch_loss", epoch_loss, logger=True)
        # top-k accuracy
        topk = epoch_preds.topk(k=self.config.training.mrr_at_k, dim=1).indices
        num_correct_topk = [(k == epoch_labels).sum().item() for k in topk.T]
        topk_correct_accuracy = sum(num_correct_topk) / len(epoch_labels)
        # top-1 accuracy
        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_epoch_accuracy", epoch_accuracy, logger=True)
        self.log(f"{mode}_epoch_topk_accuracy",
                 topk_correct_accuracy, logger=True)
