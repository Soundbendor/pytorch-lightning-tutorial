from abc import ABC, abstractmethod
import torch
from torch import nn
from abc import ABC, abstractmethod
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy


# Abstract class to inherit from for all classifiers
class Classifier(pl.LightningModule, ABC):
    def __init__(self, lr, shape, num_classes):
        super(Classifier, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.accuracy = Accuracy('MULTICLASS', num_classes=num_classes)
        self.example_input_array = torch.rand(shape)

    @abstractmethod
    def forward(self, x):
        pass

    def eval_batch(self, batch):
        # Make predictions
        x, y = batch
        y_hat = self(x)

        # Evaluate predictions
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        # Evaluate batch
        loss, acc = self.eval_batch(batch)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Return the loss
        return loss


    def validation_step(self, batch, batch_idx):
        # Evaluate batch
        loss, acc = self.eval_batch(batch)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # You can directly return the loss if you want to use it for checkpointing or early stopping
        return loss

    def test_step(self, batch, batch_idx):
        # Evaluate batch
        loss, acc = self.eval_batch(batch)

        # Log metrics
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Optionally, you can return any metric or a dictionary of metrics
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class DNNClassifier(Classifier):
    def __init__(self, hidden_layers, lr, shape):
        super(DNNClassifier, self).__init__(lr, shape)

        self.i2h = nn.Linear(np.product(shape), hidden_layers[0])
        self.h2h = nn.ModuleList(
            [nn.Linear(l1, l2) for l1, l2 in zip(hidden_layers[:-1], hidden_layers[1:])]
        )
        self.h2o = nn.Linear(hidden_layers[-1], 10)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.i2h(x))
        for l in self.h2h:
            x = torch.relu(l(x))
        x = self.h2o(x)
        return x


class CNNClassifier(Classifier):
    def __init__(self, hidden_layers, num_classes, lr, shape):
        super(CNNClassifier, self).__init__(lr, shape, num_classes)

        self.i2h = nn.Conv2d(1, hidden_layers[0], 3)
        self.h2h = nn.ModuleList(
            [nn.Conv2d(l1, l2, 3) for l1, l2 in zip(hidden_layers[:-1], hidden_layers[1:])]
        )
        self.h2o = nn.Linear(hidden_layers[-1] * (23 - len(hidden_layers))**2, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.i2h(x))

        for l in self.h2h:
            x = torch.relu(l(x))
        x = x.flatten(start_dim=1)
        x = self.h2o(x)
        return x
    


class LSTMClassifier(Classifier):
    def __init__(self, hidden_size, num_layers, num_classes, lr, shape, bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__(lr, shape, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = shape[-2]
        self.num_classes = num_classes

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Fully connected layer
        multiplier = 2 if bidirectional else 1
        self.h2o = nn.Linear(hidden_size * multiplier, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2 if bidirectional else hidden_size)

        # Average the outputs over time
        # out shape: (batch_size, sequence_length, hidden_size * num_directions)
        out = torch.mean(out, dim=1)

        # Decode the hidden state of the last time step
        # out = out[:, -1, :]

        
        out = self.h2o(out)

        return out