import torch
from torch import nn
from transformers import BertTokenizer
from dataloader import create_dataloader
from model import BertClassifier
from train import train, evaluate
import config

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenize and load datasets
tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER)
train_dataloader = create_dataloader(config.POS_DIR_TRAIN, config.NEG_DIR_TRAIN, tokenizer,
                                     config.MAX_LENGTH, config.PADDING_TOKEN, config.BATCH_SIZE)
test_dataloader = create_dataloader(config.POS_DIR_TEST, config.NEG_DIR_TEST, tokenizer,
                                    config.MAX_LENGTH, config.PADDING_TOKEN, config.BATCH_SIZE)

# Load model
model = BertClassifier(device, config.NUM_CLASSES).to(device)

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_params(), lr=config.LEARNING_RATE)

# Train and evaluate model; print accuracies
train(model=model, loss=loss_func, dataloader=train_dataloader, optimizer=optimizer, num_epochs=config.NUM_EPOCHS)
evaluate(model=model, loss=loss_func, dataloader=test_dataloader)
