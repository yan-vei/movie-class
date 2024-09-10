import torch
from torch import nn
from transformers import BertTokenizer
from dataloader import create_dataloader
from model import BertClassifier
from train import train

pos_dir_train = 'aclImdb/train/pos'
neg_dir_train = 'aclImdb/train/neg'
pos_dir_test = 'aclImdb/test/pos'
neg_dir_test = 'aclImdb/test/neg'

MAX_LEN=512
BATCH_SIZE=16
PADDING_TOKEN='-100'
NUM_CLASSES=2
LEARNING_RATE=5e-5
NUM_EPOCHS=5

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

train_dataloader = create_dataloader(pos_dir_train, neg_dir_train, tokenizer, MAX_LEN, PADDING_TOKEN, BATCH_SIZE)

model = BertClassifier(device, NUM_CLASSES).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_params(), lr=LEARNING_RATE)

train(model=model, loss=loss_func, dataloader=train_dataloader, optimizer=optimizer, num_epochs=NUM_EPOCHS)

