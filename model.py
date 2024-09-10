import torch
from transformers import BertModel

class BertClassifier(torch.nn.Module):
    """
    A sequence classification model based on BERT.
    """

    def __init__(self, device, num_classes, freeze_params=True):
        """
        Define the model
        :param device: GPU/CPU/MPS (Apple-Silicon support)
        :param num_classes: output dim of the linear layer
        """

        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = torch.nn.Linear(768, num_classes) # 768 is the output dimensionality of BERT

        # Set device
        self.device = device

        # Do not finetune the whole model if needed
        if freeze_params:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get trainable parameters of BERT
        self.params = self.get_params()


    def forward(self, input_ids, attention_mask):
        """
        Define the model's forward pass.

        :param input_seq: sequence of input tokens
        :param attention_mask: attention mask
        :return: predicted logits
        """
        # Use CLS token hidden state for classification, shape: batch_size, hidden_size
        output = self.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]

        logits = self.linear(output) # Shape: batch_size, num_classes

        return logits

    def get_params(self):
        """
        Return parameters for the optimizer to tune.

        :return: an array of optimizable parameters according to the model's definition
        """

        params = []

        for param in self.parameters():
            if param.requires_grad:
                params.append(param)

        return params