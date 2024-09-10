import torch


def train(model, dataloader, loss, optimizer, num_epochs):
    """
    Training loop for the sentiment classification task.
    :param model: BertModel
    :param dataloader: Dataloader with training data
    :param loss: loss function, e.g. cross entropy
    :param optimizer: e.g. AdamW
    :param num_epochs: number of epochs to train from config
    :return: void
    """
    # Set the correct model mode
    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} starting...')
        losses = []
        correct = 0

        for idx, batch in enumerate(dataloader):
            print(f'Training batch {idx+1}/{len(dataloader)}...')

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            label = batch['label'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_per_batch = loss(outputs, label)

            _, predicted = torch.max(outputs, dim=1)
            correct += torch.sum(predicted == label)
            losses.append(loss_per_batch.item())

            loss_per_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'\tEpoch {epoch+1}/{num_epochs}. \nAccuracy: {correct/len(dataloader.dataset)}. '
              f'Loss: {sum(losses)/len(dataloader)}')


def evaluate(model, dataloader, loss):
    """
    Evaluation loop for the sentiment classification task.
    :param model: BertModel
    :param dataloader: Dataloader with evaluation data
    :param loss: loss function, e.g. cross entropy
    :return: void
    """
    # Set the correct model mode
    model = model.eval()
    losses = []
    correct = 0

    # Disable backprop
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f'Evaluating batch {idx+1}/{len(dataloader)}...')

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            label = batch['label'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_per_batch = loss(outputs, label)

            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == label)
            losses.append(loss_per_batch.item())

    print(f'\tAccuracy on validation: {correct / len(dataloader.dataset)}. '
          f'Validation loss: {sum(losses) / len(dataloader)}.')
