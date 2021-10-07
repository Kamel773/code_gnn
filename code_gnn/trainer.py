import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def measure_performance(model, device, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        all_correct = 0
        epoch_size = 0
        losses = []
        for data in test_loader:
            data = data.to(device)
            label = data.y.to(device)
            out = model(data)
            loss = loss_fn(out, label.view(-1))
            losses.append(loss.item() * len(out))
            pred = torch.argmax(out, dim=1)
            correct = pred == label
            all_correct += correct.sum().item()
            epoch_size += len(label)
        acc = all_correct / epoch_size
        return acc, sum(losses) / len(test_loader.dataset)


def train(dataset, model, device, tensorboard_writer):
    n_train = int(len(dataset) * 0.9)
    splits = [n_train, len(dataset) - n_train]
    print('Splits:', splits[0], 'train', splits[1], 'test')
    train_set, test_set = random_split(dataset, splits)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)
    model.reset_parameters()
    print(list((p.shape for p in model.parameters())))
    loss_fn = NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=50)
    log_every = 25
    for i in range(400):
        epoch_loss = 0
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        model.train()
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y.view(-1))
            loss.backward()
            epoch_loss += loss.item() * data.num_graphs
            optimizer.step()
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc, _ = measure_performance(model, device, train_loader, loss_fn)
        tensorboard_writer.add_scalar('train_acc', train_acc, i)
        tensorboard_writer.add_scalar('train_loss', train_loss, i)
        test_acc, test_loss = measure_performance(model, device, test_loader, loss_fn)
        tensorboard_writer.add_scalar('test_acc', test_acc, i)
        tensorboard_writer.add_scalar('val_loss', test_loss, i)
        if i % log_every == 0:
            # print('Epoch:', i, 'Train Accuracy:', train_acc, 'Train loss:', train_loss, 'Test Accuracy:', acc)
            print('Epoch:', i, 'Train loss:', train_loss, 'Test Accuracy:', test_acc)
        scheduler.step()