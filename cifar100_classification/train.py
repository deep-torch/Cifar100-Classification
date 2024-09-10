import os
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from cifar100_classification.utils import load_checkpoint, save_checkpoint, get_category_mapping
from cifar100_classification.model import ClassificationModel
from cifar100_classification.data import get_dataloaders


def train(trainloader, model, loss_fn, optimizer, device, category_mapping):
    model.train()
    size = len(trainloader.dataset)
    running_loss = 0.0
    corrects = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs_general, outputs_special = model(inputs)
        labels_general = torch.tensor([category_mapping[label.item()] for label in labels]).to(device)

        # Compute loss
        loss = loss_fn(outputs_general, labels_general) + loss_fn(outputs_special, labels)

        optimizer.zero_grad()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        corrects += (outputs_special.argmax(1) == labels).type(torch.float).sum().item()

        if i % 50 == 0:
            print(f"Train loss: {loss.item():>7f} [{(i+1) * len(inputs):>5d}/{size:>5d}]")

        del inputs
        del labels
        torch.cuda.empty_cache()

    corrects /= size
    print(f"Train Accuracy: {100 * corrects:>0.1f}%")


def test(testloader, model, loss_fn, device, category_mapping):
    model.eval()
    test_loss = 0
    corrects = 0
    size = len(testloader.dataset)

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs_general, outputs_special = model(inputs)
            labels_general = torch.tensor([category_mapping[label.item()] for label in labels]).to(device)

            # Compute loss
            test_loss += loss_fn(outputs_general, labels_general).item() + loss_fn(outputs_special, labels).item()
            corrects += (outputs_special.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= len(testloader)
    corrects /= size
    print(f"Test Accuracy: {100 * corrects:>0.1f}%, Avg loss: {test_loss:>8f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # Load data
    trainloader, testloader = get_dataloaders(args.batch_size)

    # Load model
    model = ClassificationModel().to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.conv_layer.parameters(), "lr": args.lr_conv},
        {"params": model.linear_layer_general.parameters(), "lr": args.lr_fc},
        {"params": model.linear_layer_special.parameters(), "lr": args.lr_fc},
    ], weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=[1, 5, 8], gamma=0.1, verbose=True)

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)

    category_mapping = get_category_mapping(trainloader.dataset)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}\n---------------------------------")
        train(trainloader, model, criterion, optimizer, device, category_mapping)
        test(testloader, model, criterion, device, category_mapping)
        scheduler.step()

        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch, os.path.join(args.checkpoint_path, f'checkpoint_epoch_{epoch}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-100 Classifier")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr_conv', type=float, default=5e-4, help='Learning rate for conv layers')
    parser.add_argument('--lr_fc', type=float, default=1e-3, help='Learning rate for fully connected layers')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to a checkpoint file to resume training')

    args = parser.parse_args()
    main(args)
