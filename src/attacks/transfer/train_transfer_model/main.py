'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import os
import argparse

from .resnet import ResNet18
from attacks.utils import load_data



# Training
def train(epoch, 
            trainloader, 
            net,
            optimizer, 
            criterion,
            device):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss/len(trainloader), 100.*correct/total

def test(epoch, 
        testloader, 
        net,
        criterion,
        save_path,
        device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # torch.save(state, f'{save_path}/{data_name}.pth')
        torch.save(state, save_path)
        best_acc = acc

    return test_loss/len(testloader), acc


def main(data_name: str, 
         base_path: str, 
         need_train: bool=True):
    save_path = f"{base_path}/attacks/transfer/{data_name}"
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    net = ResNet18(in_c=3 if data_name=="cifar" else 1)
    net = net.to(device)
    if not need_train:
        checkpoint = torch.load(os.path.join(save_path, f"{data_name}.pth"), weights_only=False)
        checkpoint['net'] = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        net.load_state_dict(checkpoint['net'])
        return net
    # Data
    print('==> Preparing data..')
    trainloader, testloader = load_data(data_name, data_path=f"{base_path}/data")

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    base_epochs = 100
    warmup_epochs = 5

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=base_epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    for epoch in range(start_epoch, start_epoch+100):
        train_loss, train_acc = train(epoch, trainloader, net, optimizer, criterion, device)
        test_loss, test_acc = test(epoch, testloader, net, criterion, save_path=os.path.join(save_path, f"{data_name}.pth"), device=device)
        scheduler.step()
        print(f'Epoch {epoch}: Train Loss {train_loss:.3f} | Train Acc {train_acc:.3f}% || Test Loss {test_loss:.3f} | Test Acc {test_acc:.3f}%')
    return net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cifar")
    parser.add_argument("--base_path", type=str, default="/nfs/stak/users/hongsa/INR-Robustness")
    args = parser.parse_args()
    main(args.data_name, args.base_path)