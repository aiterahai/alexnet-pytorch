"""
file name: train.py

create time: 2023-02-16 08:04
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import AlexNet

# define pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
IMAGE_DIM = 227
NUM_CLASSES = 2
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
TRAIN_IMG_DIR = 'dataset/imagenet'
CHECKPOINT_DIR = 'checkpoint/'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if __name__ == "__main__":
    seed = torch.initial_seed()
    print(f"Used seed : {seed}")

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    if device == "cuda":
        alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)

    # create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # create optimizer
    # SGD was used In the original paper, but which doesn't train
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    print('Optimizer created')

    # divide LR by 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # train
    total_steps = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        # print out epoch and loss values
        print(f"EPOCH : {epoch} LOSS : {running_loss / len(dataloader)}")

    # save checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "alexnet_pt_epochs_{epoch}")
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": alexnet.state_dict(),
        "seed": seed
    }
    torch.save(state, checkpoint_path)