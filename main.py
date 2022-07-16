import torch
import torch.optim as optim
import os

from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from model import *
from os import path
from tqdm import tqdm

####################################
NUM_EPOCH   = 1
THREADS     = 8
BATCH_SIZE  = 32
PRINT_INTERVAL = 10
# SAVED_FILE = 'ResNet18_on_mnist.pth'
SAVED_FILE = 'StudentNet_on_mnist.pth'
####################################

def target_net():
    # return StudentNet(1, 10)
    return ResNet18(1, 10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
    transforms.Normalize(mean=[0.5], std=[0.5])])

train_set = MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True)

eval_set = MNIST(
    root="data",
    train=False,
    transform=transform,
    download=True)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=THREADS)

eval_loader = torch.utils.data.DataLoader(
    eval_set,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=THREADS)

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    cnn = target_net()
    cnn = cnn.to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn.parameters(), lr=0.04, momentum=0.9)
    optimizer = optim.Adam(cnn.parameters(), lr=0.01, weight_decay=1e-6)

    ###################### train ######################
    if not path.exists(SAVED_FILE): # 参数存在时不训练 想再次训练需要手动删除
        for epoch in range(NUM_EPOCH):
            cnt = 0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                cnt += 1
                if cnt%PRINT_INTERVAL == 0:
                    tqdm.write("loss:{}".format(loss))
        #save model
        torch.save(cnn.to('cpu').state_dict(), SAVED_FILE)

    #################### test on eval_set ####################
    cnn = target_net()
    cnn.load_state_dict(torch.load(SAVED_FILE))
    cnn.to(device)
    cnn.eval()
    corr_num = 0
    total_num = 0

    with torch.no_grad():
        for i, (datas, labels) in enumerate(eval_loader):
            outputs = cnn(datas.to(device))
            predicted = torch.argmax(outputs.detach(), dim=1)
            corr_num += torch.sum(predicted == labels.to(device))
            total_num += len(datas)

    print("eval set corr:{:0.4f}".format(corr_num / total_num))
    