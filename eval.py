from PIL import Image
import numpy as np
import torch
import model
import sys

net = model.ResNet18(1, 10)
net.load_state_dict(torch.load("ResNet18_on_mnist.pth"))
net.eval()

if __name__ == "__main__":
    imgfile = sys.argv[1]
    img = Image.open(imgfile).convert("L").resize((128, 128))
    img = 1 - np.array(img) / 255. * 2
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() # 增加一个维度
    print(img.shape)
    out = net(img)
    predicted = torch.argmax(out)
    print("predicted: ", int(predicted))