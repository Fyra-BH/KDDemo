import torch, model

a = torch.randn(16, 1, 128, 128)
net = model.StudentNet(1, 10)
print(net(a))
