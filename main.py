import torch

print(torch.__version__)

print(torch.cuda.is_available())

# 随机矩阵
x = torch.rand(3, 4)
print(x)
print(len(x))
print(x[:1])

y = torch.empty(5,5)

print(y)
print(len(y))


