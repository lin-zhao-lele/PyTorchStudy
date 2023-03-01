from __future__ import print_function
import torch

# tensor 张量

# 构造一个5x3矩阵，不初始化
x = torch.empty(5, 3)
print(x)

# 构造一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 构造一个矩阵全为 0，而且数据类型是 long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 构造一个张量，直接使用数据
x = torch.tensor([5.5, 3])
print(x)

# 创建一个 tensor 基于已经存在的 tensor
x = torch.zeros(3, 3, dtype=torch.long)
x = x.new_ones(3, 3, dtype=torch.double)
# new_* methods take in sizes
print(x)

# torch.randn_like 返回一个和x类型的矩阵 原始为float型任意值
x = torch.zeros(6, 6, dtype=torch.long)
x = torch.randn_like(x, dtype=torch.float)
# override dtype!
print(x)
# result has the same size

print(x.size())
# 输出 torch.Size([6, 6]), torch.Size是一个元组，所以它支持左右的元组操作。

y = torch.rand(6, 6)
print(x + y)
print(torch.add(x, y))

result = torch.empty(6, 6)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

# 注意 任何使张量会发生变化的操作都有一个前缀 '_'。例如：
print('注意 任何使张量会发生变化的操作都有一个前缀 _ 例如：x.copy_(y)')
x.copy_(y)
print(x.t_())

# 可以使用标准的  NumPy 类似的索引操作
print(x)
print(x[:, 1])

# 如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# 如果你有一个元素 tensor ，使用 .item() 来获得这个 value
x = torch.randn(1)
print(x)
print(x.item())