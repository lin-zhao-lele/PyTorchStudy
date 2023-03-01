# 安装PyTorch

python3 -m venv venv

激活虚拟环境 

. venv/bin/activate

Note:MPS acceleration is available on MacOS 12.3+

pip3 install torch torchvision torchaudio


# 什么是 PyTorch
要介绍PyTorch之前，不得不说一下Torch。Torch是一个有大量机器学习算法支持的科学计算框架，
是一个与Numpy类似的张量（Tensor） 操作库，其特点是特别灵活，
但因其采用了小众的编程语言是Lua，所以流行度不高，这也就有了PyTorch的出现。
所以其实Torch是 PyTorch的前身，它们的底层语言相同，只是使用了不同的上层包装语言。

PyTorch 是一个基于 Python 的科学计算包，主要定位两类人群:

• NumPy 的替代品，可以利用 GPU 的性能进行计算。 

• 深度学习研究平台拥有足够的灵活性和速度

# Accelerated PyTorch training on Mac
https://developer.apple.com/metal/pytorch/

Requirements:

Mac computers with Apple silicon or AMD GPUs

-- macOS 12.3 or later

-- Python 3.7 or later

-- Xcode command-line tools: xcode-select --install

# 网络教程
PyTorch官方教程中文版

https://pytorch123.com