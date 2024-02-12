from utils.dataset_utils import MNIST_truncated
from torch.utils.data import DataLoader

a = MNIST_truncated(root='./data')
b = DataLoader(a, batch_size=100, shuffle=True)
print(b.dataset)