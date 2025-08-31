from torch import nn
from collections import OrderedDict


# 简易分类器构造类
# 根据设置的维度动态创建层数
class SimpleClassifier(nn.Module):
    def __init__(self, input_dims: list[int], output_dims: list[int]):
        super(SimpleClassifier, self).__init__()
        assert len(input_dims) == len(output_dims), "The shape of input dims must same as the shape of hidden dims"
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.relu = nn.ReLU()
        self.layers = self.create_layers()

    def forward(self, x):
        out = self.layers(x)
        return out

    def create_layers(self) -> nn.Sequential:
        layers = {}
        for idx, (input_dim, output_dim) in enumerate(zip(self.input_dims, self.output_dims)):
            layers.update({f"fc{idx}": nn.Linear(input_dim, output_dim)})

            if idx == len(self.input_dims) - 1:
                break

            layers.update({f"relu{idx}": self.relu})

        return nn.Sequential(OrderedDict(layers))

    def layers_size(self):
        return len(self.input_dims) + 1

    def nodes_of_layers(self):
        return self.input_dims + [self.output_dims[-1]]
