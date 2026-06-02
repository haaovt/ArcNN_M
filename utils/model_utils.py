import torch
import torch.nn as nn
import numpy as np


# Machine Learning models to implement. Non-iterative methods like Linear Regression should have a special handler.
class MyModel(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(100,20),
                                   nn.ReLU(True),
                                   nn.Linear(20,10),
                                   nn.ReLU(True)
                                   )

    def forward(self, x):
        return self.model(x)


class MyModel2(nn.Module):
    def __init__(self, cfgs):
        # cfgs may contains model parameters
        super().__init__()
        # cfgs['model'] is str type, i.e. cfgs['model'] = 'MyModel'. In cases where many models is implemented, configurations of model should be in cfgs['MyModel'][...]

        input_size = cfgs[cfgs['model']]['input_size']
        n_classes = cfgs[cfgs['model']]['n_classes']


        self.model = nn.Sequential(nn.Linear(input_size, n_classes),
                                   nn.ReLU(True),
                                   )

    def forward(self, x):
        return self.model(x)
    



















