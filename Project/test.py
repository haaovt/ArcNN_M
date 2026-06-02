import torch
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



# data = scipy.io.loadmat('dataset/data.mat')
# train_X = data['X']
# train_Y = data['y']
# test_X = data['Xval']
# test_Y = data['yval']


# model = MyModel()
# z = model(torch.tensor(train_X, dtype=torch.float, requires_grad=False))
# print(z.shape)
# z[z<=0.5] = 0
# z[z>0.5] = 1
# z = z.detach().numpy()
# plt.subplot(121)
# plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
# plt.title('train_set')
# plt.subplot(122)
# plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y.ravel(), s=40, cmap=plt.cm.Spectral)
# plt.title('test_set')
# plt.show()











