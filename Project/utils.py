import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Work_2 utils
def trainer(model, train_loader, test_loader, num_epochs, batch_size, optimizer, loss_type, n_epoch_to_log = 1, bin = False, cal_test_loss=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    acc_list = []
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_test_loss = 0
        for data,target in train_loader:
            data = data.view(data.shape[0],-1)

            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            pred = model(data)
            if bin:
                pred = torch.flatten(pred).squeeze(-1)
 
            loss = loss_type(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if cal_test_loss:
                model.eval()
                with torch.no_grad():
                    for data,target in test_loader:
                        data = data.view(data.shape[0],-1)
                        # target = target.unsqueeze(-1)
                        data = data.to(device = DEVICE)
                        target = target.to(device = DEVICE)
                        pred = model(data)
                        if bin:
                            pred = torch.flatten(pred).squeeze(-1)     
                        loss = loss_type(pred, target)
                        total_test_loss += loss.item()
                model.train()

        train_acc = check_accuracy(train_loader, model, batch_size,bin)
        test_acc = check_accuracy(test_loader, model, batch_size,bin)
        
        if epoch % n_epoch_to_log == 0:
            # Print results after n epoch
            print(f"Epoch {epoch+1}/{num_epochs}: Train_acc is {train_acc:.2f}%. Test_acc is {test_acc:.2f}%. Train avg loss is {total_train_loss/len(train_loader):.10f}.", end=" ")
            if cal_test_loss:
                print(f"Test avg loss is {total_test_loss/len(test_loader):.10f}")
            else:
                print("")
       
        acc_list.append([train_acc, test_acc, total_train_loss, total_test_loss])
        

    return model, acc_list

def check_accuracy(loader, model, batch_size, bin=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.shape[0],-1)

            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            pred = model(data)
            if bin:
                pred = torch.flatten(pred).squeeze(-1)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                num_corrects += torch.eq(pred, target).sum()
            else:
                _, pred = pred.max(1) # same as np.argmax()
                num_corrects += torch.eq(pred, target).sum()
                
            num_samples += min(batch_size, data.shape[0])
            acc = float(num_corrects)/float(num_samples)*100

    model.train()
    return acc


# Work_3 utils
def trainer_pred5(model, train_loader, test_loader, num_epochs, batch_size, optimizer, loss_type, n_epoch_to_log = 1, cal_test_loss=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    acc_list = []
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_test_loss = 0
        for data,target in train_loader:
            data = data.view(data.shape[0],-1)
            target_5 = torch.zeros(len(target)).unsqueeze(-1)
            target_5[target==5] = 1

            data = data.to(device = DEVICE)
            target_5 = target_5.to(device = DEVICE)

            pred = model(data)
            loss = loss_type(pred, target_5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if cal_test_loss:
                model.eval()
                with torch.no_grad():
                    for data,target in test_loader:
                        data = data.view(data.shape[0],-1)
                        # target = target.unsqueeze(-1)
                        data = data.to(device = DEVICE)
                        target = target.to(device = DEVICE)
                        pred = model(data)
                        if bin:
                            pred = torch.flatten(pred).squeeze(-1)                             
                        loss = loss_type(pred, target)
                        total_test_loss += loss.item()
                model.train()

        train_acc = check_accuracy_pred5(train_loader, model, batch_size)
        test_acc = check_accuracy_pred5(test_loader, model, batch_size)

        if epoch % n_epoch_to_log == 0:
            # Print results after n epoch
            print(f"Epoch {epoch+1}/{num_epochs}: Train_acc is {train_acc:.2f}%. Test_acc is {test_acc:.2f}%. Train avg loss is {total_train_loss/len(train_loader):.10f}.", end=" ")
            if cal_test_loss:
                print(f"Test avg loss is {total_test_loss/len(test_loader):.10f}")
            else:
                print("")
        acc_list.append([train_acc, test_acc, total_train_loss, total_test_loss])


    return model, acc_list

def check_accuracy_pred5(loader, model, batch_size):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.shape[0],-1)
            target_5 = torch.zeros(len(target)).unsqueeze(-1)
            target_5[target==5] = 1

            data = data.to(device = DEVICE)
            target_5 = target_5.to(device = DEVICE)

            pred = model(data)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            num_corrects += torch.eq(pred, target_5).sum()
            num_samples += min(batch_size, data.shape[0])
            acc = float(num_corrects)/float(num_samples)*100

    model.train()
    return acc

def plot_curve(values, curve_type):
    items = torch.vstack([torch.tensor(item) for item in values])
    train_acc = items[:,0]
    test_acc = items[:,1]
    train_loss = items[:,2]
    test_loss = items[:,3]
    time = torch.arange(len(train_acc))
    
    if curve_type == 'acc':
        plt.plot(time, train_acc, label='Train acc')
        plt.plot(time, test_acc, label='Test acc')
        plt.legend()
        plt.show()
    elif curve_type == 'loss':
        plt.subplot(211)
        plt.plot(time, train_loss, label='Train loss')
        plt.legend()
        plt.subplot(212)
        plt.plot(time, test_loss, label='Test loss')
        plt.legend()
        plt.show()  
    elif curve_type == 'both':
        plt.subplot(311)
        plt.plot(time, train_acc, label='Train acc')
        plt.plot(time, test_acc, label='Test acc')
        plt.legend()
        plt.subplot(312)
        plt.plot(time, train_loss, label='Train loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(time, test_loss, label='Test loss')
        plt.legend()
        plt.show()  
    return


# Work_4 utils
def get_syn_dataset(n_samples, param, func_type = 'lin'):
    if func_type == 'lin':
        n_dim, x_range, w, bias, err = param[0], param[1], param[2], param[3], param[4]

        x = (torch.rand((n_samples, n_dim), dtype=torch.float) + x_range[0]) * (x_range[1] - x_range[0]) 
        error = torch.normal(err[0], err[1], size = (n_samples,1), dtype=torch.float)
        y = bias + torch.sum(x * w, dim = 1) + error

    elif func_type == 'logistic':
        n_dim, x_range, w, bias, err = param[0], param[1], param[2], param[3], param[4]

        x = (torch.rand((n_samples, n_dim), dtype=torch.float) + x_range[0]) * (x_range[1] - x_range[0]) 
        error = torch.normal(err[0], err[1], size = (n_samples,1), dtype=torch.float)
        y = torch.sigmoid(bias + torch.sum(x * w, dim = 1) + error)

    elif func_type == 'gauss':
        n_cluster = len(param)
        x = []
        y = []
        normal = np.random.multivariate_normal
        for i in range(n_cluster):
            mean = param[i]['mean']
            cov = param[i]['cov']
            label = param[i]['label']
            rv = normal(mean, cov, n_samples//n_cluster)
            x.append(torch.from_numpy(rv))
            y.append(torch.ones(n_samples//n_cluster) * label)
        x = torch.cat(x, dim=0).to(dtype=torch.float)
        y = torch.cat(y, dim=0)

    dataset = SynDataset(x, y)
    
    return dataset


class SynDataset(Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:], self.y[index], 

class SynCollate:
    def __init__(self):
        pass

    def __call__(self,batch):
        data = [item[:-1][0] for item in batch]
        targets = [item[-1] for item in batch]
        data = torch.stack(data,dim=0)
        targets = torch.stack(targets,dim=0)
        return data, targets


def plot_decision_boundary(x, y, model):
    if x.shape[1] != 2:
        raise ValueError('Expected x to have dimension of (n_samples,2)')
    
    x0_min, x0_max = min(x[:,0]) * (1-0.2*np.sign(min(x[:,0]))), max(x[:,0]) * (1+0.2*np.sign(max(x[:,0])))
    x1_min, x1_max = min(x[:,1]) * (1-0.2*np.sign(min(x[:,1]))), max(x[:,1]) * (1+0.2*np.sign(max(x[:,1])))

    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, ((x0_max-x0_min)/500)), 
                         np.arange(x1_min, x1_max, ((x1_max-x1_min)/500)))
    
    grid_x = torch.from_numpy(np.column_stack((x0.ravel(), x1.ravel()))).to(torch.float)
    pred = np.round(model(grid_x.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))).detach().cpu().numpy())

    pred = pred.reshape(x0.shape)
    plt.contourf(x0, x1, pred, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()

# Work_5 utils
def trainer_CNN(model, train_loader, test_loader, num_epochs, batch_size, optimizer, loss_type, n_epoch_to_log = 1, bin = False, cal_test_loss=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    acc_list = []
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_test_loss = 0
        for data,target in train_loader:
            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            pred = model(data)
            if bin:
                pred = torch.flatten(pred).squeeze(-1)

            loss = loss_type(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if cal_test_loss:
                model.eval()
                with torch.no_grad():
                    for data,target in test_loader:            
                        data = data.to(device = DEVICE)
                        target = target.to(device = DEVICE)
                        pred = model(data)
                        if bin:
                            pred = torch.flatten(pred).squeeze(-1)     
                        loss = loss_type(pred, target)
                        total_test_loss += loss.item()
                model.train()

        train_acc = check_accuracy_CNN(train_loader, model, batch_size,bin)
        test_acc = check_accuracy_CNN(test_loader, model, batch_size,bin)
        
        if epoch % n_epoch_to_log == 0:
            # Print results after n epoch
            print(f"Epoch {epoch+1}/{num_epochs}: Train_acc is {train_acc:.2f}%. Test_acc is {test_acc:.2f}%. Train avg loss is {total_train_loss/len(train_loader):.10f}.", end=" ")
            if cal_test_loss:
                print(f"Test avg loss is {total_test_loss/len(test_loader):.10f}")
            else:
                print("")
                
        acc_list.append([train_acc, test_acc, total_train_loss, total_test_loss])
        

    return model, acc_list

def check_accuracy_CNN(loader, model, batch_size, bin=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=DEVICE)
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            pred = model(data)

            if bin:
                pred = torch.flatten(pred).squeeze(-1)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                num_corrects += torch.eq(pred, target).sum()
            else:
                _, pred = pred.max(1) # same as np.argmax()
                num_corrects += torch.eq(pred, target).sum()
                
            num_samples += min(batch_size, data.shape[0])
            acc = float(num_corrects)/float(num_samples)*100

    model.train()
    return acc


# visualize
def visualize_pca(self, loader):
    '''
        Use this function to visualize features using PCA
    '''
    fea_list = []
    classes = [0,1,2,3,4,5,6,7,8,9]
    self.model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.shape[0],-1)

            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            fea = self.model(data)

            batch_data = torch.hstack((target.unsqueeze(-1),fea))
            fea_list.append(batch_data)

    fea_list = torch.cat(fea_list, dim=0) # size (num_samples,label+features)
    fea_list = fea_list.cpu().numpy()
    u,_,_ = torch.pca_lowrank(fea_list[:,1:],center=False)

    data_points = torch.hstack((fea_list[:,0].unsqueeze(-1), u[:,:2]))

    fig = plt.figure()
    ax = plt.subplot(111)
    for label in classes:
        scat = [point for point in data_points if point[0]==label]
        scat = np.stack(scat,dim=0)
        ax.scatter(scat[:,1],scat[:,2], label=f'{label}')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if loader.dataset.train:
        ax.set_title("Visualize features on train dataset")
    else:
        ax.set_title("Visualize features on test dataset")
    plt.grid()
    plt.show()

    self.model.train()
    return

def visualize_tsne(self, loader):
    '''
        Use this function to visualize features using PCA
    '''
    fea_list = []
    classes = [0,1,2,3,4,5,6,7,8,9]
    self.model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.shape[0],-1)

            data = data.to(device = DEVICE)
            target = target.to(device = DEVICE)

            fea = self.model(data)

            batch_data = torch.hstack((target.unsqueeze(-1),fea))
            fea_list.append(batch_data)

    fea_list = torch.cat(fea_list, dim=0) # size (num_samples,label+features)
    fea_list = fea_list.cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300)
    tsne_results = tsne.fit_transform(fea_list)

    data_points = torch.hstack((fea_list[:,0].unsqueeze(-1), torch.from_numpy(tsne_results)))

    fig = plt.figure()
    ax = plt.subplot(111)
    for label in classes:
        scat = [point for point in data_points if point[0]==label]
        scat = np.stack(scat,dim=0)
        ax.scatter(scat[:,1],scat[:,2], label=f'{label}')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if loader.dataset.train:
        ax.set_title("Visualize features on train dataset")
    else:
        ax.set_title("Visualize features on test dataset")
    plt.grid()
    plt.show()

    self.model.train()
    return


