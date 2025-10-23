#Implement a simple MLP for classification (e.g., MNIST dataset) using
#TensorFlow/Keras or PyTorch with different train-test splits. Vary the different
#optimizers and plot the accuracy.

import torch, matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

def run_model(train_dl, test_dl, opt_name):
    model = MiniMLP(); crit = nn.CrossEntropyLoss()
    optimizer = getattr(optim,opt_name)(model.parameters(),lr=0.01)
    tr_acc,test_acc = [],[]
    for _ in range(5):
        model.train(); correct,total=0,0
        for imgs, lbls in train_dl:
            out = model(imgs); loss = crit(out,lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            _, pred = torch.max(out,1); total+=lbls.size(0); correct+=(pred==lbls).sum().item()
        tr_acc.append(correct/total)
        model.eval(); correct,total=0,0
        with torch.no_grad():
            for imgs,lbls in test_dl:
                _, pred = torch.max(model(imgs),1); total+=lbls.size(0); correct+=(pred==lbls).sum().item()
        test_acc.append(correct/total)
    return tr_acc,test_acc

tfm = transforms.ToTensor()
data = datasets.MNIST(root='./data',train=True,download=True,transform=tfm)
splits=[0.8,0.7,0.6]; opts=['SGD','Adam','RMSprop']; results={}

for s in splits:
    n_train = int(s*len(data)); n_test = len(data)-n_train
    tr_set, te_set = random_split(data,[n_train,n_test])
    tr_dl = DataLoader(tr_set,batch_size=64,shuffle=True)
    te_dl = DataLoader(te_set,batch_size=64,shuffle=False)
    for o in opts:
        key=f"{o}_{int(s*100)}"
        print("Training",key)
        tr,test = run_model(tr_dl,te_dl,o)
        results[key]=(tr,test)

for k,(tr,test) in results.items():
    plt.plot(tr,label=f"{k}_train")
    plt.plot(test,label=f"{k}_test")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("MLP Accuracy for Optimizers & Splits")
plt.legend(); plt.grid(True); plt.show()
