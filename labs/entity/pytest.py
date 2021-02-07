import torch
import torch.nn as nn

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.l1 = nn.Linear(10, 20, bias=False)
        nn.init.constant_(self.l1.weight, 0.1)
        self.l2 = nn.Linear(20, 2, bias=False)
        nn.init.constant_(self.l2.weight, 0.1)

    def forward(self, x):
        t = self.l1(x)
        y_out = self.l2(t)
        return y_out

x = torch.rand(1, 10)
y = torch.tensor([0])
m = mymodel()
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
loss_func = nn.CrossEntropyLoss()
y_out = m(x)
loss = loss_func(y_out, y)
loss.v
print(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
for name, param in m.named_parameters():
    print(name, param.grad, param)
