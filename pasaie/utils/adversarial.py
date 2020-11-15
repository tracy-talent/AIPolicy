"""
 Author: liujian
 Date: 2020-11-15 21:47:27
 Last Modified by: liujian
 Last Modified time: 2020-11-15 21:47:27
"""

import torch

class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm!=0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    '''
    Example
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()
        # 对抗训练
        pgd.backup()
        for t in range(K):
            pgd.attack() # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def backup(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FreeLB():
    '''
    Example
    flb = FreeLB(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label, batch_mask in data:
        # 正常训练
        delta = torch.zeros_like(tuple(batch_input.size()) + (embedding_size,)).uniform(-1, 1) * batch_mask.unsqueeze(2)
        dims = batch_mask.sum(-1) * embedding_size
        mag = rand_init_mag / torch.sqrt(dims)
        delta = delta * mag.view(-1, 1, 1)
        delta.requires_grad_()
        # 对抗训练
        grad = defaultdict(lambda: 0)
        for t in range(1, K+1):
            loss_adv = model(batch_input + delta, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad[name] += param.grad / t
            if t == 1:
                flb.backup() # first attack时备份param.data
            flb.attack() # 在embedding上添加对抗扰动
            model.zero_grad()
        flb.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = grad[name]
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def backup(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r


def adversarial_perturbation(adv, model, criterion, K=3, rand_init_mag=0., label=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        label (torch.tensor, optional): labels. Defaults to None.
    """
    if adv.__class__.__name__ == 'FGM':
        loss = criterion(model(*args), label).mean()
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        adv.attack() # 在embedding上添加对抗扰动
        loss_adv = criterion(model(*args), label).mean()
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        adv.restore() # 恢复embedding参数
    elif adv.__class__.__name__ == 'PGD':
        loss = criterion(model(*args), label).mean()
        loss.backward()
        adv.backup_grad()
        # 对抗训练
        adv.backup() # first attack时备份param.data，在第一次loss.backword()后以保证有梯度
        for t in range(K):
            adv.attack() # 在embedding上添加对抗扰动
            if t != K-1:
                model.zero_grad()
            else:
                adv.restore_grad()
            loss_adv = criterion(model(*args), label).mean()
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        adv.restore() # 恢复embedding参数
    elif adv.__class__.__name__ == 'FreeLB':
        # embedding_size = self.model.sentence_encoder.bert_hidden_size
        # delta = torch.zeros_like(tuple(args[0].size) + (embedding_size,)).uniform(-1, 1) * args[-1].unsqueeze(2)
        # dims = args[-1].sum(-1) * embedding_size
        # mag = rand_init_mag / torch.sqrt(dims)
        # delta = delta * mag.view(-1, 1, 1)
        # delta.requires_grad_()
        # 对抗训练
        grad = defaultdict(lambda: 0)
        for t in range(1, K+1):
            loss_adv = criterion(model(*args), label).mean()
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad[name] += param.grad / t
            if t == 1:
                adv.backup() # first attack时备份param.data
            adv.attack() # 在embedding上添加对抗扰动
            model.zero_grad()
        adv.restore() # 恢复embedding参数
        # 梯度更新
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = grad[name]