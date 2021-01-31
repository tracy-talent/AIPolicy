"""
 Author: liujian
 Date: 2020-11-15 21:47:27
 Last Modified by: liujian
 Last Modified time: 2020-11-15 21:47:27
"""
from collections import defaultdict
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
        self.emb_backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm) and not torch.isinf(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

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
                if norm != 0 and not torch.isnan(norm) and not torch.isinf(norm):
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
                if norm != 0 and not torch.isnan(norm) and not torch.isinf(norm):
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


def adversarial_step(adv, model, K, get_loss, retain_graph=False):
    """[summary]

    Args:
        adv (object): instance object of adversarial class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        get_loss (function): function of get loss

    Returns:
        loss_adv (torch.Tensor): loss, a scalar.
    """
    ori_model = model.module if hasattr(model, 'module') else model
    if adv.__class__.__name__ == 'FGM':
        loss = get_loss()
        loss.backward(retain_graph=retain_graph)  # 反向传播，得到正常的grad
        # 对抗训练
        adv.backup() # 备份embedding
        adv.attack()  # 在embedding上添加对抗扰动
        loss_adv = get_loss()
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        adv.restore()  # 恢复embedding参数
    elif adv.__class__.__name__ == 'PGD':
        loss = get_loss()
        loss.backward(retain_graph=retain_graph)
        # 对抗训练
        adv.backup()  # first attack时备份embedding
        adv.backup_grad()  # 在第一次loss.backword()后以保证有梯度
        for t in range(K):
            adv.attack()  # 在embedding上添加对抗扰动
            if t != K - 1:
                model.zero_grad()
            else:
                adv.restore_grad()
            loss_adv = get_loss()
            if t != K - 1:
                loss_adv.backward(retain_graph=retain_graph)  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        adv.restore()  # 恢复embedding参数
    elif adv.__class__.__name__ == 'FreeLB':
        # embedding_size = self.model.sentence_encoder.bert_hidden_size
        # delta = torch.zeros_like(tuple(args[0].size) + (embedding_size,)).uniform(-1, 1) * args[-1].unsqueeze(2)
        # dims = args[-1].sum(-1) * embedding_size
        # mag = rand_init_mag / torch.sqrt(dims)
        # delta = delta * mag.view(-1, 1, 1)
        # delta.requires_grad_()
        # 对抗训练
        grad = defaultdict(lambda: 0)
        for t in range(1, K + 1):
            loss_adv = get_loss()
            if t != K:
                loss_adv.backward(retain_graph=retain_graph)  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad[name] += param.grad / t
            if t == 1:
                adv.backup()  # first attack时备份param.data
            adv.attack()  # 在embedding上添加对抗扰动
            model.zero_grad()
        adv.restore()  # 恢复embedding参数
        # 梯度更新
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = grad[name]
    
    return loss_adv


def adversarial_perturbation(adv, model, criterion, K=3, rand_init_mag=0., labels=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        labels (torch.tensor, optional): labels. Defaults to None.
    """
    use_mask = False
    if len(labels.size()) > 1 and labels.size(-1) > 1:
        use_mask = True
    ori_model = model.module if hasattr(model, 'module') else model

    def get_loss():
        logits = model(*args)
        if use_mask:
            mask = args[-1]
            if hasattr(ori_model, 'crf') and ori_model.crf is not None:
                log_likelihood = ori_model.crf(logits, labels, mask=mask, reduction='none')
                loss = -log_likelihood / torch.sum(mask, dim=-1)
            else:
                loss = torch.sum(criterion(logits.permute(0, 2, 1), labels) * mask, dim=-1) / torch.sum(mask, dim=-1)
        else:
            loss = criterion(logits, labels)
        loss = loss.mean()
        return loss

    loss_adv = adversarial_step(adv, model, K, get_loss)
    return loss_adv


def adversarial_perturbation_span_mtl(adv, model, criterion, autoweighted_loss=None, K=3, rand_init_mag=0.,
                                      start_labels=None, end_labels=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        start_labels (torch.tensor, optional): labels of span start pos. Defaults to None.
        end_labels (torch.tensor, optional): labels of span end pos. Defaults to None.
    """
    mask = args[-1]
    seqs_len = mask.sum(dim=-1)
    ori_model = model.module if hasattr(model, 'module') else model

    def get_loss():
        if 'StartPrior' in ori_model.__class__.__name__:
            start_logits, end_logits = model(start_labels, *args)
        else:
            start_logits, end_logits = model(*args)
        start_loss = torch.sum(criterion(start_logits.permute(0, 2, 1), start_labels) * mask, dim=-1) / seqs_len
        end_loss = torch.sum(criterion(end_logits.permute(0, 2, 1), end_labels) * mask, dim=-1) / seqs_len
        if autoweighted_loss is not None:
            loss = autoweighted_loss(start_loss, end_loss)
        else:
            loss = (start_loss + end_loss) / 2
        loss = loss.mean()
        return loss

    loss_adv = adversarial_step(adv, model, K, get_loss)
    return loss_adv


def adversarial_perturbation_mrc_span_mtl(adv, model, criterion, span_bce=None, autoweighted_loss=None, K=3, rand_init_mag=0., start_labels=None, end_labels=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0.
        start_labels (torch.tensor, optional): labels of span start pos. Defaults to None.
        end_labels (torch.tensor, optional): labels of span end pos. Defaults to None.
    """
    margs = args[:-2] + args[-1:]
    loss_mask, inputs_mask = args[-2], args[-1]
    inputs_seq_len, valid_seq_len = inputs_mask.sum(dim=-1), loss_mask.sum(dim=-1)

    def get_loss():
        seq_out, start_logits, end_logits = model(*margs)
        if span_bce is not None:
            span_out = []
            span_labels = []
            for i in range(start_logits.size(0)):
                spos, tpos = inputs_seq_len[i] - valid_seq_len[i] - 1, inputs_seq_len[i] - 1
                for j in range(spos, tpos):
                    if start_logits[i][j][1] <= start_logits[i][j][0]:
                        continue
                    for k in range(spos, tpos):
                        if end_logits[i][k][1] > end_logits[i][k][0]:
                            span_out.append(torch.cat([seq_out[i][j], seq_out[i][k]]))
                            if start_labels[i][j] == 1 and end_labels[i][k] == 1:
                                span_labels.append(1)
                            else:
                                span_labels.append(0)
            if len(span_out) > 0:
                span_out = torch.stack(span_out, dim=0)
                span_logits = eval('model.module' if hasattr(model, module) else 'model').span_fc(span_out).squeeze(dim=-1)
                span_labels = torch.tensor(span_labels)
        start_loss = (torch.sum(criterion(start_logits.permute(0, 2, 1), start_labels) * loss_mask, dim=-1) / valid_seq_len).mean()
        end_loss = (torch.sum(criterion(end_logits.permute(0, 2, 1), end_labels) * loss_mask, dim=-1) / valid_seq_len).mean()
        if span_bce is not None and len(span_out) > 0:
            span_loss = span_bce(span_logits, span_labels)
        if autoweighted_loss is not None:
            if span_bce is not None and len(span_out) > 0:
                loss = autoweighted_loss(start_loss, end_loss, span_loss)
            else:
                loss = autoweighted_loss(start_loss, end_loss)
        else:
            if span_bce is not None and len(span_out) > 0:
                loss = (start_loss + end_loss) / 3
            else:
                loss = (start_loss + end_loss) / 2
        return loss

    loss_adv = adversarial_step(adv, model, K, get_loss)
    return loss_adv


def adversarial_perturbation_span_attr_mtl(adv, model, criterion, autoweighted_loss=None, K=3, rand_init_mag=0.,
                                      span_labels=None, attr_labels=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        span_labels (torch.tensor, optional): labels of entity span. Defaults to None.
        attr_labels (torch.tensor, optional): labels of entity attr. Defaults to None.
    """
    mask = args[-1]
    seqs_len = mask.sum(dim=-1)
    ori_model = model.module if hasattr(model, 'module') else model
    span_sid, span_eid = ori_model.span2id['S'], ori_model.span2id['E']

    def get_loss():
        span_logits, attr_logits = model(span_labels, *args)
        if hasattr(ori_model, 'crf_span') and ori_model.crf_span is not None:
            log_likelihood = ori_model.crf_span(span_logits, span_labels, mask=mask, reduction='none') # B
            loss_span = -log_likelihood / seqs_len # B
        else:
            loss_span = criterion(span_logits.permute(0, 2, 1), span_labels) # B * S
            loss_span = torch.sum(loss_span * mask, dim=-1) / seqs_len # B
        if hasattr(ori_model, 'crf_attr') and ori_model.crf_attr is not None:
            log_likelihood = ori_model.crf_attr(attr_logits, attr_labels, mask=mask, reduction='none') # B
            loss_attr = -log_likelihood / seqs_len # B
        else:
            loss_attr = criterion(attr_logits.permute(0, 2, 1), attr_labels) # B * S
            tag_masks = ((span_labels == span_eid) | (span_labels == span_sid)).float()
            tag_len = tag_masks.sum(dim=-1)
            assert (tag_len != 0.).all()
            if (tag_len == 0.).any():
                raise ZeroDivisionError('division by zero in tag_len')
            # tag_masks = (attr_labels != attr_negid).float()
            loss_attr = torch.sum(loss_attr * tag_masks, dim=-1) / tag_len # B
        loss_span, loss_attr = loss_span.mean(), loss_attr.mean()
        if autoweighted_loss is not None:
            loss = autoweighted_loss(loss_span, loss_attr)
        else:
            loss = (loss_span + loss_attr) / 2
        return loss
    
    loss_adv = adversarial_step(adv, model, K, get_loss)
    return loss_adv


def adversarial_perturbation_span_attr_boundary_mtl(adv, model, criterion, autoweighted_loss=None, K=3, rand_init_mag=0.,
                                      span_labels=None, attr_start_labels=None, attr_end_labels=None, retain_graph=False, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        span_labels (torch.tensor, optional): labels of entity span. Defaults to None.
        attr_labels (torch.tensor, optional): labels of entity attr. Defaults to None.
    """
    mask = args[-1]
    seqs_len = mask.sum(dim=-1)
    ori_model = model.module if hasattr(model, 'module') else model
    if (seqs_len == 0.).any():
        raise ZeroDivisionError('division by zero in seqs_len')

    def get_loss():
        if 'StartPrior' in ori_model.__class__.__name__:
            span_logits, attr_start_logits, attr_end_logits = model(attr_start_labels, *args)
        else:
            span_logits, attr_start_logits, attr_end_logits = model(*args)
        if hasattr(ori_model, 'crf_span') and ori_model.crf_span is not None:
            log_likelihood = ori_model.crf_span(span_logits, span_labels, mask=mask, reduction='none') # B
            loss_span = -log_likelihood / seqs_len # B
        else:
            loss_span = criterion(span_logits.permute(0, 2, 1), span_labels) # B * S
            loss_span = torch.sum(loss_span * mask, dim=-1) / seqs_len # B
        loss_attr_start = criterion(attr_start_logits.permute(0, 2, 1), attr_start_labels) # B * S
        loss_attr_start = torch.sum(loss_attr_start * mask, dim=-1) / seqs_len # B
        loss_attr_end = criterion(attr_end_logits.permute(0, 2, 1), attr_end_labels) # B * S
        loss_attr_end = torch.sum(loss_attr_end * mask, dim=-1) / seqs_len # B
        loss_span, loss_attr_start, loss_attr_end = loss_span.mean(), loss_attr_start.mean(), loss_attr_end.mean()
        if torch.abs(loss_span) > 10:
            loss_span = 0.
        if autoweighted_loss is not None:
            loss = autoweighted_loss(loss_span, loss_attr_start, loss_attr_end)
        else:
            loss = (loss_span + loss_attr_start + loss_attr_end) / 3
        return loss
    
    loss_adv = adversarial_step(adv, model, K, get_loss, retain_graph)
    return loss_adv


def adversarial_perturbation_span_attr_boundary_together_mtl(adv, model, criterion, autoweighted_loss=None, K=3, rand_init_mag=0.,
                                      span_labels=None, attr_labels=None, retain_graph=False, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        span_labels (torch.tensor, optional): labels of entity span. Defaults to None.
        attr_labels (torch.tensor, optional): labels of entity attr. Defaults to None.
    """
    mask = args[-1]
    seqs_len = mask.sum(dim=-1)
    ori_model = model.module if hasattr(model, 'module') else model
    if (seqs_len == 0.).any():
        raise ZeroDivisionError('division by zero in seqs_len')

    def get_loss():
        span_logits, attr_logits = model(*args)
        if hasattr(ori_model, 'crf_span') and ori_model.crf_span is not None:
            log_likelihood = ori_model.crf_span(span_logits, span_labels, mask=mask, reduction='none') # B
            loss_span = -log_likelihood / seqs_len # B
        else:
            loss_span = criterion(span_logits.permute(0, 2, 1), span_labels) # B * S
            loss_span = torch.sum(loss_span * mask, dim=-1) / seqs_len # B
        loss_attr = criterion(attr_logits.permute(0, 2, 1), attr_labels) # B * S
        loss_attr = torch.sum(loss_attr * mask, dim=-1) / seqs_len # B
        loss_span, loss_attr = loss_span.mean(), loss_attr.mean()
        if autoweighted_loss is not None:
            loss = autoweighted_loss(loss_span, loss_attr)
        else:
            loss = (loss_span + loss_attr) / 2
        return loss
    
    loss_adv = adversarial_step(adv, model, K, get_loss, retain_graph)
    return loss_adv


def adversarial_perturbation_xlnet_ner(adv, model, criterion, K=3, rand_init_mag=0.,
                                      labels=None, *args):
    """adversarial perturbation process

    Args:
        adv (object): instance object of adversarial class
        criterion (object): instance object of loss class
        model (object): instance object of model class
        K (int, optional): number of perturbation . Defaults to 3.
        rand_init_mag (float, optional): used for FreeLB's initial perturbation . Defaults to 0..
        span_labels (torch.Tensor, optional): labels of entity span. Defaults to None.
        attr_labels (torch.Tensor, optional): labels of entity attr. Defaults to None.
    """
    mask = args[-1]
    seqs_len = mask.sum(dim=-1)
    ori_model = model.module if hasattr(model, 'module') else model

    def get_loss():
        logits = model(*args)
        if hasattr(ori_model, 'crf') and ori_model.crf is not None:
            log_likelihood = []
            for i in range(labels.size(0)):
                log_likelihood.append(ori_model.crf(logits[i][-seqs_len[i]:].unsqueeze(0), labels[i][-seqs_len[i]:].unsqueeze(0), mask=mask[i][-seqs_len[i]:].unsqueeze(0), reduction='none'))
            log_likelihood = torch.cat(log_likelihood, dim=0)
            # log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
            loss = -log_likelihood / seqs_len
        else:
            loss_span = criterion(logits.permute(0, 2, 1), span_labels) # B * S
            loss_span = torch.sum(loss_span * mask, dim=-1) / seqs_len # B
        loss = loss.mean()
        return loss
    
    loss_adv = adversarial_step(adv, model, K, get_loss)
    return loss_adv
