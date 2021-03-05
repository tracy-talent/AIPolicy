import torch
from torch import nn, optim
from .base_model import SentenceRE

class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, dropout_rate=0.5):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout(dropout_rate)
        if 'Entity-Origin(e1,e2)' in self.rel2id:
            self.layernorm = nn.LayerNorm(num_class, elementwise_affine=True) # for Semeval and Fewrel
        for rel, rid in rel2id.items():
            self.id2rel[rid] = rel

    def infer(self, item):
        self.eval()
        with torch.no_grad():
            seqs = list(self.sentence_encoder.tokenize(item))
            if len(seqs) >= 6:
                seqs = seqs[:3] + seqs[5:]
            # if list(self.sentence_encoder.parameters())[0].device.type.startswith('cuda'):
            #     for i in range(len(seqs)):
            #         seqs[i] = seqs[i].cuda()
            logits = self.forward(*seqs)
            logits = self.softmax(logits)
            score, pred = logits.max(-1)
            score = score.item()
            pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        if 'Entity-Origin(e1,e2)' in self.rel2id:
            logits = self.layernorm(logits) # (B, N), for Semeval and Fewrel
        return logits
