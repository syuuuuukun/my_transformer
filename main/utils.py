import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset


def weights_init(m, norm_type="normal"):
    classname = m.__class__.__name__
    if classname.find('Norm') != -1:
        if norm_type == "normal":
            init.normal_(m.weight, 1.0, 0.02)
            init.constant_(m.bias, 0.0)
        elif norm_type == "xavier":
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0.0)
        elif norm_type == "orthogonal":
            init.orthogonal_(m.weight, np.sqrt(2))
            init.constant_(m.bias, 0.0)

    elif classname.find('Linear') != -1:

        if norm_type == "normal":
            init.normal_(m.weight, 0, 0.02)
            init.constant_(m.bias, 0.0)
        elif norm_type == "xavier":
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0.0)
        elif norm_type == "orthogonal":
            init.orthogonal_(m.weight, np.sqrt(2))
            init.constant_(m.bias, 0.0)

    elif classname.find('Embedding') != -1:
        if norm_type == "normal":
            init.normal_(m.weight, 0, 0.02)
        elif norm_type == "xavier":
            init.xavier_uniform_(m.weight)
        elif norm_type == "orthogonal":
            init.orthogonal_(m.weight, np.sqrt(2))


def to_list(x):
    return x.detach().cpu().numpy().tolist()


class MyDataset(Dataset):
    def __init__(self, train_ja, train_en, ja_enc, en_enc, ja_dic, en_dic, padding_len=128, padding_id=8000,
                 tokenizer="spm"):
        self.train_ja = train_ja
        self.train_en = train_en

        self.ja_enc = ja_enc
        self.en_enc = en_enc

        self.ja_dic = ja_dic
        self.en_dic = en_dic

        self.padding_len = padding_len
        self.padding_id = padding_id

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_ja)

    def __getitem__(self, idx):
        ## データ読み込み
        data = self.train_ja[idx]
        target = self.train_en[idx]
        if self.tokenizer == "spm":
            data, input_target, target = self.sentencepiece_tokenizer(data, target)
        else:
            data, input_target, target = self.word_tokenizer(data, target)

        return data, input_target, target

    def make_padding(self, x, padding_id=None):
        if padding_id is None:
            x = x + [self.padding_id] * (self.padding_len - len(x))
        elif padding_id is not None:
            x = x + [padding_id] * (self.padding_len - len(x))
        return x

    def sentencepiece_tokenizer(self, data, target):
        ## sentencepiece_tokenize
        data = self.ja_enc.encode_as_ids(data)
        target = self.en_enc.encode_as_ids(target)

        ## BOSとEOSトークンidの追加
        input_target = [self.en_enc.bos_id()] + target
        target = target + [self.en_enc.eos_id()]

        data = to_tensor(self.make_padding(data))
        input_target = to_tensor(self.make_padding(input_target))
        target = to_tensor(self.make_padding(target))

        return data, input_target, target

    def word_tokenizer(self, data, target):
        bos_id = len(self.en_dic)
        eos_id = len(self.en_dic) + 1

        data = [self.ja_dic[word] for word in data.split()]
        target = [self.en_dic[word] for word in target.split()]

        input_target = [bos_id] + target
        target = target + [eos_id]

        data = to_tensor(self.make_padding(data, padding_id=len(self.ja_dic)))
        input_target = to_tensor(self.make_padding(input_target, padding_id=eos_id + 1))
        target = to_tensor(self.make_padding(target, padding_id=eos_id + 1))

        return data, input_target, target


def to_tensor(x):
    if isinstance(x, list):
        return torch.Tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)


def generate_attention_mask(inp, inp_tgt, pad_id_inp, pad_id_tgt):
    ## encoderのAttentionMask
    inp_mask = inp != pad_id_inp
    inp_masks = torch.bmm(inp_mask.unsqueeze(-1).float(), inp_mask.unsqueeze(1).float())

    ## masked_MHAのAttentionMask
    b, l = inp_tgt.shape
    mask = (torch.triu(torch.ones(b, l, l)) == 1).transpose(1, 2).to(inp_tgt.device)
    tgt_mask = (inp_tgt != pad_id_tgt).unsqueeze(-2)
    tgt_masks = (mask & tgt_mask).float()

    ## source_target_attentionMask
    memory_mask = torch.matmul(inp_mask.float().unsqueeze(-1), tgt_mask.float())
    memory_masks = memory_mask

    return inp_masks.unsqueeze(1), tgt_masks.unsqueeze(1), memory_masks.unsqueeze(1)


def count_param(model):
    params = 0
    for p in model.parameters():
        if isinstance(p, nn.Parameter):
            params += len(p.view(-1))
    print(f"パラメータ数{round(params / (10 ** 6), 2)}M")


def make_dict(data):
    data = [word for text in data.tolist() for word in text.split()]
    data = set(data)
    dicts = {}
    for i, word in enumerate(data):
        dicts[word] = i
    return dicts