import math
import codecs
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from translate.model import translation_model

class Predicter():
    def __init__(self, dim=512,head_num=8,layer_num=4,pad_id=8000,seq_len=96,weight_path=None,sp_path=None):
        self.dim = dim
        self.head_num = head_num
        self.layer_num = layer_num
        self.pad_id = pad_id
        self.seq_len = seq_len

        self.model = translation_model(pad_id + 1, pad_id + 1, dim=dim, head=head_num, layer_num=layer_num,
                              seq_len=seq_len)
        self.model.eval()
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.sp_en = spm.SentencePieceProcessor()
        if sp_path is not None:
            self.sp_en.Load(sp_path)


    def predict(self, text):
        text_id = self.sp_en.encode_as_ids(text)
        src = torch.LongTensor(text_id).unsqueeze(0)
        get_text = self.sp_en.decode_ids(self.model.predicter(src,1))
        return get_text


if __name__ == "__main__":
    

    model = Predicter()
    print(model.predict("ごめんなさい。早く帰らなくちゃ"))