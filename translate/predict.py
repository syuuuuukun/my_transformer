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
    def __init__(self, dim=512,head_num=8,layer_num=4,pad_id=8000,seq_len=96,pretraiend=True):
        self.dim = dim
        self.head_num = head_num
        self.layer_num = layer_num
        self.pad_id = pad_id
        self.seq_len = seq_len

        self.model = translation_model(pad_id + 1, pad_id + 1, dim=dim, head=head_num, layer_num=layer_num,
                              seq_len=seq_len)
        self.model.eval()
        self.model.load_state_dict(torch.load("../main/data/model_result_0033001iteration.pt"))

        self.sp_en = spm.SentencePieceProcessor()
        self.sp_en.Load("../main/data/en_ja_8000.model")


    def predict(self, text):
        text_id = self.sp_en.encode_as_ids(text)
        src = torch.LongTensor(text_id).unsqueeze(0)
        get_text = self.sp_en.decode_ids(self.model.predicter(src,1))
        return get_text


if __name__ == "__main__":
    model = Predicter()
    print(model.predict("ごめんなさい。早く帰らなくちゃ"))