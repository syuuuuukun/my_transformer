import math
import codecs
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from model2 import *
from utils import *

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    warm_up = 4000
    model_d = 512
    head_num = 4
    layer_num = 1
    pad_id_ja = 8000
    pad_id_en = 8000
    seq_len = 96

    sp_en = spm.SentencePieceProcessor()
    sp_ja = spm.SentencePieceProcessor()

    sp_en.Load("./data/en_ja_8000.model")
    sp_ja.Load("./data/en_ja_8000.model")

    ja_path = "./data/train50000jatext.txt"
    en_path = "./data/train50000entext.txt"

    with codecs.open(ja_path, "r", "utf-8") as r:
        line = r.readlines()
        line = list(map(lambda x: (re.sub("\n", "", x)), line))

    train_data_ja = np.array(line)

    with codecs.open(en_path, "r", "utf-8") as r:
        line = r.readlines()
        line = list(map(lambda x: re.sub("\n", "", x), line))

    train_data_en = np.array(line)

    ja_path = "./data/test1000.ja"
    en_path = "./data/test1000.en"

    with codecs.open(ja_path, "r", "utf-8") as r:
        line = r.readlines()
        test_ja = list(map(lambda x: "".join(re.sub("\n", "", x).split()), line))

    with codecs.open(en_path, "r", "utf-8") as r:
        line = r.readlines()
        test_en = list(map(lambda x: re.sub("\n", "", x), line))

    ja_dic = make_dict(train_data_ja)
    en_dic = make_dict(train_data_en)

    ja_vocab = sp_en.get_piece_size()
    en_vocab = sp_ja.get_piece_size()

    dataset = MyDataset(train_data_ja, train_data_en, sp_ja, sp_en, ja_dic, en_dic, seq_len, padding_id=pad_id_en)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=True)

    testset = MyDataset(test_ja, test_en, sp_ja, sp_en, ja_dic, en_dic, seq_len, padding_id=pad_id_en)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    device = "cuda:1"

    # model = translation_model(transformer,ja_vocab+1,en_vocab+3,dim=dim).to(device)
    model = translation_model(ja_vocab + 1, en_vocab + 1, dim=model_d, head=head_num, layer_num=layer_num,
                              seq_len=seq_len).to(device)
    lossfc = nn.CrossEntropyLoss(ignore_index=pad_id_en)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    count_param(model)

    losses = []
    for ite in range(1, 100000):
        optimizer.param_groups[0]['lr'] = (512 ** -0.5) * min(ite ** -0.5, ite * (warm_up ** -1.5))
        data = iter(dataloader).next()
        inputs, input_target, target = tuple([b.long().to(device) for b in data])
        out = model(inputs, input_target, src_pad_id=pad_id_ja, tgt_pad_id=pad_id_en)
        out = out.view(-1, pad_id_en + 1)
        loss = lossfc(out, target.view(-1))
        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (ite % 100) == 0:
            print(f"mean_loss_{ite}iteration: ", np.mean(losses))
            losses = []
            with torch.no_grad():
                model.eval()
                print("train_predict")
                for data in dataloader:
                    inputs, input_target, target = tuple([b.long().to(device) for b in data])
                    inp_token = inputs[1]
                    non_pad_ids = inp_token != pad_id_ja
                    inp_token = to_list(inp_token[non_pad_ids])
                    inp_text = sp_ja.decode_ids(inp_token)

                    tgt_token = target[1]
                    non_pad_ids = tgt_token != pad_id_en
                    tgt_token = to_list(tgt_token[non_pad_ids])
                    tgt_text = sp_en.decode_ids(tgt_token)

                    predict = model.predicter(inputs[1:2], bos_id=1, seq_len=inputs.shape[1], src_pad_id=pad_id_ja,
                                              tgt_pad_id=pad_id_en)
                    prd_token = np.array(predict)
                    non_pad_ids = prd_token != pad_id_en
                    prd_token = (prd_token[non_pad_ids]).tolist()
                    if prd_token.count(2) >= 1:
                        prd_text = sp_en.decode_ids(prd_token[:prd_token.index(2)])
                    else:
                        prd_text = "none"
                    print(f"input: {inp_text}\ntarget: {tgt_text}\npredict: {prd_text}")
                    break
                print("test_predict")
                for data in testloader:
                    inputs, input_target, target = tuple([b.long().to(device) for b in data])
                    inp_token = inputs[1]
                    non_pad_ids = inp_token != pad_id_ja
                    inp_token = to_list(inp_token[non_pad_ids])
                    inp_text = sp_ja.decode_ids(inp_token)

                    tgt_token = target[1]
                    non_pad_ids = tgt_token != pad_id_en
                    tgt_token = to_list(tgt_token[non_pad_ids])
                    tgt_text = sp_en.decode_ids(tgt_token)

                    predict = model.predicter(inputs[1:2], bos_id=1, seq_len=inputs.shape[1], src_pad_id=pad_id_ja,
                                              tgt_pad_id=pad_id_en)
                    prd_token = np.array(predict)
                    non_pad_ids = prd_token != pad_id_en
                    prd_token = (prd_token[non_pad_ids]).tolist()
                    if prd_token.count(2) >= 1:
                        prd_text = sp_en.decode_ids(prd_token[:prd_token.index(2)])
                    else:
                        prd_text = "none"
                    print(f"input: {inp_text}\ntarget: {tgt_text}\npredict: {prd_text}")
                    break
            model.train()
        if (ite % 1000) == 1:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()},
                f"./model_result_{ite:07}iteration.pt")
