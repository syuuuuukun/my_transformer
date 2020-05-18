import codecs
import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm

from model import *
from utils import *

if __name__ == "__main__":
    warm_up = 4000
    model_d = 512
    head_num = 8
    layer_num = 6
    pad_id = 8000


    sp_en = spm.SentencePieceProcessor()
    sp_ja = spm.SentencePieceProcessor()

    sp_en.Load("./en_text8000.model")
    sp_ja.Load("./ja_text8000.model")

    #     ja_path = "../data/trans_data2/ja_text.txt"
    ja_path = "./train50000jatext.txt"
    #     en_path = "../data/trans_data2/en_text.txt"
    en_path = "./train50000entext.txt"

    with codecs.open(ja_path, "r", "utf-8") as r:
        line = r.readlines()
        line = list(map(lambda x: (re.sub("\n", "", x)), line))

    train_data_ja = np.array(line)

    with codecs.open(en_path, "r", "utf-8") as r:
        line = r.readlines()
        line = list(map(lambda x: re.sub("\n", "", x), line))

    train_data_en = np.array(line)

    ja_path = "./test1000.ja"
    en_path = "./test1000.en"

    with codecs.open(ja_path, "r", "utf-8") as r:
        line = r.readlines()
        test_ja = list(map(lambda x: "".join(re.sub("\n", "", x).split()), line))

    with codecs.open(en_path, "r", "utf-8") as r:
        line = r.readlines()
        test_en = list(map(lambda x: re.sub("\n", "", x), line))

    ja_dic = make_dict(train_data_ja)
    en_dic = make_dict(train_data_en)

    ja_vocab = sp_ja.get_piece_size()
    en_vocab = sp_en.get_piece_size()

    dataset = MyDataset(train_data_ja, train_data_en, sp_ja, sp_en, ja_dic, en_dic)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    testset = MyDataset(test_ja, test_en, sp_ja, sp_en, ja_dic, en_dic)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    device = "cuda:1"

    # model = translation_model(transformer,ja_vocab+1,en_vocab+3,dim=dim).to(device)
    model = translation_model(ja_vocab + 1, en_vocab + 1, dim=model_d, head=head_num, layer_num=layer_num).to(device)
    lossfc = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters())

    count_param(model)
    model.apply(weights_init)

    losses = []
    for ite in range(1, 100000):
        optimizer.param_groups[0]['lr'] = (512 ** -0.5) * min(ite ** -0.5, ite * (warm_up ** -1.5))
        data = iter(dataloader).next()
        inputs, input_target, target = tuple([b.long().to(device) for b in data])
        out = model(inputs, input_target)
        out = out.view(-1, pad_id+1)
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
                for data in testloader:
                    inputs, input_target, target = tuple([b.long().to(device) for b in data])
                    inp_token = inputs[1]
                    non_pad_ids = inp_token != 8000
                    inp_token = to_list(inp_token[non_pad_ids])
                    inp_text = sp_ja.decode_ids(inp_token)

                    tgt_token = target[1]
                    non_pad_ids = tgt_token != 8000
                    tgt_token = to_list(tgt_token[non_pad_ids])
                    tgt_text = sp_en.decode_ids(tgt_token)

                    predict = model.predicter(inputs[1:2])
                    prd_token = predict[0]
                    non_pad_ids = prd_token != 8000
                    prd_token = to_list(prd_token[non_pad_ids])
                    if prd_token.count(2) >= 1:
                        prd_text = sp_en.decode_ids(prd_token[:prd_token.index(2)])
                    else:
                        prd_text = "none"
                    print(f"input: {inp_text}\ntarget: {tgt_text}\npredict: {prd_text}")
                    break
            model.train()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()},
        "./model_result.pt")
