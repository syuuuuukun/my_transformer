import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim=512, head=8, drop_rate=0.1, ffn_dim=2048):
        super(Encoder, self).__init__()
        self.dim = dim
        self.head = head
        self.h_dim = dim // head
        self.activation = nn.ReLU(True)

        self.encoder_attention = MultiHeadAttention(dim, head, ffn_dim, drop_rate)
        self.drop1 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(dim)

        self.FFN = nn.Sequential(nn.Linear(dim, ffn_dim),
                                 self.activation,
                                 nn.Dropout(drop_rate),
                                 nn.Linear(ffn_dim, dim))

        self.drop2 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, att_mask):
        """
        att_mask: 0 or -infのpadding
        att_mask2: 0 or 1のpadding
        """
        att_out, att1 = self.encoder_attention(x, x, x, att_mask)

        ##add norm
        att_out = self.norm1(x + self.drop1(att_out))

        ## FeedForward
        ffn_out = self.FFN(att_out)
        output = self.norm2(att_out + self.drop2(ffn_out))

        return output, att1


class Decoder(nn.Module):
    def __init__(self, dim=512, head=8, drop_rate=0.1, ffn_dim=2048):
        super(Decoder, self).__init__()
        self.dim = dim
        self.head = head
        self.h_dim = dim // head
        self.activation = nn.ReLU(True)

        self.tgt_MHA = MultiHeadAttention(dim, head, ffn_dim, drop_rate)
        self.enc_dec_MHA = MultiHeadAttention(dim, head, ffn_dim, drop_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(drop_rate)

        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(drop_rate)

        self.FFN = nn.Sequential(nn.Linear(dim, ffn_dim),
                                 self.activation,
                                 nn.Dropout(drop_rate),
                                 nn.Linear(ffn_dim, dim))

        self.drop3 = nn.Dropout(drop_rate)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, enc_out, tgt, enc_att_mask=None, tgt_att_mask=None):
        tgt_out, att1 = self.tgt_MHA(tgt, tgt, tgt, tgt_att_mask)
        tgt_out = self.norm1(tgt + self.drop1(tgt_out))

        enc_out, att2 = self.enc_dec_MHA(tgt_out, enc_out, enc_out, enc_att_mask)
        enc_out = self.norm2(tgt_out + self.drop2(enc_out))

        ## FeedForward
        ffn_out = self.FFN(enc_out)
        output = self.norm3(enc_out + self.drop3(ffn_out))
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=512, head=8, ffn_dim=2048, drop_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.head = head
        self.h_dim = dim // head
        self.activation = nn.ReLU(True)
        self.scale = dim ** 0.5

        self.k_enc = nn.Linear(self.dim, self.dim, bias=False)
        self.q_enc = nn.Linear(self.dim, self.dim, bias=False)
        self.v_enc = nn.Linear(self.dim, self.dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)
        self.att_drop = nn.Dropout(drop_rate)

    def forward(self, query, key, value, att_mask):
        ## encoder-decoder-attention
        b = query.shape[0]

        ## key query value
        query = self.q_enc(query)
        key = self.k_enc(key)
        value = self.v_enc(value)

        q = query.view(b, -1, self.head, self.h_dim).permute(0, 2, 1, 3)
        k = key.view(b, -1, self.head, self.h_dim).permute(0, 2, 1, 3)
        v = value.view(b, -1, self.head, self.h_dim).permute(0, 2, 1, 3)

        ##attention
        att_weight = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        att_weight = torch.add(att_weight, att_mask)
        att_weight = torch.softmax(att_weight, dim=-1)
        att_weight = self.att_drop(att_weight)

        enc_out = torch.matmul(att_weight, v)
        enc_out = enc_out.permute(0, 2, 1, 3).contiguous()
        enc_out = enc_out.view(b, -1, self.dim)
        enc_out = self.out(enc_out)

        return enc_out, att_weight


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, ):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class translation_model(nn.Module):
    def __init__(self, ja_vocab, en_vocab, dim=512, head=8, layer_num=4, seq_len=64):
        super(translation_model, self).__init__()
        self.layer_num = layer_num
        self.head = head
        self.dim = dim
        self.seq_len = seq_len

        self.ja_embedding = nn.Embedding(ja_vocab, dim, padding_idx=-1)
        self.en_embedding = self.ja_embedding

        self.ja_pe = PositionalEncoding(dim, 0.1, seq_len)
        self.en_pe = PositionalEncoding(dim, 0.1, seq_len)

        encoder = Encoder(dim=dim, head=head, drop_rate=0.1, ffn_dim=2048)
        self.encoders = nn.ModuleList(encoder for i in range(layer_num))

        decoder = Decoder(dim=dim, head=head, drop_rate=0.1, ffn_dim=2048)
        self.decoders = nn.ModuleList(decoder for i in range(layer_num))

        self.apply(weights_init)

        ## sharing_embedding
        self.fc1 = nn.Linear(dim, en_vocab, bias=False)
        self.fc1.weight = self.en_embedding.weight

        self.ja_vocab = ja_vocab

    def forward(self, src, tgt, src_pad_id=8000, tgt_pad_id=8000):

        pos_id = torch.LongTensor([i for i in range(src.shape[1])]).to(src.device).unsqueeze(0)

        src_masks, tgt_masks, memory_masks = generate_attention_mask(src,
                                                                     tgt,
                                                                     src_pad_id,
                                                                     tgt_pad_id)

        src_emb = self.ja_embedding(src) * (self.dim ** 0.5)
        src_emb = self.ja_pe(src_emb)

        tgt_emb = self.en_embedding(tgt) * (self.dim ** 0.5)
        tgt_emb = self.en_pe(tgt_emb)

        for encoder in self.encoders:
            src_emb, att_weight = encoder(src_emb, src_masks)

        for decoder in self.decoders:
            tgt_emb = decoder(src_emb, tgt_emb, src_masks, tgt_masks)
        out = self.fc1(tgt_emb)
        return out

    def predicter(self, src, bos_id=1, seq_len=64, src_pad_id=8000, tgt_pad_id=8000):
        src = src.long()
        src_masks, _, _ = generate_attention_mask(src, src, src_pad_id, tgt_pad_id)
        ##入力文のエンコード
        src_emb = self.ja_embedding(src) * (self.dim ** 0.5)
        src_emb = src_emb + self.en_pe.pe[:, :src_emb.shape[1], :]

        for encoder in self.encoders:
            src_emb, att_weight = encoder(src_emb, src_masks)

            ##出力文のデコード <bos>から
        predict_tokens = [bos_id]
        for i in range(seq_len - 1):
            input_target = torch.LongTensor(predict_tokens).unsqueeze(0).to(src.device)
            src_masks, tgt_masks, memory_masks = generate_attention_mask(src, input_target, src_pad_id, tgt_pad_id)
            tgt_emb = self.en_embedding(input_target) * (self.dim ** 0.5)
            tgt_emb = tgt_emb + self.en_pe.pe[:, :input_target.shape[1], :]

            for decoder in self.decoders:
                tgt_emb = decoder(src_emb, tgt_emb, memory_masks, tgt_masks)
            out = self.fc1(tgt_emb)
            out = out.argmax(dim=-1)
            out_id = out[0, i].item()
            predict_tokens += [out_id]
            if out_id == 2:
                break
        return predict_tokens