import torch.nn as nn
from embedding import Embedding
from positional_encoding import PositionalEncoding
from encoder import EncoderLayer
from encoder import Encoder
from decoder import DecoderLayer
from decoder import Decoder
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from projection import Projection


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        enc_embed: Embedding,
        dec_embed: Embedding,
        pos_encod: PositionalEncoding,
        projection: Projection,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed
        self.pos_encod = pos_encod
        self.projection = projection

    def encode(self, enc_input, enc_self_attn_mask):
        # (batch, src_len) -> (batch, src_len, d_model)
        enc_input = self.enc_embed(enc_input)
        enc_input = self.pos_encod(enc_input)
        return self.encoder(enc_input, enc_self_attn_mask)

    def decode(
        self, enc_output, dec_input, dec_self_attn_mask, dec_cross_attn_mask
    ):
        # (batch, tgt_len) -> (batch, tgt_len, d_model)
        dec_input = self.dec_embed(dec_input)
        dec_input = self.pos_encod(dec_input)
        return self.decoder(
            dec_input, enc_output, dec_self_attn_mask, dec_cross_attn_mask
        )

    def project(self, x):
        # (batch, len, d_model) -> (batch, len, vocab_size) log-probs
        return self.projection(x)

    def forward(
        self,
        enc_input,
        dec_input,
        enc_self_attn_mask,
        dec_self_attn_mask,
        dec_cross_attn_mask,
    ):
        enc_output = self.encode(enc_input, enc_self_attn_mask)
        output = self.decode(
            enc_output, dec_input, dec_self_attn_mask, dec_cross_attn_mask
        )
        return self.project(output)


def build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    N=6,
    d_model=512,
    d_ff=2048,
    h=8,
    seq_len=5000,
    dropout=0.1,
):
    # Encoder
    enc_self_attention = MultiHeadAttention(h, d_model, dropout)
    enc_ffn = FeedForwardNetwork(d_model, d_ff, dropout)
    encoderLayer = EncoderLayer(enc_self_attention, enc_ffn, d_model, dropout)
    encoder = Encoder(encoderLayer, N)
    enc_embed = Embedding(d_model, src_vocab_size)

    # Decoder
    dec_self_attention = MultiHeadAttention(h, d_model, dropout)
    dec_cross_attention = MultiHeadAttention(h, d_model, dropout)
    dec_ffn = FeedForwardNetwork(d_model, d_ff, dropout)
    decoderLayer = DecoderLayer(
        dec_self_attention, dec_cross_attention, dec_ffn, d_model, dropout
    )
    decoder = Decoder(decoderLayer, N)
    dec_embed = Embedding(d_model, tgt_vocab_size)

    # Positional Encoding is reused for both encoder and decoder
    pos_encod = PositionalEncoding(seq_len, d_model, dropout)
    projection = Projection(d_model, tgt_vocab_size)

    model = Transformer(
        encoder,
        decoder,
        enc_embed,
        dec_embed,
        pos_encod,
        projection,
    )

    # Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
