import torch.nn as nn
from embedding import Embedding
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder
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

    def forward(
        self,
        enc_input,
        dec_input,
        enc_self_attn_mask,
        dec_self_attn_mask,
        dec_cross_attn_mask,
    ):
        # Encoder
        enc_input = self.enc_embed(enc_input)
        enc_input = self.pos_encod(enc_input)

        enc_output = self.encoder(
            enc_input,
            enc_self_attn_mask,
        )

        # Decoder
        dec_input = self.dec_embed(dec_input)
        dec_input = self.pos_encod(dec_input)

        output = self.decoder(
            dec_input,
            enc_output,
            dec_self_attn_mask,
            dec_cross_attn_mask,
        )

        # Linear + Softmax
        return self.projection(output)
