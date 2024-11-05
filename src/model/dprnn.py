from torch import nn
import torch.nn.functional as F
import torch

######################################################
## Inspired by https://github.com/kaituoxu/Conv-TasNet
######################################################


class DPRNNBlock(nn.Module):

    def __init__(
            self, 
            rnn_block,
            non_linearity,
            chunk_size=64,
            n_chunks=63,
            hidden_size=128,
            n_features=512,
        ):
        super().__init__()

        if rnn_block.lower() == "rnn":
            rnn_block = nn.RNN
        elif rnn_block.lower() == "lstm":
            rnn_block = nn.LSTM
        else:
            raise Exception()

        if non_linearity is None:
            pass
        elif non_linearity.lower() == "relu":
            non_linearity = nn.ReLU
        else:
            raise Exception()

        self.part1_rnn = rnn_block(n_chunks, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.part1_lin = nn.Linear(hidden_size * 2, n_chunks)
        self.part1_act = non_linearity() if non_linearity is not None else None
        self.part1_ln = nn.LayerNorm([n_features * chunk_size, n_chunks])

        self.part2_rnn = rnn_block(chunk_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.part2_lin = nn.Linear(hidden_size * 2, chunk_size)
        self.part2_act = non_linearity() if non_linearity is not None else None
        self.part2_ln = nn.LayerNorm([n_features * n_chunks, chunk_size])


    def forward(self, input_block, **kwargs):
        batch_size, n_features, chunk_size, n_chunks = input_block.shape
        input_block = input_block.reshape(batch_size, n_features * chunk_size, n_chunks)
        part1_block, _= self.part1_rnn(input_block)
        part1_block = self.part1_lin(part1_block)
        if self.part1_act:
            part1_block = self.part1_act(part1_block)
        part1_block = self.part1_ln(part1_block)
        part1_block = part1_block + input_block

        part1_block = part1_block.reshape(batch_size, n_features, chunk_size, n_chunks)
        part1_block = part1_block.transpose(-1, -2)
        part1_block = part1_block.reshape(batch_size, n_features * n_chunks, chunk_size)

        part2_block, _ = self.part2_rnn(part1_block)
        part2_block = self.part2_lin(part2_block)
        if self.part2_act:
            part2_block = self.part2_act(part2_block)
        part2_block = self.part2_ln(part2_block)
        part2_block = part2_block + part1_block

        part2_block = part2_block.reshape(batch_size, n_features, n_chunks, chunk_size)
        part2_block = part2_block.transpose(-1, -2)
        return part2_block


class Encoder(nn.Module):
    def __init__(
            self, 
            hidden_encoder_dim, 
            kernel_size, 
            norm=None,
            sample_rate=16000,
            audio_length=2,
        ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_encoder_dim,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )

        self.norm = None
        if norm == "batch":
            self.norm = nn.BatchNorm1d(hidden_encoder_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm([hidden_encoder_dim, 2 * sample_rate * audio_length // kernel_size - 1])


    def forward(self, audio_mix, **kwargs):
        """
        Args:
            audio_mix: [batch_size, 1, audio_length * sample_rate]
        
        Return:
            audio_emb: [batch_size, hidden_encoder_dim, K], K = 2 * audio_length * sample_rate / kernel_size - 1
        """
        emb = self.conv(audio_mix)

        if self.norm:
            emb = self.norm(emb)

        return F.relu(emb)
    

class Decoder(nn.Module):
    def __init__(
            self,
            hidden_encoder_dim, 
            kernel_size,
        ):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels=hidden_encoder_dim,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2
        )

    def forward(self, audio_emb: torch.Tensor, audio_mask: torch.Tensor, **kwargs):
        """
        Args:
            audio_emb: [batch_size, hidden_encoder_dim, K]
            audio_mask: [batch_size, speakers_n, hidden_encoder_dim, K]

        Return:
            audio_src: [batch_size, speakers_n, chunk_size]
        """
        audio_s1 = audio_emb * audio_mask
        audio_s2 = audio_emb * (1 - audio_mask)

        audio_s1 = self.upsample(audio_s1)
        audio_s2 = self.upsample(audio_s2)     
        
        return audio_s1, audio_s2


class DPRNN(nn.Module):

    def __init__(
            self, 
            rnn_block="LSTM", 
            non_linearity="relu",
            pad=(25, 24), 
            chunk_size=64,
            hidden_encoder_dim=64,
            sr=16000,
            audio_length=2,
            dprnn_blocks_n=6,
        ):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        #####################################################
        ##########             Encoder            ###########
        #####################################################
        self.window_size = int(sr * audio_length // 1000)
        self.stride = self.window_size // 2
        self.encoder = Encoder(hidden_encoder_dim, self.window_size, "layer")


        #####################################################
        ##########             Decoder            ###########
        #####################################################
        self.pad = pad
        self.chunk_size = chunk_size
        self.separator = nn.Sequential(*[
            DPRNNBlock(rnn_block, non_linearity, chunk_size=chunk_size, n_features=hidden_encoder_dim)
            for _ in range(dprnn_blocks_n)
        ])

        #####################################################
        ##########             Decoder            ###########
        #####################################################
        self.decoder = Decoder(hidden_encoder_dim, self.window_size)
        

    def forward(self, audio_mix, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        audio_emb = self.encoder(audio_mix)

        padded_audio_emb = F.pad(audio_emb, self.pad)
        dprnn_input_block = padded_audio_emb.unfold(-1, self.chunk_size, self.chunk_size // 2) # [batch_size, n_features, n_chunks, chunk_size]
        dprnn_input_block = dprnn_input_block.transpose(-1, -2)
        dprnn_output_block = self.separator(dprnn_input_block)
        dprnn_output_block = dprnn_output_block.transpose(-1, -2)
        dprnn_output_block = F.fold(
            dprnn_output_block.transpose(-1, -2).reshape(-1, dprnn_output_block.size(-1), dprnn_output_block.size(-2)),
            output_size=(1, padded_audio_emb.shape[-1]),
            kernel_size=(1, self.chunk_size),
            stride=(1, self.chunk_size // 2)
        ).reshape(padded_audio_emb.shape)
        dprnn_output_block = dprnn_output_block[:, :, self.pad[0]:-self.pad[1]]
        audio_mask = F.sigmoid(dprnn_output_block)

        audio_s1, audio_s2 = self.decoder(audio_emb, audio_mask)

        batch["pred_audio_s1"] = audio_s1
        batch["pred_audio_s2"] = audio_s2

        return batch

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        encoder_parameters = sum([p.numel() for p in self.encoder.parameters()])
        separator_parameters = sum([p.numel() for p in self.separator.parameters()])
        decoder_parameters = sum([p.numel() for p in self.decoder.parameters()])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nEncoder parameters: {encoder_parameters}"
        result_info = result_info + f"\nSeparator parameters: {separator_parameters}"
        result_info = result_info + f"\nDecoder parameters: {decoder_parameters}"

        return result_info
