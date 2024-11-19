import torch.nn.functional as F
from torch import nn
import torch

######################################################
## Inspired by https://arxiv.org/pdf/1809.07454
######################################################


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        # x shape: [B, num_channels, T]
        mean = x.mean(dim=(2), keepdim=True)  # [B, num_channels, 1]
        std = x.std(dim=(2), keepdim=True)    # [B, num_channels, 1]

        return self.gamma * (x - mean) / (std + 1e-5) + self.beta


class Conv1DBlock(nn.Module):
    """
    1-D Conv Block
    """

    def __init__(self, in_channels, bottleneck_channels, kernel_size, dilation):
        """
        Args:
            in_channels (int): Number of input channels.
            bottleneck_channels (int): Bottleneck channel size.
            kernel_size (int): Kernel size for depthwise convolution.
            dilation (int): Dilation factor.
        """
        super().__init__()
        self.bottleneck_conv = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1
        )
        self.depthwise_conv = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size,
            groups=bottleneck_channels,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.residual_conv = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)

        self.prelu = nn.PReLU()
        self.norm = GlobalLayerNorm(bottleneck_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape [B, in_channels, T].

        Returns:
            Tuple[Tensor, Tensor]: Residual and skip outputs.
        """
        out = self.bottleneck_conv(x)  # [B, bottleneck_channels, T]

        out = self.norm(out)
        out = self.prelu(out)

        out = self.depthwise_conv(out)  # [B, bottleneck_channels, T]

        out = self.norm(out)
        out = self.prelu(out)

        residual = self.residual_conv(out)  # [B, in_channels, T]
        skip = self.skip_conv(out)  # [B, in_channels, T]

        return x + residual, skip


class Encoder(nn.Module):
    """
    Encoder for ConvTasNet: Converts waveform into embedding.
    """

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        """
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Kernel size for convolution.
            stride (int): Stride for convolution.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T].

        Returns:
            Tensor: Encoded features [B, N, T'].
        """
        return F.relu(self.conv(x))


class Separator(nn.Module):
    """
    Separator for ConvTasNet: Generates masks for each source.
    """

    def __init__(
            self, input_channels, bottleneck_channels, n_blocks, kernel_size, n_sources
    ):
        """
        Args:
            input_channels (int): Number of input channels.
            bottleneck_channels (int): Bottleneck size for Conv1D blocks.
            n_blocks (int): Number of Conv1D blocks in the separator.
            kernel_size (int): Kernel size for depthwise convolutions.
            n_sources (int): Number of sources to separate.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(n_blocks):
            self.blocks.append(
                Conv1DBlock(input_channels, bottleneck_channels, kernel_size, dilation)
            )
            dilation *= 2

        self.mask_conv = nn.Conv1d(
            input_channels, n_sources * input_channels, kernel_size=1
        )

        self.output_conv = nn.Conv1d(input_channels, input_channels, kernel_size=1)

        self.conv = nn.Conv1d(input_channels, input_channels, kernel_size=1)

        self.norm = GlobalLayerNorm(input_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Encoded features [B, N, T'].

        Returns:
            Tensor: Masks for each source [B, n_sources, N, T'].
        """

        x = self.norm(x)
        x = self.conv(x)

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        skip_sum = sum(skip_connections)
        skip_sum = self.prelu(skip_sum)
        skip_sum = self.output_conv(skip_sum)

        masks = self.mask_conv(skip_sum)  # [B, n_sources * N, T']
        masks = masks.view(
            masks.size(0), -1, x.size(1), x.size(2)
        )  # [B, n_sources, N, T']
        return F.sigmoid(masks)


class Decoder(nn.Module):
    """
    Decoder for ConvTasNet: Converts embeddings to waveforms.
    """

    def __init__(self, input_channels, kernel_size, stride):
        """
        Args:
            input_channels (int): Number of input channels.
            kernel_size (int): Kernel size for transposed convolution.
            stride (int): Stride for transposed convolution.
        """
        super().__init__()
        self.deconv = nn.ConvTranspose1d(input_channels, 1, kernel_size, stride=stride)

    def forward(self, x):
        """
        Args:
            x (Tensor): Separated features [B, N, T'].

        Returns:
            Tensor: Waveform [B, 1, T].
        """
        return self.deconv(x)


class ConvTasNet(nn.Module):
    """
    Full ConvTasNet model.
    """

    def __init__(self, encoder_params, separator_params, decoder_params):
        super().__init__()
        self.encoder = Encoder(**encoder_params)
        self.separator = Separator(**separator_params)
        self.decoder = Decoder(**decoder_params)

    def forward(self, audio_mix, **batch):
        """
        Args:
            mixture (Tensor): Input waveform [B, 1, T].

        Returns:
            Tensor: Separated waveforms [B, n_sources, T].
        """
        encoded = self.encoder(audio_mix)  # [B, N, T']
        masks = self.separator(encoded)  # [B, n_sources, N, T']
        separated = masks * encoded.unsqueeze(1)

        audio_s1 = self.decoder(separated[:, 0])
        audio_s2 = self.decoder(separated[:, 1])

        batch["pred_audio_s1"] = audio_s1 / audio_s1.max(-1, keepdim=True).values * 0.9
        batch["pred_audio_s2"] = audio_s2 / audio_s2.max(-1, keepdim=True).values * 0.9

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
