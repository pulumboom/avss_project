import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential


class Conv1DBlock(nn.Module):
    """
    1-D Conv Block for ConvTasNet with residual and skip connections.
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
        self.bottleneck_conv = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            bottleneck_channels, bottleneck_channels, kernel_size,
            groups=bottleneck_channels, dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2  # Maintain sequence length
        )
        self.residual_conv = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)

        self.prelu = nn.PReLU()
        self.norm = nn.BatchNorm1d(bottleneck_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape [B, in_channels, T].

        Returns:
            Tuple[Tensor, Tensor]: Residual and skip outputs.
        """
        out = self.bottleneck_conv(x)  # [B, bottleneck_channels, T]
        out = self.prelu(out)
        out = self.norm(out)
        out = self.depthwise_conv(out)  # [B, bottleneck_channels, T]
        out = self.prelu(out)
        out = self.norm(out)
        residual = self.residual_conv(out)  # [B, in_channels, T]
        skip = self.skip_conv(out)  # [B, in_channels, T]
        return x + residual, skip


class Encoder(nn.Module):
    """
    Encoder for ConvTasNet: Converts waveform into feature representation.
    """

    def __init__(self, input_channels, output_channels, kernel_size, stride):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 1 for mono audio).
            output_channels (int): Number of output channels (N in the paper).
            kernel_size (int): Kernel size for convolution.
            stride (int): Stride for convolution.
        """
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T].

        Returns:
            Tensor: Encoded features [B, N, T'].
        """
        return self.conv(x)  # ReLU is applied to the encoded features


class Separator(nn.Module):
    """
    Separator for ConvTasNet: Generates masks for each source.
    """

    def __init__(self, input_channels, bottleneck_channels, n_blocks, kernel_size, n_sources):
        """
        Args:
            input_channels (int): Number of input channels (N from Encoder).
            bottleneck_channels (int): Bottleneck size for Conv1D blocks.
            n_blocks (int): Number of Conv1D blocks in the separator.
            kernel_size (int): Kernel size for depthwise convolutions.
            n_sources (int): Number of sources to separate.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(n_blocks):
            self.blocks.append(Conv1DBlock(input_channels, bottleneck_channels, kernel_size, dilation))
            dilation *= 2  # Dilation increases exponentially

        self.mask_conv = nn.Conv1d(input_channels, n_sources * input_channels, kernel_size=1)

        self.norm = nn.BatchNorm1d(input_channels)
        self.conv = nn.Conv1d(input_channels, input_channels, kernel_size=1)
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

        skip_sum = sum(skip_connections)  # Aggregate skip connections
        masks = self.prelu(skip_sum)
        masks = self.mask_conv(masks)  # [B, n_sources * N, T']
        masks = masks.view(masks.size(0), -1, x.size(1), x.size(2))  # [B, n_sources, N, T']
        return F.softmax(masks, dim=1)  # Softmax over sources


class Decoder(nn.Module):
    """
    Decoder for ConvTasNet: Converts separated features back to waveforms.
    """

    def __init__(self, input_channels, kernel_size, stride):
        """
        Args:
            input_channels (int): Number of input channels (N from Encoder).
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
            audio_mix (Tensor): Input waveform [B, 1, T].

        Returns:
            Dict: Separated waveforms with keys "pred_audio_s1" and "pred_audio_s2".
        """
        # Параметры сегментации
        L = self.encoder.conv.kernel_size[0]  # Длина окна (из kernel_size Encoder)
        stride = self.encoder.conv.stride[0]  # Шаг между окнами (из stride Encoder)

        # Параметры сигнала
        B, _, T = audio_mix.shape
        padded_length = (T - L) % stride  # Выравниваем длину для сегментации
        if padded_length > 0:
            audio_mix = F.pad(audio_mix, (0, stride - padded_length))  # Padding до ближайшего stride
        T_padded = audio_mix.shape[-1]

        # 1. Разделение сигнала на перекрывающиеся отрезки
        separated_audio_s1 = torch.zeros(B, T_padded, device=audio_mix.device)
        separated_audio_s2 = torch.zeros(B, T_padded, device=audio_mix.device)
        overlap_count = torch.zeros(T_padded, device=audio_mix.device)  # Счётчик перекрытий для нормализации

        for start in range(0, T_padded - L + 1, stride):
            print(f"start: {start} of {T_padded - L + 1}")
            # Извлекаем текущий отрезок
            segment = audio_mix[:, :, start:start + L]  # [B, 1, L]

            # 2. Обработка отрезка
            encoded = self.encoder(segment)  # [B, N, T']
            masks = self.separator(encoded)  # [B, n_sources, N, T']
            separated = masks * encoded.unsqueeze(1)  # [B, n_sources, N, T']

            audio_s1 = self.decoder(separated[:, 0])  # [B, 1, L]
            audio_s2 = self.decoder(separated[:, 1])  # [B, 1, L]

            # 3. Накладываем результат с учётом перекрытия
            separated_audio_s1[:, start:start + L] += audio_s1.squeeze(1)  # [B, T_padded]
            separated_audio_s2[:, start:start + L] += audio_s2.squeeze(1)  # [B, T_padded]
            overlap_count[start:start + L] += 1  # [T_padded]

        # 4. Нормализация перекрытий
        separated_audio_s1 /= overlap_count.unsqueeze(0)  # [B, T_padded]
        separated_audio_s2 /= overlap_count.unsqueeze(0)  # [B, T_padded]

        # 5. Обрезка до исходной длины
        separated_audio_s1 = separated_audio_s1[:, :T]
        separated_audio_s2 = separated_audio_s2[:, :T]

        # Нормализация
        batch["pred_audio_s1"] = separated_audio_s1 / separated_audio_s1.max(-1, keepdim=True).values * 0.9
        batch["pred_audio_s2"] = separated_audio_s2 / separated_audio_s2.max(-1, keepdim=True).values * 0.9

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
