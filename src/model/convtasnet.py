from torch import nn
from torch.nn import Sequential

from src.model.dprnn import Encoder


class ConvTasNet(nn.Module):
    """
    ConvTasNet architecture.
    """

    def __init__(self, n_feats, n_src, n_repeats, n_blocks, n_hidden, kernel_size):
        """
        Args:
            n_feats (int): number of input features.
            n_src (int): number of sources.
            n_repeats (int): number of repeats in the encoder.
            n_blocks (int): number of blocks in the separator.
            n_hidden (int): number of hidden features in the separator.
            kernel_size (int): kernel size of the conv1d layers
        """
        super().__init__()

        self.encoder = Encoder(n_feats, n_repeats, kernel_size)
        self.separator = Separator(n_blocks, n_hidden, kernel_size)
        self.decoder = Decoder(n_feats, n_src, n_repeats, kernel_size)


    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
