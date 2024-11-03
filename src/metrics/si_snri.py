import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SI_SNRi(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ScaleInvariantSignalNoiseRatio().to(device)

    def __call__(
            self, 
            audio_mix: torch.Tensor,
            audio_s1: torch.Tensor, 
            audio_s2: torch.Tensor,
            pred_audio_s1: torch.Tensor,
            pred_audio_s2: torch.Tensor, 
            **kwargs
        ):
        perm1 = (
            (self.metric(pred_audio_s1, audio_s1) - self.metric(audio_mix, audio_s1)) + 
            (self.metric(pred_audio_s2, audio_s2) - self.metric(audio_mix, audio_s2))
        ) / 2

        perm2 = (
            (self.metric(pred_audio_s2, audio_s1) - self.metric(audio_mix, audio_s1)) + 
            (self.metric(pred_audio_s1, audio_s2) - self.metric(audio_mix, audio_s2))
        ) / 2

        return max(perm1, perm2)