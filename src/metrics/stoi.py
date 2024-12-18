import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class STOI(BaseMetric):
    def __init__(self, device, fs=16000, extended=False, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs, extended).to(device)

    def __call__(
            self, 
            audio_s1: torch.Tensor, 
            audio_s2: torch.Tensor,
            pred_audio_s1: torch.Tensor,
            pred_audio_s2: torch.Tensor, 
            **kwargs
        ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        return (
            self.metric(pred_audio_s1, audio_s1) + 
            self.metric(pred_audio_s2, audio_s2)
        ) / 2