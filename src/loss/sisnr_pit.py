import torch
from torch import nn

EPS = 1e-8

class SiSNR_PIT(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
            self, 
            audio_s1: torch.Tensor, 
            audio_s2: torch.Tensor, 
            pred_audio_s1: torch.Tensor,
            pred_audio_s2: torch.Tensor,
            **batch
        ):
        """
        Args:
            audio_s1: [batch_size, 1, audio_length * sample_rate]
            audio_s2: [batch_size, 1, audio_length * sample_rate]
            pred_audio_s1: [batch_size, 1, audio_length * sample_rate]
            pred_audio_s1: [batch_size, 1, audio_length * sample_rate]
        """
        audio_s1 = audio_s1 - audio_s1.mean(dim=-1, keepdim=True)
        audio_s2 = audio_s2 - audio_s2.mean(dim=-1, keepdim=True)
        pred_audio_s1 = pred_audio_s1 - pred_audio_s1.mean(dim=-1, keepdim=True)
        pred_audio_s2 = pred_audio_s2 - pred_audio_s2.mean(dim=-1, keepdim=True)

        #########################################################
        ########## First Permutation (1 -> 1, 2 -> 2) ###########
        #########################################################

        audio_s1_norm = audio_s1 @ audio_s1.transpose(-2, -1)
        audio_s2_norm = audio_s2 @ audio_s2.transpose(-2, -1)

        audio_s1_target_perm1 = audio_s1 @ pred_audio_s1.transpose(-2, -1) / audio_s1_norm * audio_s1
        audio_s2_target_perm1 = audio_s2 @ pred_audio_s2.transpose(-2, -1) / audio_s2_norm * audio_s2

        noise_s1_perm1 = pred_audio_s1 - audio_s1_target_perm1
        noise_s2_perm1 = pred_audio_s2 - audio_s2_target_perm1

        si_snr_s1_perm1 = 10 * torch.log10(
            (audio_s1_target_perm1 @ audio_s1_target_perm1.transpose(-2, -1)) / 
            (noise_s1_perm1 @ noise_s1_perm1.transpose(-2, -1) + EPS) + EPS)
        si_snr_s2_perm1 = 10 * torch.log10(
            (audio_s2_target_perm1 @ audio_s2_target_perm1.transpose(-2, -1)) / 
            (noise_s2_perm1 @ noise_s2_perm1.transpose(-2, -1) + EPS) + EPS)
        si_snr_perm1 = si_snr_s1_perm1 + si_snr_s2_perm1

        #########################################################
        ######### Second Permutation (1 -> 2, 2 -> 1) ###########
        #########################################################

        audio_s1_target_perm2 = audio_s1 @ pred_audio_s2.transpose(-2, -1) / audio_s1_norm * audio_s1
        audio_s2_target_perm2 = audio_s2 @ pred_audio_s1.transpose(-2, -1) / audio_s2_norm * audio_s2

        noise_s1_perm2 = pred_audio_s2 - audio_s1_target_perm2
        noise_s2_perm2 = pred_audio_s1 - audio_s2_target_perm2

        si_snr_s1_perm2 = 10 * torch.log10(
            (audio_s1_target_perm2 @ audio_s1_target_perm2.transpose(-2, -1)) / 
            (noise_s1_perm2 @ noise_s1_perm2.transpose(-2, -1) + EPS) + EPS)
        si_snr_s2_perm2 = 10 * torch.log10(
            (audio_s2_target_perm2 @ audio_s2_target_perm2.transpose(-2, -1)) / 
            (noise_s2_perm2 @ noise_s2_perm2.transpose(-2, -1) + EPS) + EPS)
        si_snr_perm2 = si_snr_s1_perm2 + si_snr_s2_perm2

        #########################################################
        #################### Combining ##########################
        #########################################################
        si_snr = torch.hstack([si_snr_perm1, si_snr_perm2]).squeeze(-1)
        si_snr_pit = torch.max(si_snr, dim=-1)
        best_perm = si_snr_pit.indices.reshape(-1, 1, 1)

        return {
            "loss": si_snr_pit.values.sum(),
            "pred_audio_s1": torch.where(best_perm == 0, pred_audio_s1, pred_audio_s2),
            "pred_audio_s2": torch.where(best_perm == 0, pred_audio_s2, pred_audio_s1),
        }
