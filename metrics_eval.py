import warnings
import os
import torchaudio
import hydra
from hydra.utils import instantiate

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Metrics evaluation script

    Args:
        config: Hydra experiment config.
    """

    metrics = instantiate(config.metrics)

    pred_path = config.pred_path
    true_path = config.true_path

    pred_s1_path = os.path.join(pred_path, 's1')
    pred_s2_path = os.path.join(pred_path, 's2')
    true_s1_path = os.path.join(true_path, 's1')
    true_s2_path = os.path.join(true_path, 's2')

    true_files = sorted(os.listdir(true_s1_path))
    pred_files = sorted(os.listdir(pred_s1_path))

    true_files_set = set(os.path.splitext(f)[0] for f in true_files)
    pred_files_set = set(os.path.splitext(f)[0] for f in pred_files)
    common_files = true_files_set.intersection(pred_files_set)

    if not common_files:
        print("No intersection between true and predicted files.")
        return

    metric_sums = {met.name: 0.0 for met in metrics["inference"]}
    num_samples = 0

    for audio_file in sorted(common_files):
        pred_s1_file = os.path.join(pred_s1_path, audio_file + '.wav')
        pred_s2_file = os.path.join(pred_s2_path, audio_file + '.wav')

        true_s1_file = os.path.join(true_s1_path, audio_file + '.wav')
        true_s2_file = os.path.join(true_s2_path, audio_file + '.wav')

        if not (os.path.exists(pred_s1_file) and os.path.exists(pred_s2_file)
                and os.path.exists(true_s1_file) and os.path.exists(true_s2_file)):
            print(f"Skipping {audio_file}: files not found in both datasets.")
            continue

        pred_audio_s1, sr_pred_s1 = torchaudio.load(pred_s1_file)
        pred_audio_s2, sr_pred_s2 = torchaudio.load(pred_s2_file)
        true_audio_s1, sr_true_s1 = torchaudio.load(true_s1_file)
        true_audio_s2, sr_true_s2 = torchaudio.load(true_s2_file)

        if not (sr_pred_s1 == sr_pred_s2 == sr_true_s1 == sr_true_s2):
            print(f"Skipping {audio_file}: different sampling rates.")
            continue

        audio_mix = true_audio_s1 + true_audio_s2

        batch = {
            'audio_mix': audio_mix,
            'audio_s1': true_audio_s1,
            'audio_s2': true_audio_s2,
            'pred_audio_s1': pred_audio_s1,
            'pred_audio_s2': pred_audio_s2,
        }

        if config.show_all:
            print(f"\nFile: {audio_file}")
        for met in metrics["inference"]:
            metric_value = met(**batch)
            if config.show_all:
                print(f"{met.name}: {metric_value:.4f}")

            metric_sums[met.name] += metric_value

        num_samples += 1

    if num_samples == 0:
        print("No samples calculated.")
        return

    print("\nAverage Metrics:")
    for met_name, total in metric_sums.items():
        avg_value = total / num_samples
        print(f"{met_name}: {avg_value:.4f}")

if __name__ == "__main__":
    main()
