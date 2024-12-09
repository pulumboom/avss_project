import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json


class CustomDirDataset(BaseDataset):
    def __init__(
        self, name="train", target_sr=16000, dataset_path=Path("data"), *args, **kwargs
    ):
        self.name = name
        self.target_sr = target_sr
        self.dataset_path = Path(dataset_path)

        self.audio_path = self.dataset_path / "audio"
        self.mouths_path = self.dataset_path / "mouths"

        self.index_audio_path = self.dataset_path / f"{name}_index.json"

        if self.index_audio_path.exists():
            index = read_json(str(self.index_audio_path))
        else:
            index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        data_audio_mix_path = data_dict["audio_mix_path"]
        data_audio_mix = self.load_audio(data_audio_mix_path)

        if self.name == "test":
            instance_data = {
                "audio_mix": data_audio_mix,
                "audio_name": Path(data_audio_mix_path).stem,
            }
            instance_data = self.preprocess_data(instance_data)
            return instance_data

        data_audio_s1 = self.load_optional_audio(data_dict.get("audio_s1_path"))
        data_audio_s2 = self.load_optional_audio(data_dict.get("audio_s2_path"))

        data_mouth_s1 = self.load_optional_mouth(data_dict.get("mouth_s1_path"))
        data_mouth_s2 = self.load_optional_mouth(data_dict.get("mouth_s2_path"))

        instance_data = {
            "audio_mix": data_audio_mix,
            "audio_s1": data_audio_s1,
            "audio_s2": data_audio_s2,
            "mouth_s1": data_mouth_s1,
            "mouth_s2": data_mouth_s2,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def load_optional_audio(self, path):
        if path and Path(path).exists():
            return self.load_audio(path)
        return None

    def load_optional_mouth(self, path):
        if path and Path(path).exists():
            with np.load(path) as data:
                return torch.from_numpy(data["data"])
        return None

    def _create_index(self):
        index = []
        mix_path = self.audio_path / "mix"
        s1_path = self.audio_path / "s1"
        s2_path = self.audio_path / "s2"

        print(f"Creating index for CustomDirDataset ({self.name})...")
        assert mix_path.exists(), f"Mix directory not found: {mix_path}"

        for file in tqdm(os.listdir(mix_path)):
            if file.endswith((".wav", ".flac", ".mp3")):
                speaker_ids = file[:-4].split("_")
                entry = {
                    "audio_mix_path": str(mix_path / file),
                }
                if (s1_path / file).exists():
                    entry["audio_s1_path"] = str(s1_path / file)
                if (s2_path / file).exists():
                    entry["audio_s2_path"] = str(s2_path / file)
                if len(speaker_ids) >= 2:
                    entry["mouth_s1_path"] = str(
                        self.mouths_path / f"{speaker_ids[0]}.npz"
                    )
                    entry["mouth_s2_path"] = str(
                        self.mouths_path / f"{speaker_ids[1]}.npz"
                    )
                index.append(entry)

        write_json(index, self.index_audio_path)
        return index

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert "audio_mix_path" in entry, "Each item must have 'audio_mix_path'."
            if self.name != "test":
                assert (
                    "audio_s1_path" in entry or "audio_s2_path" in entry
                ), "Train/validation items must have at least one source audio path (s1 or s2)."
