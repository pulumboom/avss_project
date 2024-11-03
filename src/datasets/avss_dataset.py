import os
import torch
import torchaudio
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

class AvssDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, 
        name="train", 
        target_sr=16000, 
        audio_path=ROOT_PATH / "data" / "audio", 
        mouths_path=ROOT_PATH / "data" / "mouths",
        *args, **kwargs
    ):
        """
        Args:
            name (str): partition name
        """
        self.target_sr = target_sr
        self.audio_path = audio_path
        self.mouths_path = mouths_path

        index_audio_path = audio_path / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_audio_path.exists():
            index = read_json(str(index_audio_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]

        data_audio_mix_path = data_dict["audio_mix_path"]
        data_audio_mix = self.load_audio(data_audio_mix_path)

        data_audio_s1_path = data_dict["audio_s1_path"]
        data_audio_s1 = self.load_audio(data_audio_s1_path)

        data_audio_s2_path = data_dict["audio_s2_path"]
        data_audio_s2 = self.load_audio(data_audio_s2_path)

        data_mouth_s1_path = data_dict["mouth_s1_path"]
        data_mouth_s1 = None
        if Path(data_mouth_s1_path).exists():
            with np.load(data_mouth_s1_path) as data:
                data_mouth_s1 = torch.from_numpy(data["data"])

        data_mouth_s2_path = data_dict["mouth_s2_path"]
        data_mouth_s2 = None
        if Path(data_mouth_s2_path).exists():
            with np.load(data_mouth_s2_path) as data:
                data_mouth_s2 = torch.from_numpy(data["data"])

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
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        audio_data_path = self.audio_path / name
        audio_data_path.mkdir(exist_ok=True, parents=True)

        mouths_data_path = self.mouths_path
        mouths_data_path.mkdir(exist_ok=True, parents=True)

        print(f"Creating Avss Dataset ({name})...")
        assert (audio_data_path / "mix").exists(), f"There is no mix folder in {audio_data_path}"

        for _, _, files in tqdm(os.walk(audio_data_path / "mix")):
            for file in files:
                mouth_path = file[:-4].split("_")

                index.append({
                    "audio_mix_path": str(audio_data_path / "mix" / file),
                    "audio_s1_path": str(audio_data_path / "s1" / file),
                    "audio_s2_path": str(audio_data_path / "s2" / file),
                    "mouth_s1_path": str(mouths_data_path / (mouth_path[0] + ".npz")),
                    "mouth_s2_path": str(mouths_data_path / (mouth_path[1] + ".npz"))
                })

        # write index to disk
        write_json(index, str(audio_data_path / "index.json"))

        return index
    
    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "audio_mix_path" in entry, (
                "Each dataset item should include field 'audio_mix_path'" " - path to mix audio file."
            )
            assert "audio_s1_path" in entry, (
                "Each dataset item should include field 'audio_s1_path'"
                " - object ground-truth speaker 1."
            )
            assert "audio_s2_path" in entry, (
                "Each dataset item should include field 'audio_s1_path'"
                " - object ground-truth speaker 2."
            )