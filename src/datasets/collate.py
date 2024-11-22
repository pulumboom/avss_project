import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audio_mix = []
    audio_s1 = []
    audio_s2 = []
    mouth_s1 = []
    mouth_s2 = []

    for item in dataset_items:
        audio_mix.append(item["audio_mix"].unsqueeze(0))
        audio_s1.append(item["audio_s1"].unsqueeze(0))
        audio_s2.append(item["audio_s2"].unsqueeze(0))
        mouth_s1.append(item["mouth_s1"].unsqueeze(0))
        mouth_s2.append(item["mouth_s2"].unsqueeze(0))

    return {
        "audio_mix": torch.vstack(audio_mix),
        "audio_s1": torch.vstack(audio_s1),
        "audio_s2": torch.vstack(audio_s2),
        "mouth_s1": torch.vstack(mouth_s1),
        "mouth_s2": torch.vstack(mouth_s2),
    }


def collate_fn_test(dataset_items: list[dict]):
    """
    Collate and pad fields in the train dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audio_mix = []

    for item in dataset_items:
        audio_mix.append(item["audio_mix"].unsqueeze(0))

    return {
        "audio_mix": torch.vstack(audio_mix),
        "audio_name": [item["audio_name"] for item in dataset_items],
    }