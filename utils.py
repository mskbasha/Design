from typing import List, Tuple

import torch


def pad(
    batch: List[torch.Tensor], dim: int = 512, max_len: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences in the batch to have the same length and create attention masks.

    Args:
        batch (List[torch.Tensor]): List of input tensors (sequences).
        dim (int, optional): Dimension of padding. Defaults to 512.
        max_len (int, optional): Maximum length of sequences. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded batch tensor and attention mask tensor.
    """
    if max_len is None:
        max_len = max([len(x) for x in batch])
    attention_mask = []
    for i in range(len(batch)):
        ones = [0] * len(batch[i])
        pad_length = max_len - len(batch[i])
        if pad_length == 0:
            attention_mask.append(ones)
            continue
        zeros = [1] * pad_length
        padding = torch.zeros(pad_length, dim)
        batch[i] = torch.cat([batch[i], padding])
        attention_mask.append(ones + zeros)
    return torch.stack(batch), torch.tensor(attention_mask)


def filter_and_pad_outputs(outputs: List[List[int]]) -> torch.Tensor:
    """
    Filter and pad outputs with special tokens.

    Args:
        outputs (List[List[int]]): List of output sequences.

    Returns:
        torch.Tensor: Padded output tensor.
    """
    max_len = max([len(x) for x in outputs])
    for sample in outputs:
        for _ in range(max_len - len(sample)):
            sample.append(0)
        past = 0
        for ind, val in enumerate(sample):
            if val == 1 and past == 0:
                sample[ind] = 2
            past = val
    return torch.tensor(outputs)
