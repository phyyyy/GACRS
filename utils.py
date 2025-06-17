from __future__ import annotations

import itertools
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

def choose_input(
    img: torch.Tensor, img_affine: torch.Tensor, p: float
) -> torch.Tensor:
    """Per-sample choose original vs. affine image with prob. p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0,1]")
    mask = torch.rand(img.size(0), 1, 1, 1, device=img.device) < p
    return torch.where(mask, img, img_affine)


def get_dynamic_p(
    epoch: int, total_epochs: int, initial_p: float = 1.0, final_p: float = 0.0
) -> float:
    """Linear decay from initial_p to final_p over `total_epochs` epochs."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if epoch >= total_epochs:
        return final_p
    step = (initial_p - final_p) / total_epochs
    return initial_p - step * epoch


def build_text_prompts(
    adj_mat: np.ndarray,
    class_list: Sequence[str],
    min_size: int,
    max_size: int,
    templates: Sequence[str],
    *,
    max_prompts: int = 10_000,
    shuffle: bool = False,
) -> List[Tuple[List[int], str]]:
    """Return [(class_indices, prompt), ...] for all fully-connected cliques."""
    if adj_mat.shape[0] != len(class_list):
        raise ValueError("adj_mat rows must equal len(class_list)")

    mask = adj_mat.astype(bool)
    idx_range = range(len(class_list))
    prompts: list[Tuple[List[int], str]] = []

    for k in range(min_size, max_size + 1):
        for combo in itertools.combinations(idx_range, k):
            if not np.all(mask[np.ix_(combo, combo)]):
                continue
            names = [class_list[i] for i in combo]
            phrase = (
                " and ".join(names)
                if len(names) == 2
                else ", ".join(names[:-1]) + f", and {names[-1]}"
            )
            for tpl in templates:
                prompts.append((list(combo), tpl.format(phrase)))
                if len(prompts) >= max_prompts:
                    return random.sample(prompts, len(prompts)) if shuffle else prompts
    return random.sample(prompts, len(prompts)) if shuffle else prompts


TRAINING_CFG = dict(
    affine_deg_min=-180,
    affine_deg_max=180,
    affine_translate=(0, 0),
    affine_scale=1.414,
    affine_shear=0,
    samples_per_cluster=10,
    num_clusters=10,
    max_text_combinations=10000,
)

