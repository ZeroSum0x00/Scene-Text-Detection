import numpy as np


def get_craft_groundtruth(polygons, words, perspective_transfrom, size):
    confidence_mask = np.ones(size, dtype=np.float32)

    region_score = perspective_transfrom.generate_region(
        size,
        polygons,
        horizontal_text_bools=[True for _ in range(len(words))],
    )

    affinity_score, all_affinity_bbox = perspective_transfrom.generate_affinity(
        size,
        polygons,
        horizontal_text_bools=[True for _ in range(len(words))],
    )
    return region_score, affinity_score, confidence_mask
