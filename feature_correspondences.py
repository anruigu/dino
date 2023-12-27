from pathlib import Path
from typing import List, Tuple
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .extractor import ViTExtractor


def find_correspondences_with_anchors(
    anchor_descriptors: np.ndarray,
    image2: Image,
    num_pairs: int = 10,
    load_size: int = 512,
    layer: int = 9,
    facet: str = "key",
    bin: bool = True,
    model_type: str = "dino_vits8",
    stride: int = 4,
) -> Tuple[List[Tuple[float, float]], Image.Image]:
    """
    Find point correspondences between the anchor image and the second image.
    :param anchor_descriptors: a numpy array of shape (num_pairs, feature_dimension) containing the descriptors of the anchor image.
    :param image2: the new image containing the object we want the correspondences for.
    Returns: a list of (y, x) coordinates of image2, corresponding to the anchor image.
    """
    # Initialize the device and extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", device)
    extractor = ViTExtractor(model_type, stride, device=device)

    # Process the second image
    image2_batch, image2_pil = extractor.preprocess_pil(image2, load_size)
    descriptors2 = extractor.extract_descriptors(
        image2_batch.to(device), layer, facet, bin
    )
    num_patches2, _ = extractor.num_patches, extractor.load_size

    # Convert anchor_descriptors to a tensor and normalize
    anchor_descriptors_tensor = torch.from_numpy(anchor_descriptors).to(device)
    anchor_descriptors_tensor = anchor_descriptors_tensor.view(
        1, 1, -1, anchor_descriptors_tensor.shape[-1]
    )

    # Calculate similarities and find best matches for each anchor descriptor
    similarities = chunk_cosine_sim(anchor_descriptors_tensor, descriptors2)[0][0]
    _, best_matches = torch.max(similarities, dim=-1)

    # Extract coordinates of the matched points in image2
    points2 = []
    descriptors2_matched = []
    for i in range(num_pairs):  # iterate over each descriptor
        match_index = best_matches[
            i
        ].item()  # find the index that best matches this descriptor
        y, x = divmod(match_index, num_patches2[1])

        # Calculate the display coordinates
        x_show = (x - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y_show = (y - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points2.append((y_show, x_show))

        # Extract the matched descriptor
        descriptors2_matched.append(descriptors2[:, :, match_index].cpu().numpy())
    descriptors2_matched = np.array(descriptors2_matched).reshape(num_pairs, -1)
    descriptors2_matched_normalized = descriptors2_matched / np.linalg.norm(
        descriptors2_matched, axis=1, keepdims=True
    )
    return points2, image2_pil, descriptors2_matched_normalized


def draw_correspondences(
    points2: List[Tuple[float, float]], image2: Image.Image
) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    num_points = len(points2)
    fig2, ax2 = plt.subplots()
    ax2.axis("off")
    ax2.imshow(image2)
    # if num_points > 15:
    #     cmap = plt.get_cmap('tab10')
    # else:
    cmap = ListedColormap(
        [
            "red",
            "yellow",
            "blue",
            "lime",
            "magenta",
            "indigo",
            "orange",
            "cyan",
            "darkgreen",
            "maroon",
            "black",
            "white",
            "chocolate",
            "gray",
            "blueviolet",
        ]
    )
    colors = np.array([cmap(x % 15) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point2, color in zip(points2, colors):
        y2, x2 = point2
        circ2_1 = plt.Circle(
            (x2, y2), radius1, facecolor=color, edgecolor="white", alpha=0.5
        )
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor="white")
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig2


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y)
    """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def process_new_image(
    anchor_descriptors,
    image2_pil: Image,
    anchor_name,
    name2,
    num_pairs=5,
    load_size=224,
    layer=9,
    facet="key",
    bin=True,
    model_type="dino_vits8",
    stride=4,
) -> np.ndarray:
    """
    Find point correspondences between the anchor image and the second image. Saves to dinos/logs.
    """
    with torch.no_grad():
        # compute point correspondences for the second image
        points2, processed_image2_pil, desc2 = find_correspondences_with_anchors(
            anchor_descriptors,
            image2_pil,
            name2,
            num_pairs,
            load_size,
            layer,
            facet,
            bin,
            model_type,
            stride,
        )

        comb_name = anchor_name + "_" + name2
        points2_array = np.array([(x, y) for y, x in points2])

        curr_save_dir = Path("dino/logs/") / comb_name
        curr_save_dir.mkdir(parents=True, exist_ok=True)

        # drawing and saving correspondences for the second image
        fig2 = draw_correspondences(points2, processed_image2_pil)
        fig2.savefig(
            curr_save_dir / f"{comb_name}_corresp.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close("all")

    return points2_array, desc2
