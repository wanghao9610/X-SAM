import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils


def decode_mask(segmentation, height, width):
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            segmentation = mask_utils.frPyObjects(segmentation, *segmentation["size"])
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
        mask = mask_utils.decode(segmentation).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], dict):
        for seg in segmentation:
            mask = mask_utils.decode(seg).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    elif isinstance(segmentation[0], list):
        for seg in segmentation:
            rles = mask_utils.frPyObjects([seg], height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            binary_mask = np.maximum(binary_mask, mask.squeeze())
    else:
        raise ValueError(f"Invalid segmentation type: {type(segmentation)}")

    return binary_mask


def encode_mask(mask, encoding="ascii"):
    assert set(np.unique(mask)).issubset({0, 1})
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode(encoding)
    return rle


def calculate_iou(output, target, C, ignore_index=255):
    """
    Calculate intersection and union for numpy arrays
    Args:
        output: numpy array, prediction result
        target: numpy array, ground truth
        C: int, number of classes
        ignore_index: int, label value to ignore
    Returns:
        area_intersection: numpy array, intersection area for each class
        area_union: numpy array, union area for each class
        area_target: numpy array, target area for each class
    """
    # Ensure input dimensions are valid
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape

    # Flatten arrays
    output = output.flatten()
    target = target.flatten()

    # Handle ignored regions
    output[target == ignore_index] = ignore_index

    # Calculate intersection
    mask = output == target
    intersection = output[mask]

    # Ensure correct number of bins
    valid_area = lambda x: (x != ignore_index) & (x < C)
    area_intersection = np.bincount(intersection[valid_area(intersection)], minlength=C)
    area_output = np.bincount(output[valid_area(output)], minlength=C)
    area_target = np.bincount(target[valid_area(target)], minlength=C)

    # Calculate union
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


# Copied from transformers.models.detr.image_processing_detr.binary_mask_to_rle
def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    if isinstance(segmentation, np.ndarray):
        segmentation = torch.from_numpy(segmentation)
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = mask_utils.frPyObjects(polygons, height, width)
        mask = mask_utils.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def convert_mask_to_coco_poly(mask):
    """
    Convert a binary mask to COCO polygon format.

    Args:
    mask: torch.Tensor or numpy.ndarray of shape (H, W) with values 0 or 1

    Returns:
    list: List of polygons in COCO format
    """
    # Ensure the mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentations = []
    for contour in contours:
        # Convert contour to polygon format
        contour = contour.flatten().tolist()
        # Only add valid polygons (at least 3 points, i.e., 6 coordinates)
        if len(contour) >= 6:
            segmentations.append(contour)

    return segmentations
