import random

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import line


def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) between two bounding boxes"""

    # Calculate coordinates of the intersection area
    x_min_intersection = max(box1[1], box2[1])
    y_min_intersection = max(box1[0], box2[0])
    x_max_intersection = min(box1[3], box2[3])
    y_max_intersection = min(box1[2], box2[2])

    # Calculate the area of intersection
    intersection_width = max(0, x_max_intersection - x_min_intersection + 1)
    intersection_height = max(0, y_max_intersection - y_min_intersection + 1)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of each bounding box
    box1_width = box1[3] - box1[1] + 1
    box1_height = box1[2] - box1[0] + 1
    box1_area = box1_width * box1_height

    box2_width = box2[3] - box2[1] + 1
    box2_height = box2[2] - box2[0] + 1
    box2_area = box2_width * box2_height

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / float(union_area)

    return iou


def draw_circle(mask, center, radius):
    y, x = np.ogrid[: mask.shape[0], : mask.shape[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[distance <= radius] = 1


def enhance_with_circles(binary_mask, radius=5):
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = np.array(binary_mask)

    binary_mask = binary_mask.astype(np.uint8)

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    points = np.argwhere(binary_mask == 1)
    for point in points:
        draw_circle(output_mask, (point[0], point[1]), radius)
    return output_mask


def _create_valid_box(mask, prop, min_row, min_col, max_row, max_col, center_row, center_col, max_retries=1000):
    """Create a valid box for scribble with IOU < 0.5 with original box."""
    original_box = (min_row, min_col, max_row, max_col)

    for _ in range(max_retries):
        # Generate a new box with random scaling
        new_height = (max_row - min_row) * random.uniform(0.5, 1.2)
        new_width = (max_col - min_col) * random.uniform(0.5, 1.2)

        new_min_row = int(center_row - new_height / 2)
        new_min_col = int(center_col - new_width / 2)
        new_max_row = int(center_row + new_height / 2)
        new_max_col = int(center_col + new_width / 2)

        # Clip to image boundaries
        new_min_row = max(0, new_min_row)
        new_min_col = max(0, new_min_col)
        new_max_row = min(mask.shape[0], new_max_row)
        new_max_col = min(mask.shape[1], new_max_col)

        new_box = (new_min_row, new_min_col, new_max_row, new_max_col)

        # Check if the new box has reasonable overlap with original box
        if calculate_iou(new_box, original_box) >= 0.5:
            return new_box

    # If no valid box found after 1000 attempts, return None
    return None


def generate_point_vprompt(mask, props, max_retries=1000, point_radius=10):
    """Generate a point visual prompt by selecting a point within each region."""
    point_prompt = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        centroid = prop.centroid

        # Try to find a valid point within the region
        for _ in range(max_retries):
            radius = min(max_row - min_row, max_col - min_col) * 0.5
            angle = random.uniform(0, 2 * np.pi)
            offset = (
                random.uniform(0, radius) * np.cos(angle),
                random.uniform(0, radius) * np.sin(angle),
            )

            point = (int(centroid[0] + offset[0]), int(centroid[1] + offset[1]))
            point = (
                np.clip(point[0], min_row, max_row - 1),
                np.clip(point[1], min_col, max_col - 1),
            )

            if mask[point[0], point[1]] > 0:
                point_prompt[point[0], point[1]] = 1
                break

    point_prompt = enhance_with_circles(point_prompt, point_radius)

    return point_prompt


def generate_scribble_vprompt(mask, props, max_retries=1000, scribble_radius=5):
    """Generate a scribble visual prompt as a wavy line for each region."""
    scribble_prompt = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        center_row, center_col = prop.centroid

        # Create a new box with random scaling
        new_box = _create_valid_box(
            mask, prop, min_row, min_col, max_row, max_col, center_row, center_col, max_retries
        )
        if new_box is None:
            continue

        new_min_row, new_min_col, new_max_row, new_max_col = new_box

        # Define corners of the box
        corners = [
            (new_min_row, new_min_col),
            (new_min_row, new_max_col),
            (new_max_row, new_min_col),
            (new_max_row, new_max_col),
        ]

        # Select diagonal points for the line
        start_point = random.choice(corners)
        corners.remove(start_point)

        # Choose the opposite corner
        if start_point in [(new_min_row, new_min_col), (new_max_row, new_max_col)]:
            end_point = (
                new_max_row if start_point[0] == new_min_row else new_min_row,
                new_max_col if start_point[1] == new_min_col else new_min_col,
            )
        else:
            end_point = (
                new_max_row if start_point[0] == new_min_row else new_min_row,
                new_min_col if start_point[1] == new_max_col else new_max_col,
            )

        # Draw a wavy line between the points
        rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])
        rr = np.array(rr, dtype=np.float32)
        cc = np.array(cc, dtype=np.float32)

        # Add sine wave deformation to the line
        amplitude = random.uniform(10, 20)
        frequency = random.uniform(0.2, 1)
        phase_shift = random.uniform(0, 2 * np.pi)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, 1, len(rr)) + phase_shift)
        rr += sine_wave

        # Ensure coordinates are valid
        rr = np.clip(rr, 0, mask.shape[0] - 1).astype(np.int32)
        cc = np.clip(cc, 0, mask.shape[1] - 1).astype(np.int32)

        scribble_prompt[rr, cc] = 1

    scribble_prompt = enhance_with_circles(scribble_prompt, scribble_radius)

    return scribble_prompt


def generate_box_vprompt(mask, props):
    """Generate a box visual prompt for each region with slight random scaling."""
    box_prompt = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        scale_factor = random.uniform(0.9, 1.1)

        height = max_row - min_row
        width = max_col - min_col

        # Apply scaling to the box
        delta_height = height * (scale_factor - 1)
        delta_width = width * (scale_factor - 1)

        scaled_min_row = max(0, int(min_row - delta_height / 2))
        scaled_min_col = max(0, int(min_col - delta_width / 2))
        scaled_max_row = min(mask.shape[0], int(max_row + delta_height / 2))
        scaled_max_col = min(mask.shape[1], int(max_col + delta_width / 2))

        box_prompt[scaled_min_row:scaled_max_row, scaled_min_col:scaled_max_col] = 1

    return box_prompt


def generate_mask_vprompt(mask):
    """Generate a mask visual prompt by applying Gaussian filter to the original mask."""
    mask_float = mask.astype(float)
    blurred_mask = gaussian_filter(mask_float, sigma=2)
    mask_prompt = (blurred_mask > blurred_mask.mean()).astype(np.uint8)
    return mask_prompt
