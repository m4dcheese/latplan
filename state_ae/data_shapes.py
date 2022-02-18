from typing import Union
import cv2 as cv
import numpy as np

from parameters import parameters

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 128, 0)
LIGHTBLUE = (0, 128, 255)
LIME = (128, 255, 0)

def circle(
    img: np.array,
    top_offset: int,
    left_offset: int,
    field_resolution: int,
    field_padding: int,
    color: tuple
) -> None:
    center = (
        left_offset + int(field_resolution / 2),
        top_offset + int(field_resolution / 2)
    )
    size = (int(field_resolution / 2) - field_padding, int(field_resolution / 2) - field_padding)
    cv.ellipse(img, center, size, angle=0, startAngle=0, endAngle=360, color=color, thickness=-1)

def square(
    img: np.array,
    top_offset: int,
    left_offset: int,
    field_resolution: int,
    field_padding: int,
    color: tuple
) -> None:
    top_left = (left_offset + field_padding, top_offset + field_padding)
    bottom_right = (left_offset + field_resolution - field_padding, top_offset + field_resolution - field_padding)
    cv.rectangle(img, pt1=top_left, pt2=bottom_right, color=color, thickness=-1)

def rectangle(
    img: np.array,
    top_offset: int,
    left_offset: int,
    field_resolution: int,
    field_padding: int,
    color: tuple
) -> None:
    if parameters.random_orientation:
        rotation = np.random.randint(0, 2)
    else:
        rotation = 0

    if rotation == 0:
        top_left = (left_offset + field_padding, top_offset + field_padding + int((field_resolution - field_padding*2)/3))
        bottom_right = (
            left_offset + field_resolution - field_padding,
            top_offset + field_resolution - field_padding - int((field_resolution - field_padding*2)/3)
        )
    elif rotation == 1:
        top_left = (left_offset + field_padding + int((field_resolution - field_padding*2)/3), top_offset + field_padding)
        bottom_right = (
            left_offset + field_resolution - field_padding - int((field_resolution - field_padding*2)/3),
            top_offset + field_resolution - field_padding
        )

    cv.rectangle(img, pt1=top_left, pt2=bottom_right, color=color, thickness=-1)

def triangle(
    img: np.array,
    top_offset: int,
    left_offset: int,
    field_resolution: int,
    field_padding: int,
    color: tuple
) -> None:
    if parameters.random_orientation:
        rotation = np.random.randint(0, 3)
    else:
        rotation = 0

    if rotation == 0:
        bottom_left = (left_offset + field_padding, top_offset + field_resolution - field_padding)
        bottom_right = (left_offset + field_resolution - field_padding, top_offset + field_resolution - field_padding)
        center_top = (left_offset + int(field_resolution / 2), top_offset + field_padding)
    elif rotation == 1: # +90 degrees
        bottom_left = (left_offset + field_padding, top_offset + field_padding)
        bottom_right = (left_offset + field_padding, top_offset + field_resolution - field_padding)
        center_top = (left_offset + field_resolution - field_padding, top_offset + int(field_resolution / 2))
    elif rotation == 2: # -90 degrees
        bottom_left = (left_offset + field_resolution - field_padding, top_offset + field_resolution - field_padding)
        bottom_right = (left_offset + field_resolution - field_padding, top_offset + field_padding)
        center_top = (left_offset + field_padding, top_offset + int(field_resolution / 2))

    pts=np.array([bottom_left, bottom_right, center_top])
    cv.fillPoly(img, pts=[pts], color=color)


colors = (CYAN, MAGENTA, YELLOW, RED, GREEN, BLUE, ORANGE, LIGHTBLUE, LIME)
shapes = (circle, rectangle, triangle)


def generate_shapes(
    permutations: np.array,
    field_resolution: int = parameters.field_resolution,
    field_padding: Union[int, float] = parameters.field_padding,
    field_random_offset: int = parameters.field_random_offset,
    blur: float = parameters.blur,
    remove_target_offset: bool = parameters.remove_target_offset,
    random_distribution_prob: float = parameters.random_distribution_prob,
    color_table_size: int = parameters.color_table_size,
    background: int = parameters.background,
    random_colors: bool = parameters.random_colors
):
    shape_permutations_input = []
    shape_permutations_target = []
    size = int(np.sqrt(permutations.shape[1]))
    rng = np.random.default_rng()

    if type(field_padding) == float:
        field_padding = int(field_resolution * field_padding)
    
    if random_distribution_prob == 1:
        random_distribution = True
    elif random_distribution_prob == 0:
        random_distribution = False

    for p in permutations:
        image_input = background + np.zeros((field_resolution * size, field_resolution * size, 3))
        image_target = background + np.zeros((field_resolution * size, field_resolution * size, 3))
        color_random_indices = rng.permutation(np.arange(permutations.shape[1]))
        for row in range(size):
            for col in range(size):
                if random_distribution_prob > 0 and random_distribution_prob < 1:
                    random_distribution = rng.binomial(1, random_distribution_prob) == 1

                permutation_index = row * size + col
                permutation_value = p[permutation_index]
                if (permutation_value >= permutations.shape[1]):
                    # Deletion occured at this position
                    continue
                perm_row, perm_col = (permutation_value // size, permutation_value % size)

                if random_distribution:
                    top_offset = int(rng.uniform(low=0, high=(size - 1) * field_resolution))
                    left_offset = int(rng.uniform(low=0, high=(size - 1) * field_resolution))
                else:
                    top_offset = row * field_resolution + int(rng.uniform(low=-field_random_offset, high=field_random_offset))
                    left_offset = col * field_resolution + int(rng.uniform(low=-field_random_offset, high=field_random_offset))

                # Determine color
                if random_colors:
                    color_idx = color_random_indices[perm_row * size + perm_col] % color_table_size
                else:
                    color_idx = (perm_row * size + perm_col) % color_table_size

                # shapes[perm_row] is one of the shape functions and adapts the image parameter in place
                shapes[perm_row](
                    image_input,
                    top_offset,
                    left_offset,
                    field_resolution,
                    field_padding,
                    colors[color_idx]
                )

                # Image target
                if remove_target_offset and not random_distribution:
                    top_offset = row * field_resolution
                    left_offset = col * field_resolution

                shapes[perm_row](
                    image_target,
                    top_offset,
                    left_offset,
                    field_resolution,
                    field_padding,
                    colors[color_idx]
                )
        if blur != 0:
            image_input = cv.GaussianBlur(image_input, ksize=(5, 5), sigmaX=blur)
            image_target = cv.GaussianBlur(image_target, ksize=(5, 5), sigmaX=blur)
        shape_permutations_input.append(image_input)
        shape_permutations_target.append(image_target)
    return shape_permutations_input, shape_permutations_target
