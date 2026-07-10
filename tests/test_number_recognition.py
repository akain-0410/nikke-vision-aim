from pathlib import Path

import cv2
import numpy as np

from marble_aim.geometry import Obstacle, Rect
from marble_aim.number_recognition import (
    recognize_obstacle_durabilities,
    recognize_volley_count,
)


def _template(digit: int, group: str) -> np.ndarray:
    path = (
        Path(__file__).parents[1]
        / "marble_aim"
        / "assets"
        / "digit_templates.npz"
    )
    with np.load(path) as data:
        labels = data[f"{group}_labels"]
        templates = data[f"{group}_templates"]
        return np.array(templates[np.flatnonzero(labels == digit)[0]], dtype=np.uint8)


def _paint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    height, width = mask.shape
    region = image[y : y + height, x : x + width]
    region[mask > 0] = color


def test_recognizes_three_digit_block_durability():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for index, digit in enumerate((1, 1, 0)):
        _paint_mask(
            image,
            _template(digit, "block"),
            52 + index * 32,
            55,
            (255, 255, 255),
        )
    obstacle = Obstacle(Rect(20, 20, 180, 180), corner_radius=8)

    recognized = recognize_obstacle_durabilities(image, [obstacle])

    assert recognized[0].durability == 110
    assert recognized[0].durability_confidence >= 0.9


def test_recognizes_prefixed_volley_count():
    image = np.zeros((1320, 1000, 3), dtype=np.uint8)
    red = (55, 55, 170)
    first = cv2.resize(_template(3, "count"), (24, 36), interpolation=cv2.INTER_AREA)
    second = cv2.resize(_template(1, "count"), (24, 36), interpolation=cv2.INTER_AREA)
    _paint_mask(image, first, 500, 1220, red)
    _paint_mask(image, second, 530, 1220, red)
    cv2.line(image, (478, 1232), (494, 1250), red, 5, cv2.LINE_AA)
    cv2.line(image, (494, 1232), (478, 1250), red, 5, cv2.LINE_AA)

    recognition = recognize_volley_count(
        image,
        Rect(5, 5, 995, 1300),
        launch_x=485,
    )

    assert recognition is not None
    assert recognition.value == 31
    assert recognition.confidence >= 0.5
