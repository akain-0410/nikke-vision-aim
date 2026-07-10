from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Obstacle, Rect

Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class NumberRecognition:
    value: int
    confidence: float


@dataclass(frozen=True, slots=True)
class _Component:
    x: int
    y: int
    width: int
    height: int
    mask: Image
    digit: int
    score: float


def _normalize_glyph(mask: Image) -> Image:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((48, 32), dtype=np.uint8)
    glyph = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min(28.0 / glyph.shape[1], 44.0 / glyph.shape[0])
    width = max(1, round(glyph.shape[1] * scale))
    height = max(1, round(glyph.shape[0] * scale))
    glyph = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)
    normalized = np.zeros((48, 32), dtype=np.uint8)
    top = (48 - height) // 2
    left = (32 - width) // 2
    normalized[top : top + height, left : left + width] = glyph
    return normalized


def _features(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    flattened = images.reshape(images.shape[0], -1).astype(np.float32)
    flattened -= np.mean(flattened, axis=1, keepdims=True)
    lengths = np.linalg.norm(flattened, axis=1, keepdims=True)
    return flattened / np.maximum(lengths, 1e-6)


@lru_cache(maxsize=1)
def _template_data() -> tuple[
    NDArray[np.float32],
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.uint8],
]:
    path = Path(__file__).with_name("assets") / "digit_templates.npz"
    with np.load(path) as data:
        block_templates = np.array(data["block_templates"], dtype=np.uint8)
        block_labels = np.array(data["block_labels"], dtype=np.uint8)
        count_templates = np.array(data["count_templates"], dtype=np.uint8)
        count_labels = np.array(data["count_labels"], dtype=np.uint8)
    return (
        _features(block_templates),
        block_labels,
        _features(count_templates),
        count_labels,
    )


def _classify(
    mask: Image,
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
) -> tuple[int, float]:
    glyph = _normalize_glyph(mask).reshape(1, -1).astype(np.float32)
    glyph -= np.mean(glyph)
    glyph /= max(float(np.linalg.norm(glyph)), 1e-6)
    scores = features @ glyph[0]
    index = int(np.argmax(scores))
    return int(labels[index]), float(scores[index])


def _block_components(
    hsv: Image,
    obstacle: Obstacle,
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
) -> list[_Component]:
    left, top, right, bottom = map(
        round,
        (
            obstacle.rect.left,
            obstacle.rect.top,
            obstacle.rect.right,
            obstacle.rect.bottom,
        ),
    )
    width = right - left
    height = bottom - top
    x0 = max(0, left + round(width * 0.08))
    x1 = min(hsv.shape[1], right - round(width * 0.08))
    y0 = max(0, top + round(height * 0.07))
    y1 = min(hsv.shape[0], top + round(height * 0.64))
    if x1 <= x0 or y1 <= y0:
        return []
    mask = np.uint8(
        (hsv[y0:y1, x0:x1, 1] < 95)
        & (hsv[y0:y1, x0:x1, 2] > 180)
    ) * 255
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[_Component] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[index]
        )
        if not (
            height * 0.17 <= component_height <= height * 0.38
            and width * 0.04 <= component_width <= width * 0.29
            and area >= height * width * 0.010
        ):
            continue
        glyph = (
            np.uint8(
                component_map[
                    y : y + component_height,
                    x : x + component_width,
                ]
                == index
            )
            * 255
        )
        digit, score = _classify(glyph, features, labels)
        candidates.append(
            _Component(
                x + x0,
                y + y0,
                component_width,
                component_height,
                glyph,
                digit,
                score,
            )
        )
    return candidates


def _recognize_block(
    hsv: Image,
    obstacle: Obstacle,
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
) -> NumberRecognition | None:
    candidates = sorted(
        _block_components(hsv, obstacle, features, labels),
        key=lambda item: item.x,
    )
    width = obstacle.rect.width
    height = obstacle.rect.height
    center_x = float(obstacle.rect.center[0])
    best: tuple[float, NumberRecognition] | None = None
    for length in (1, 2, 3):
        for sequence in itertools.combinations(candidates, length):
            if (
                max(item.y for item in sequence) - min(item.y for item in sequence)
                > height * 0.055
            ):
                continue
            if (
                max(item.height for item in sequence)
                - min(item.height for item in sequence)
                > height * 0.06
            ):
                continue
            if any(
                sequence[index + 1].x
                < sequence[index].x + sequence[index].width * 0.80
                or sequence[index + 1].x
                - (sequence[index].x + sequence[index].width)
                > width * 0.12
                for index in range(length - 1)
            ):
                continue
            sequence_center = (
                sequence[0].x
                + sequence[-1].x
                + sequence[-1].width
            ) / 2
            center_error = abs(sequence_center - center_x) / max(1.0, width)
            if center_error > 0.22:
                continue
            raw_confidence = float(np.mean([item.score for item in sequence]))
            if raw_confidence < 0.52 or min(item.score for item in sequence) < 0.45:
                continue
            text = "".join(str(item.digit) for item in sequence)
            if len(text) > 1 and text.startswith("0"):
                continue
            value = int(text)
            if not 1 <= value <= 999:
                continue
            spread = max(item.score for item in sequence) - min(
                item.score for item in sequence
            )
            ranking = (
                raw_confidence
                + (length - 1) * 0.03
                - center_error * 0.55
                - spread * 0.08
            )
            confidence = float(np.clip((raw_confidence - 0.45) / 0.50, 0.0, 1.0))
            recognition = NumberRecognition(value, confidence)
            if best is None or ranking > best[0]:
                best = (ranking, recognition)
    return best[1] if best is not None else None


def recognize_obstacle_durabilities(
    bgr: Image,
    obstacles: list[Obstacle],
) -> list[Obstacle]:
    if not obstacles:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    block_features, block_labels, _, _ = _template_data()
    recognized: list[Obstacle] = []
    for obstacle in obstacles:
        number = _recognize_block(hsv, obstacle, block_features, block_labels)
        recognized.append(
            Obstacle(
                obstacle.rect,
                obstacle.corner_radius,
                obstacle.confidence,
                obstacle.identifier,
                number.value if number is not None else None,
                number.confidence if number is not None else 0.0,
            )
        )
    return recognized


def recognize_volley_count(
    bgr: Image,
    board: Rect,
    launch_x: float | None = None,
) -> NumberRecognition | None:
    if board.width <= 0 or board.height <= 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 50)
        & (hsv[:, :, 2] <= 225)
    )
    scale = board.width / 990.0
    x0 = max(0, round(board.left - board.width * 0.03))
    x1 = min(bgr.shape[1], round(board.right + board.width * 0.03))
    y0 = max(0, round(board.bottom - board.width * 0.11))
    y1 = min(bgr.shape[0], round(board.bottom + board.width * 0.02))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.uint8(red[y0:y1, x0:x1]) * 255
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(mask)
    block_features, block_labels, count_features, count_labels = _template_data()
    del block_features, block_labels
    digits: list[_Component] = []
    prefixes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        absolute_x = x + x0
        absolute_y = y + y0
        if (
            28 * scale <= height <= 43 * scale
            and 9 * scale <= width <= 32 * scale
            and area >= 175 * scale * scale
        ):
            glyph = (
                np.uint8(
                    component_map[y : y + height, x : x + width] == index
                )
                * 255
            )
            digit, score = _classify(glyph, count_features, count_labels)
            digits.append(
                _Component(
                    absolute_x,
                    absolute_y,
                    width,
                    height,
                    glyph,
                    digit,
                    score,
                )
            )
        if (
            18 * scale <= height <= 32 * scale
            and 9 * scale <= width <= 31 * scale
            and area >= 90 * scale * scale
        ):
            prefixes.append((absolute_x, absolute_y, width, height))

    digits.sort(key=lambda item: item.x)
    best: tuple[float, NumberRecognition] | None = None
    for length in (1, 2, 3):
        for sequence in itertools.combinations(digits, length):
            if (
                max(item.y for item in sequence) - min(item.y for item in sequence)
                > 6 * scale
            ):
                continue
            if (
                max(item.height for item in sequence)
                - min(item.height for item in sequence)
                > 6 * scale
            ):
                continue
            if any(
                not (
                    2 * scale
                    <= sequence[index + 1].x
                    - (sequence[index].x + sequence[index].width)
                    <= 12 * scale
                )
                for index in range(length - 1)
            ):
                continue
            prefix = next(
                (
                    item
                    for item in prefixes
                    if 1 * scale
                    <= sequence[0].x - (item[0] + item[2])
                    <= 14 * scale
                    and 2 * scale
                    <= item[1] - sequence[0].y
                    <= 18 * scale
                ),
                None,
            )
            if prefix is None:
                continue
            raw_confidence = float(np.mean([item.score for item in sequence]))
            if raw_confidence < 0.48 or min(item.score for item in sequence) < 0.42:
                continue
            text = "".join(str(item.digit) for item in sequence)
            if len(text) > 1 and text.startswith("0"):
                continue
            value = int(text)
            if not 1 <= value <= 200:
                continue
            group_center = (
                prefix[0] + sequence[-1].x + sequence[-1].width
            ) / 2
            launch_error = (
                abs(group_center - launch_x) / max(1.0, board.width)
                if launch_x is not None
                else 0.0
            )
            ranking = (
                raw_confidence
                + (length - 1) * 0.05
                - min(0.15, launch_error) * 0.25
            )
            confidence = float(np.clip((raw_confidence - 0.40) / 0.45, 0.0, 1.0))
            recognition = NumberRecognition(value, confidence)
            if best is None or ranking > best[0]:
                best = (ranking, recognition)
    return best[1] if best is not None else None
