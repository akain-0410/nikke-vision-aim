from __future__ import annotations

import json

from marble_aim.config import AppConfig


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig(window_title="Example Game")
    config.physics.ball_radius = 8.25
    config.calibration.boundary_offsets = [1.0, -2.0, 3.0, -4.0]
    config.save(path)

    loaded = AppConfig.load(path)
    assert loaded.window_title == "Example Game"
    assert loaded.physics.ball_radius == 8.25
    assert loaded.calibration.boundary_offsets == [1.0, -2.0, 3.0, -4.0]


def test_missing_config_uses_defaults(tmp_path):
    loaded = AppConfig.load(tmp_path / "missing.json")
    assert loaded.physics.max_collisions == 8
    assert loaded.physics.recommendation_count == 1
    assert loaded.physics.volley_count == 1
    assert loaded.physics.advanced_simulation is False
    assert loaded.overlay.visible is True
    assert loaded.overlay.show_collision_frame is True
    assert loaded.vision.aim_transition_delay_ms == 600


def test_legacy_config_is_bounded_to_manual_overlay_search(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "physics": {
                    "max_collisions": 40,
                    "angle_min_deg": -180,
                    "angle_max_deg": 180,
                    "recommendation_count": 3,
                    "volley_count": 67,
                    "ball_radius_to_block_ratio": 0.215,
                    "ball_radius_to_board_width_ratio": 0.0345,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = AppConfig.load(path)

    assert loaded.physics.max_collisions == 10
    assert loaded.physics.angle_min_deg == -89
    assert loaded.physics.angle_max_deg == 89
    assert loaded.physics.recommendation_count == 1
    assert loaded.physics.volley_count == 67
    assert loaded.physics.advanced_simulation is False
    assert loaded.physics.ball_radius_to_block_ratio == 0.202
    assert loaded.physics.ball_radius_to_board_width_ratio == 0.032


def test_invalid_trajectory_mode_falls_back_to_mode_a(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"overlay": {"trajectory_mode": "unknown"}}),
        encoding="utf-8",
    )

    loaded = AppConfig.load(path)

    assert loaded.overlay.trajectory_mode == "A"


def test_mode_d_and_advanced_simulation_round_trip(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig()
    config.overlay.trajectory_mode = "D"
    config.physics.advanced_simulation = True
    config.physics.volley_count = 31
    config.save(path)

    loaded = AppConfig.load(path)

    assert loaded.overlay.trajectory_mode == "D"
    assert loaded.physics.advanced_simulation is True
    assert loaded.physics.volley_count == 31
