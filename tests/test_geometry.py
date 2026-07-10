from __future__ import annotations

import math

import numpy as np

from marble_aim.geometry import (
    Obstacle,
    Rect,
    direction_from_angle,
    estimate_ball_radius,
    ray_rounded_rect,
    reflect,
    search_recommendations,
    simulate_volley_trajectory,
    simulate_trajectory,
    survival_priority_score,
    vec,
)


def test_reflect_from_vertical_wall():
    result = reflect(vec(1, -0.25), vec(-1, 0))
    expected = np.array([-1, -0.25], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert np.allclose(result, expected)


def test_wall_collision_respects_ball_radius():
    board = Rect(0, 0, 100, 100)
    trajectory = simulate_trajectory(
        vec(50, 80),
        vec(0, -1),
        board,
        [],
        ball_radius=5,
        max_collisions=1,
    )
    assert len(trajectory.collisions) == 1
    assert trajectory.collisions[0].kind == "wall"
    assert np.allclose(trajectory.points[1], [50, 5])


def test_simultaneous_corner_collision_combines_normals():
    board = Rect(0, 0, 100, 100)
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(-1, -1),
        board,
        [],
        ball_radius=5,
        max_collisions=1,
    )
    collision = trajectory.collisions[0]
    assert np.allclose(collision.point, [5, 5])
    assert np.allclose(collision.normal, np.array([1, 1]) / math.sqrt(2))


def test_square_block_front_collision():
    obstacle = Obstacle(Rect(40, 35, 60, 55), corner_radius=0)
    hit = ray_rounded_rect(vec(50, 80), vec(0, -1), obstacle, ball_radius=5)
    assert hit is not None
    assert math.isclose(hit.point[1], 60, abs_tol=1e-6)
    assert np.allclose(hit.normal, [0, 1])


def test_rounded_corner_collision_has_diagonal_normal():
    obstacle = Obstacle(Rect(40, 40, 60, 60), corner_radius=5)
    hit = ray_rounded_rect(vec(25, 25), vec(1, 1), obstacle, ball_radius=2)
    assert hit is not None
    assert hit.normal[0] < 0
    assert hit.normal[1] < 0
    assert math.isclose(float(np.linalg.norm(hit.normal)), 1.0, abs_tol=1e-8)


def test_block_reflection_is_counted():
    board = Rect(0, 0, 100, 120)
    obstacle = Obstacle(Rect(40, 35, 60, 55), corner_radius=3)
    trajectory = simulate_trajectory(
        vec(50, 100),
        vec(0, -1),
        board,
        [obstacle],
        ball_radius=4,
        max_collisions=2,
        reflect_bottom=True,
    )
    assert trajectory.block_hits == 1
    assert trajectory.collisions[0].kind == "block"
    assert trajectory.collisions[1].kind == "wall"


def test_simulation_stops_at_collision_limit():
    trajectory = simulate_trajectory(
        vec(50, 50),
        direction_from_angle(35),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=3,
        max_collisions=5,
        max_distance=10000,
        reflect_bottom=True,
    )
    assert len(trajectory.collisions) == 5
    assert len(trajectory.points) == 6


def test_recommendations_are_separated():
    board = Rect(0, 0, 160, 220)
    obstacles = [
        Obstacle(Rect(35, 50, 65, 75), corner_radius=4),
        Obstacle(Rect(95, 50, 125, 75), corner_radius=4),
        Obstacle(Rect(67, 100, 93, 125), corner_radius=4),
    ]
    recommendations = search_recommendations(
        vec(80, 205),
        board,
        obstacles,
        ball_radius=4,
        max_collisions=12,
        max_distance=1600,
        coarse_step=4,
        fine_step=1,
        count=3,
        separation=2,
    )
    assert len(recommendations) == 3
    angles = [item.angle_deg for item in recommendations]
    assert all(angle is not None for angle in angles)
    assert min(
        abs(float(first) - float(second))
        for index, first in enumerate(angles)
        for second in angles[index + 1 :]
    ) >= 2


def test_recommendations_stay_in_forward_firing_arc():
    recommendations = search_recommendations(
        vec(80, 205),
        Rect(0, 0, 160, 220),
        [Obstacle(Rect(65, 70, 95, 100), corner_radius=4)],
        angle_min=-180,
        angle_max=180,
        ball_radius=4,
        max_collisions=8,
        max_distance=1600,
        coarse_step=5,
        fine_step=1,
        count=2,
    )

    assert recommendations
    assert all(
        item.angle_deg is not None and -89 <= item.angle_deg <= 89
        for item in recommendations
    )


def test_bottom_is_an_open_exit_without_reflection():
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(0, 1),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=7,
        max_collisions=10,
    )
    assert len(trajectory.collisions) == 0
    assert np.allclose(trajectory.points[-1], [50, 100])


def test_side_wall_contact_uses_ball_edge_not_center():
    trajectory = simulate_trajectory(
        vec(50, 50),
        vec(-1, 0),
        Rect(0, 0, 100, 100),
        [],
        ball_radius=7,
        max_collisions=1,
    )
    assert np.allclose(trajectory.points[1], [7, 50])
    assert np.allclose(trajectory.collisions[0].normal, [1, 0])


def test_durability_caps_effective_damage_and_tracks_coverage():
    board = Rect(0, 0, 120, 160)
    obstacle = Obstacle(
        Rect(45, 50, 75, 75),
        corner_radius=4,
        durability=1,
    )
    trajectory = simulate_trajectory(
        vec(60, 140),
        vec(0, -1),
        board,
        [obstacle],
        ball_radius=4,
        max_collisions=4,
        reflect_bottom=True,
    )
    assert trajectory.unique_block_hits == 1
    assert trajectory.effective_damage == 1
    assert trajectory.destroyed_blocks == 1


def test_volley_removes_block_after_durability_is_consumed():
    board = Rect(0, 0, 120, 160)
    obstacle = Obstacle(
        Rect(45, 50, 75, 75),
        corner_radius=4,
        durability=2,
    )
    trajectory = simulate_volley_trajectory(
        vec(60, 160),
        vec(0, -1),
        board,
        [obstacle],
        volley_count=5,
        ball_radius=4,
        max_collisions=8,
    )
    assert trajectory.unique_block_hits == 1
    assert trajectory.effective_damage == 2
    assert trajectory.destroyed_blocks == 1
    assert trajectory.stable_balls_before_change == 2


def test_volley_recomputes_path_after_front_block_is_destroyed():
    board = Rect(0, 0, 120, 180)
    obstacles = [
        Obstacle(Rect(45, 100, 75, 125), corner_radius=4, durability=1),
        Obstacle(Rect(45, 40, 75, 65), corner_radius=4, durability=1),
    ]

    trajectory = simulate_volley_trajectory(
        vec(60, 180),
        vec(0, -1),
        board,
        obstacles,
        volley_count=2,
        ball_radius=4,
        max_collisions=8,
    )

    assert trajectory.unique_block_hits == 2
    assert trajectory.effective_damage == 2
    assert trajectory.destroyed_blocks == 2


def test_unknown_durability_is_not_removed_during_the_volley():
    trajectory = simulate_volley_trajectory(
        vec(60, 160),
        vec(0, -1),
        Rect(0, 0, 120, 160),
        [Obstacle(Rect(45, 50, 75, 75), corner_radius=4)],
        volley_count=5,
        ball_radius=4,
        max_collisions=8,
    )

    assert trajectory.destroyed_blocks == 0


def test_ball_radius_scales_with_detected_block_size():
    board = Rect(0, 0, 430, 574)
    obstacle = Obstacle(Rect(20, 20, 89, 89))
    radius = estimate_ball_radius(board, [obstacle])
    scaled_radius = estimate_ball_radius(
        Rect(0, 0, 860, 1148),
        [Obstacle(Rect(40, 40, 178, 178))],
    )
    assert math.isclose(radius, 69 * 0.202, rel_tol=1e-6)
    assert math.isclose(scaled_radius, radius * 2, rel_tol=1e-6)


def test_ball_radius_falls_back_to_board_scale_without_blocks():
    radius = estimate_ball_radius(Rect(0, 0, 430, 574), [])
    assert math.isclose(radius, 430 * 0.032, rel_tol=1e-6)


def test_bottom_row_hit_has_survival_priority_over_upper_hits():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, 120, 190, 210)),
        Obstacle(Rect(300, 610, 390, 700)),
    ]

    upper_score = survival_priority_score(board, obstacles, {0: 6})
    bottom_score = survival_priority_score(board, obstacles, {1: 1})

    assert bottom_score > upper_score


def test_survival_priority_increases_for_every_playable_row():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, top, 190, top + 80))
        for top in (110, 210, 310, 410, 510, 610)
    ]

    scores = [
        survival_priority_score(board, obstacles, {index: 1})
        for index in range(len(obstacles))
    ]

    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


def test_one_hit_near_loss_row_outweighs_many_hits_near_spawn():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, 110, 190, 190)),
        Obstacle(Rect(300, 610, 390, 690)),
    ]

    upper_score = survival_priority_score(board, obstacles, {0: 8})
    critical_score = survival_priority_score(board, obstacles, {1: 1})

    assert critical_score > upper_score


def test_critical_row_damage_can_outweigh_destroying_safer_block():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, 510, 190, 590), durability=1),
        Obstacle(Rect(300, 610, 390, 690), durability=100),
    ]

    safer_destroyed = survival_priority_score(board, obstacles, {0: 1}, {0})
    critical_damaged = survival_priority_score(board, obstacles, {1: 50}, set())

    assert critical_damaged > safer_destroyed


def test_high_hp_critical_block_keeps_priority_after_one_hit():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, 110, 190, 190), durability=500),
        Obstacle(Rect(300, 610, 390, 690), durability=500),
    ]

    upper_score = survival_priority_score(board, obstacles, {0: 8}, set())
    critical_score = survival_priority_score(board, obstacles, {1: 1}, set())

    assert critical_score > upper_score * 100


def test_clearing_all_critical_blocks_receives_survival_bonus():
    board = Rect(0, 0, 600, 800)
    obstacles = [
        Obstacle(Rect(100, 610, 190, 690), durability=10),
        Obstacle(Rect(300, 610, 390, 690), durability=10),
    ]

    one_destroyed = survival_priority_score(
        board,
        obstacles,
        {0: 10, 1: 9},
        {0},
    )
    all_destroyed = survival_priority_score(
        board,
        obstacles,
        {0: 10, 1: 10},
        {0, 1},
    )

    assert all_destroyed > one_destroyed * 2
