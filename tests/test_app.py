from pathlib import Path

import marble_aim.app as app_module
from marble_aim.app import ApplicationController, SceneSnapshot
from marble_aim.config import AppConfig
from marble_aim.geometry import Obstacle, Rect
from marble_aim.vision import DetectionResult


def test_runtime_contains_no_mouse_input_simulation():
    package = Path(__file__).parents[1] / "marble_aim"
    source = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for path in package.glob("*.py")
    )
    forbidden = (
        "pydirectinput",
        "pyautogui",
        "sendinput",
        "mouse_event",
        "logitech",
        "ghub",
    )

    assert not any(token in source for token in forbidden)


def test_scene_search_returns_one_bounded_visual_recommendation():
    controller = ApplicationController.__new__(ApplicationController)
    controller.config = AppConfig()
    scene = SceneSnapshot(
        Rect(0, 0, 160, 220),
        (
            Obstacle(Rect(35, 50, 65, 75), corner_radius=4),
            Obstacle(Rect(95, 50, 125, 75), corner_radius=4),
            Obstacle(Rect(67, 100, 93, 125), corner_radius=4),
        ),
        (80, 220),
        4,
    )

    recommendations = controller._search_for_scene(scene)

    assert len(recommendations) == 1
    assert len(recommendations[0].collisions) <= 8
    assert recommendations[0].angle_deg is not None
    assert -89 <= recommendations[0].angle_deg <= 89


def test_scene_uses_actual_ball_scale_instead_of_large_aim_marker():
    controller = ApplicationController.__new__(ApplicationController)
    controller.config = AppConfig()
    detection = DetectionResult(
        board=Rect(1433, 323, 2403, 1620),
        obstacles=[Obstacle(Rect(1438, 498, 1592, 652), 0)],
        launch_origin=(1900, 1620),
        aim_line=((1900, 1620), (1800, 1100)),
        aim_radius=45,
    )

    scene = controller._make_scene(detection)

    assert scene is not None
    assert scene.ball_radius == 154 * 0.202


def test_advanced_scene_uses_detected_volley_and_durability():
    controller = ApplicationController.__new__(ApplicationController)
    controller.config = AppConfig()
    controller.config.physics.advanced_simulation = True
    obstacle = Obstacle(
        Rect(1438, 498, 1592, 652),
        0,
        durability=48,
        durability_confidence=0.95,
    )
    detection = DetectionResult(
        board=Rect(1433, 323, 2403, 1620),
        obstacles=[obstacle],
        launch_origin=(1900, 1620),
        aim_line=((1900, 1620), (1800, 1100)),
        volley_count=31,
        volley_confidence=0.9,
    )

    scene = controller._make_scene(detection)

    assert scene is not None
    assert scene.volley_count == 31
    assert scene.obstacles[0].durability == 48


def test_advanced_search_passes_detected_volley_count(monkeypatch):
    controller = ApplicationController.__new__(ApplicationController)
    controller.config = AppConfig()
    controller.config.physics.advanced_simulation = True
    captured: dict[str, int] = {}

    def fake_search(*args, **kwargs):
        captured["volley_count"] = kwargs["volley_count"]
        return []

    monkeypatch.setattr(app_module, "search_recommendations", fake_search)
    scene = SceneSnapshot(
        Rect(0, 0, 160, 220),
        (Obstacle(Rect(35, 50, 65, 75), durability=2),),
        (80, 220),
        4,
        volley_count=25,
    )

    controller._search_for_scene(scene)

    assert captured["volley_count"] == 25

    controller.config.physics.advanced_simulation = False
    controller._search_for_scene(scene)

    assert captured["volley_count"] == 1


def test_block_signature_ignores_global_sprite_bobbing():
    controller = ApplicationController.__new__(ApplicationController)
    board = Rect(621, 139, 1042, 702)
    first = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 210, 689, 277), 0),
            Obstacle(Rect(692, 210, 759, 277), 0),
            Obstacle(Rect(762, 280, 829, 347), 0),
        ),
        (830, 702),
        11.5,
    )
    bobbed = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 212, 689, 279), 0),
            Obstacle(Rect(692, 212, 759, 279), 0),
            Obstacle(Rect(762, 282, 829, 349), 0),
        ),
        (830, 702),
        11.5,
    )
    shifted_down_one_row = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 280, 689, 347), 0),
            Obstacle(Rect(692, 280, 759, 347), 0),
            Obstacle(Rect(762, 350, 829, 417), 0),
        ),
        (830, 702),
        11.5,
    )

    assert controller._block_signature(first) == controller._block_signature(bobbed)
    assert (
        controller._block_signature(first)
        != controller._block_signature(shifted_down_one_row)
    )


def test_new_aim_keeps_checking_same_blocks_for_clean_frames():
    controller = ApplicationController.__new__(ApplicationController)
    board = Rect(100, 50, 520, 610)
    obstacles = (Obstacle(Rect(101, 120, 168, 187), 0),)
    scene = SceneSnapshot(board, obstacles, (300, 610), 14.5)
    detection = DetectionResult(
        board=board,
        obstacles=list(obstacles),
        launch_origin=scene.origin,
        aim_line=((300, 610), (500, 200)),
    )

    class DetectorStub:
        def detect(self, frame, *, debug_masks=False):
            return detection

    controller.detector = DetectorStub()
    controller.debug_view = False
    controller.latest_scene = scene
    controller.locked_block_signature = controller._block_signature(scene)
    controller.round_candidate_signature = None
    controller.round_candidate_count = 0
    controller.round_candidate_origins = []
    controller.transition_same_count = 0
    controller._make_scene = lambda current: scene

    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "same"
