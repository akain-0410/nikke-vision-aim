from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .config import AppConfig
from .geometry import Obstacle, Rect, Trajectory


def capture_to_overlay_scale(
    overlay_width: int,
    overlay_height: int,
    capture_width: int,
    capture_height: int,
) -> tuple[float, float]:
    if capture_width <= 0 or capture_height <= 0:
        return 1.0, 1.0
    return overlay_width / capture_width, overlay_height / capture_height


def trajectory_mode_visibility(mode: str) -> tuple[bool, bool]:
    mode = mode.upper()
    return mode in {"A", "C"}, mode in {"C", "D"}


class RefreshButton(QWidget):
    refresh_requested = Signal()
    select_window_requested = Signal()
    mode_cycle_requested = Signal()
    advanced_toggle_requested = Signal()
    exit_requested = Signal()

    def __init__(
        self,
        trajectory_mode: str = "A",
        advanced_simulation: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        refresh_button = QPushButton("重启识别")
        refresh_button.setFixedSize(92, 34)
        refresh_button.setStyleSheet(
            "QPushButton {"
            "background:#163A50; color:white; border:2px solid #00E7FF;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#235A77; }"
            "QPushButton:pressed { background:#0B2635; }"
        )
        refresh_button.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(refresh_button)

        select_button = QPushButton("选择窗口")
        select_button.setFixedSize(88, 34)
        select_button.setStyleSheet(
            "QPushButton {"
            "background:#33305A; color:white; border:2px solid #A69CFF;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#4A467C; }"
            "QPushButton:pressed { background:#24213F; }"
        )
        select_button.clicked.connect(self.select_window_requested.emit)
        layout.addWidget(select_button)

        self.mode_button = QPushButton()
        self.mode_button.setFixedSize(78, 34)
        self.mode_button.setStyleSheet(
            "QPushButton {"
            "background:#204A31; color:white; border:2px solid #50FF7A;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#2E6844; }"
            "QPushButton:pressed { background:#153421; }"
        )
        self.mode_button.clicked.connect(self.mode_cycle_requested.emit)
        layout.addWidget(self.mode_button)
        self.set_trajectory_mode(trajectory_mode)

        self.advanced_button = QPushButton()
        self.advanced_button.setFixedSize(92, 34)
        self.advanced_button.clicked.connect(self.advanced_toggle_requested.emit)
        layout.addWidget(self.advanced_button)
        self.set_advanced_simulation(advanced_simulation)

        exit_button = QPushButton("退出助手")
        exit_button.setFixedSize(82, 34)
        exit_button.setStyleSheet(
            "QPushButton {"
            "background:#4A2028; color:white; border:2px solid #FF6075;"
            "border-radius:7px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#70313D; }"
            "QPushButton:pressed { background:#32151B; }"
        )
        exit_button.clicked.connect(self.exit_requested.emit)
        layout.addWidget(exit_button)
        self.adjustSize()

    def set_trajectory_mode(self, mode: str) -> None:
        mode = mode if mode in {"A", "B", "C", "D"} else "A"
        descriptions = {
            "A": "绿落点＋紫预测",
            "B": "仅绿落点",
            "C": "绿落点＋绿预测＋紫预测",
            "D": "绿落点＋绿预测",
        }
        self.mode_button.setText(f"模式 {mode}")
        self.mode_button.setToolTip(descriptions[mode])

    def set_advanced_simulation(self, enabled: bool) -> None:
        state = "ON" if enabled else "OFF"
        border = "#FFB13B" if enabled else "#82909A"
        background = "#5A3B16" if enabled else "#293138"
        hover = "#76501F" if enabled else "#3A464E"
        self.advanced_button.setText(f"高级 {state}")
        self.advanced_button.setToolTip(
            "识别砖块血量和弹珠数量，并模拟中途消块后的后续弹道"
        )
        self.advanced_button.setStyleSheet(
            "QPushButton {"
            f"background:{background}; color:white; border:2px solid {border};"
            "border-radius:7px; font-weight:bold;"
            "}"
            f"QPushButton:hover {{ background:{hover}; }}"
            "QPushButton:pressed { background:#211B15; }"
        )


class OverlayWindow(QWidget):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.current: Trajectory | None = None
        self.recommendations: list[Trajectory] = []
        self.board: Rect | None = None
        self.recognition_region: Rect | None = None
        self.obstacles: list[Obstacle] = []
        self.ball_radius = config.physics.ball_radius
        self.status = "正在等待游戏画面…"
        self.debug_view = False
        self.calibrating = False
        self.capture_width = 0
        self.capture_height = 0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def set_capture_size(self, width: int, height: int) -> None:
        self.capture_width = width
        self.capture_height = height
        self.update()

    def update_scene(
        self,
        current: Trajectory | None,
        recommendations: list[Trajectory],
        board: Rect | None,
        obstacles: list[Obstacle],
        status: str,
        ball_radius: float | None = None,
        recognition_region: Rect | None = None,
    ) -> None:
        self.current = current
        self.recommendations = recommendations
        self.board = board
        self.recognition_region = recognition_region
        self.obstacles = obstacles
        self.status = status
        if ball_radius is not None:
            self.ball_radius = ball_radius
        self.update()

    def _draw_trajectory(
        self, painter: QPainter, trajectory: Trajectory, color: QColor, width: float
    ) -> None:
        if len(trajectory.points) < 2:
            return
        path = QPainterPath(QPointF(*trajectory.points[0]))
        for point in trajectory.points[1:]:
            path.lineTo(QPointF(*point))
        visual_scale = (
            max(0.70, min(2.75, self.board.width / 420.0))
            if self.board is not None
            else 1.0
        )
        painter.setPen(
            QPen(color, width * visual_scale, Qt.PenStyle.SolidLine)
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        if self.config.overlay.show_collision_points:
            radius = self.ball_radius
            ball_fill = QColor(255, 255, 255, 150)
            painter.setBrush(ball_fill)
            painter.setPen(QPen(color, 1.8 * visual_scale))
            for collision in trajectory.collisions:
                painter.drawEllipse(QPointF(*collision.point), radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_recommendation_marker(
        self,
        painter: QPainter,
        trajectory: Trajectory,
        color: QColor,
    ) -> None:
        if len(trajectory.points) < 2:
            return
        point = trajectory.points[1]
        visual_scale = (
            max(0.70, min(2.75, self.board.width / 420.0))
            if self.board is not None
            else 1.0
        )
        radius = 8.0 * visual_scale
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70))
        painter.setPen(QPen(color, 2.2 * visual_scale))
        center = QPointF(float(point[0]), float(point[1]))
        painter.drawEllipse(center, radius, radius)
        painter.drawLine(
            QPointF(center.x() - radius * 1.5, center.y()),
            QPointF(center.x() + radius * 1.5, center.y()),
        )
        painter.drawLine(
            QPointF(center.x(), center.y() - radius * 1.5),
            QPointF(center.x(), center.y() + radius * 1.5),
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self.config.overlay.opacity)
        if self.calibrating:
            painter.setPen(QColor("#FFFFFF"))
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(15, 15, min(560, self.width() - 30), 58, 10, 10)
            painter.drawText(
                30,
                38,
                "校准录制中：请分别向左、中、右发射，尽量制造墙壁反弹。",
            )
            painter.drawText(30, 60, self.status)
            return
        scale_x, scale_y = capture_to_overlay_scale(
            self.width(),
            self.height(),
            self.capture_width,
            self.capture_height,
        )
        painter.save()
        painter.scale(scale_x, scale_y)
        if self.recognition_region and self.config.overlay.show_collision_frame:
            region = self.recognition_region
            visual_scale = max(0.70, min(2.75, region.width / 420.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    QColor("#00E7FF"),
                    1.0 * visual_scale,
                    Qt.PenStyle.SolidLine,
                )
            )
            painter.drawRect(
                region.left,
                region.top,
                region.width,
                region.height,
            )
        if self.board and self.config.overlay.show_collision_frame:
            board = self.board
            visual_scale = max(0.70, min(2.75, board.width / 420.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    QColor("#A69CFF"),
                    1.0 * visual_scale,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.drawLine(
                QPointF(board.left, board.bottom),
                QPointF(board.left, board.top),
            )
            painter.drawLine(
                QPointF(board.left, board.top),
                QPointF(board.right, board.top),
            )
            painter.drawLine(
                QPointF(board.right, board.top),
                QPointF(board.right, board.bottom),
            )
            painter.setPen(
                QPen(
                    QColor("#FFB13B"),
                    1.8 * visual_scale,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.drawLine(
                QPointF(board.left, board.bottom),
                QPointF(board.right, board.bottom),
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(
                QPointF(board.left + 6, board.bottom - 6),
                "OPEN BOTTOM",
            )
        show_current, show_recommendation_path = trajectory_mode_visibility(
            self.config.overlay.trajectory_mode
        )
        if self.current and show_current:
            self._draw_trajectory(
                painter,
                self.current,
                QColor(self.config.overlay.current_color),
                self.config.overlay.line_width,
            )
        for index, trajectory in enumerate(self.recommendations):
            color = QColor(
                self.config.overlay.recommendation_colors[
                    index % len(self.config.overlay.recommendation_colors)
                ]
            )
            if show_recommendation_path:
                self._draw_trajectory(
                    painter, trajectory, color, self.config.overlay.line_width + 0.5
                )
            self._draw_recommendation_marker(painter, trajectory, color)
            if len(trajectory.points) > 1:
                point = trajectory.points[1]
                painter.setPen(color)
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.drawText(
                    QPointF(float(point[0] + 6), float(point[1] - 6)),
                    (
                        f"最佳点 覆盖{trajectory.unique_block_hits}块 "
                        f"碰撞{trajectory.block_hits}次 "
                        f"{trajectory.angle_deg:.1f}°"
                    ),
                )
        if self.config.overlay.show_locked_boxes:
            visual_scale = (
                max(0.70, min(2.75, self.board.width / 420.0))
                if self.board is not None
                else 1.0
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    QColor("#54FF9A"),
                    2.0 * visual_scale,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            for index, obstacle in enumerate(self.obstacles, start=1):
                rect = obstacle.rect
                painter.drawRect(
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                )
                painter.drawText(
                    QPointF(rect.left + 4, rect.top + 14),
                    (
                        f"LOCK {index} HP {obstacle.durability}"
                        if obstacle.durability is not None
                        else f"LOCK {index}"
                    ),
                )
        if self.debug_view:
            painter.setPen(QPen(QColor("#00FFFF"), 1, Qt.PenStyle.DashLine))
            if self.board:
                painter.drawRect(
                    self.board.left,
                    self.board.top,
                    self.board.width,
                    self.board.height,
                )
            painter.setPen(QPen(QColor("#FF4060"), 1))
            for obstacle in self.obstacles:
                rect = obstacle.rect
                painter.drawRect(
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                )
                painter.drawText(
                    QPointF(rect.left + 3, rect.top + 13),
                    f"BLOCK {obstacle.identifier}",
                )
        painter.restore()
        painter.setPen(QColor("#FFFFFF"))
        painter.setBrush(QColor(0, 0, 0, 145))
        painter.drawRoundedRect(10, max(10, self.height() - 34), min(620, self.width() - 20), 25, 7, 7)
        painter.drawText(20, self.height() - 16, self.status)


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("弹珠轨迹助手设置")
        layout = QFormLayout(self)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1.0, 30.0)
        self.radius.setDecimals(2)
        self.radius.setValue(config.physics.ball_radius)
        self.max_collisions = QSpinBox()
        self.max_collisions.setRange(5, 10)
        self.max_collisions.setValue(config.physics.max_collisions)
        self.advanced_simulation = QCheckBox()
        self.advanced_simulation.setChecked(config.physics.advanced_simulation)
        self.volley_count = QSpinBox()
        self.volley_count.setRange(1, 200)
        self.volley_count.setValue(config.physics.volley_count)
        self.reflection_bias = QDoubleSpinBox()
        self.reflection_bias.setRange(-10.0, 10.0)
        self.reflection_bias.setDecimals(2)
        self.reflection_bias.setValue(config.physics.reflection_bias_deg)
        self.origin_x = QDoubleSpinBox()
        self.origin_x.setRange(0.0, 1.0)
        self.origin_x.setSingleStep(0.01)
        self.origin_x.setValue(config.calibration.launch_origin_normalized[0])
        self.origin_y = QDoubleSpinBox()
        self.origin_y.setRange(0.0, 1.0)
        self.origin_y.setSingleStep(0.01)
        self.origin_y.setValue(config.calibration.launch_origin_normalized[1])
        self.boundary_offsets: list[QDoubleSpinBox] = []
        for value in config.calibration.boundary_offsets:
            field = QDoubleSpinBox()
            field.setRange(-30.0, 30.0)
            field.setDecimals(1)
            field.setValue(value)
            self.boundary_offsets.append(field)
        self.water_h_low = QSpinBox()
        self.water_h_low.setRange(0, 179)
        self.water_h_low.setValue(config.vision.water_hsv_low[0])
        self.water_h_high = QSpinBox()
        self.water_h_high.setRange(0, 179)
        self.water_h_high.setValue(config.vision.water_hsv_high[0])
        self.block_h_low = QSpinBox()
        self.block_h_low.setRange(0, 179)
        self.block_h_low.setValue(config.vision.block_hsv_low[0])
        self.block_h_high = QSpinBox()
        self.block_h_high.setRange(0, 179)
        self.block_h_high.setValue(config.vision.block_hsv_high[0])
        self.debug = QCheckBox()
        layout.addRow("无方块时后备半径（像素）", self.radius)
        layout.addRow("最大碰撞次数", self.max_collisions)
        layout.addRow("高级血量/多球模拟", self.advanced_simulation)
        layout.addRow("弹珠数识别失败时使用", self.volley_count)
        layout.addRow("反射角修正（度）", self.reflection_bias)
        layout.addRow("发射点 X（0–1）", self.origin_x)
        layout.addRow("发射点 Y（0–1）", self.origin_y)
        layout.addRow("左边界偏移", self.boundary_offsets[0])
        layout.addRow("右边界偏移", self.boundary_offsets[1])
        layout.addRow("上边界偏移", self.boundary_offsets[2])
        layout.addRow("下边界偏移", self.boundary_offsets[3])
        layout.addRow("水池色相下限", self.water_h_low)
        layout.addRow("水池色相上限", self.water_h_high)
        layout.addRow("方块色相下限", self.block_h_low)
        layout.addRow("方块色相上限", self.block_h_high)
        layout.addRow("显示识别框", self.debug)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def apply(self) -> None:
        self.config.physics.ball_radius = self.radius.value()
        self.config.physics.max_collisions = self.max_collisions.value()
        self.config.physics.advanced_simulation = (
            self.advanced_simulation.isChecked()
        )
        self.config.physics.volley_count = self.volley_count.value()
        self.config.physics.reflection_bias_deg = self.reflection_bias.value()
        self.config.calibration.launch_origin_normalized = [
            self.origin_x.value(),
            self.origin_y.value(),
        ]
        self.config.calibration.boundary_offsets = [
            field.value() for field in self.boundary_offsets
        ]
        self.config.vision.water_hsv_low[0] = self.water_h_low.value()
        self.config.vision.water_hsv_high[0] = self.water_h_high.value()
        self.config.vision.block_hsv_low[0] = self.block_h_low.value()
        self.config.vision.block_hsv_high[0] = self.block_h_high.value()
