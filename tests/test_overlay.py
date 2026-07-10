from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QPushButton

from marble_aim.overlay import (
    RefreshButton,
    capture_to_overlay_scale,
    trajectory_mode_visibility,
)


def test_capture_pixels_scale_to_qt_logical_pixels():
    assert capture_to_overlay_scale(800, 450, 1200, 675) == (2 / 3, 2 / 3)
    assert capture_to_overlay_scale(800, 450, 0, 0) == (1.0, 1.0)


def test_control_panel_exposes_restart_and_exit_buttons():
    app = QApplication.instance() or QApplication([])
    panel = RefreshButton()
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    select_spy = QSignalSpy(panel.select_window_requested)
    mode_spy = QSignalSpy(panel.mode_cycle_requested)
    advanced_spy = QSignalSpy(panel.advanced_toggle_requested)
    exit_spy = QSignalSpy(panel.exit_requested)

    assert "重启识别" in buttons
    assert "选择窗口" in buttons
    assert "模式 A" in buttons
    assert "高级 OFF" in buttons
    assert "退出助手" in buttons

    buttons["选择窗口"].click()
    buttons["模式 A"].click()
    buttons["高级 OFF"].click()
    buttons["退出助手"].click()

    assert select_spy.count() == 1
    assert mode_spy.count() == 1
    assert advanced_spy.count() == 1
    assert exit_spy.count() == 1
    panel.close()
    app.processEvents()


def test_trajectory_modes_control_current_and_green_paths():
    assert trajectory_mode_visibility("A") == (True, False)
    assert trajectory_mode_visibility("B") == (False, False)
    assert trajectory_mode_visibility("C") == (True, True)
    assert trajectory_mode_visibility("D") == (False, True)


def test_control_panel_updates_advanced_simulation_state():
    app = QApplication.instance() or QApplication([])
    panel = RefreshButton("D", advanced_simulation=True)
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}

    assert "模式 D" in buttons
    assert "高级 ON" in buttons

    panel.set_advanced_simulation(False)
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    assert "高级 OFF" in buttons
    panel.close()
    app.processEvents()
