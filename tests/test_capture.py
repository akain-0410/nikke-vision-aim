from marble_aim import capture


class WindowStub:
    def winId(self) -> int:
        return 123


def test_native_overlay_uses_capture_physical_pixels(monkeypatch):
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        capture.win32gui,
        "SetWindowPos",
        lambda *arguments: calls.append(arguments),
        raising=False,
    )
    geometry = capture.WindowGeometry(7, "game", 320, 180, 1280, 720)

    capture.set_native_window_geometry(WindowStub(), geometry)

    assert calls == [
        (
            123,
            capture.win32con.HWND_TOPMOST,
            320,
            180,
            1280,
            720,
            capture.win32con.SWP_NOACTIVATE
            | capture.win32con.SWP_NOOWNERZORDER,
        )
    ]
