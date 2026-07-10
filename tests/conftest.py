from __future__ import annotations

import os
import sys
from types import ModuleType


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if sys.platform != "win32":
    win32con = ModuleType("win32con")
    win32con.HWND_TOPMOST = -1
    win32con.SWP_NOACTIVATE = 0x0010
    win32con.SWP_NOOWNERZORDER = 0x0200
    win32con.SWP_NOSIZE = 0x0001
    win32con.VK_F7 = 0x76
    win32con.VK_F8 = 0x77
    win32con.VK_F9 = 0x78
    win32con.VK_F10 = 0x79
    win32con.VK_F11 = 0x7A
    win32con.VK_F12 = 0x7B
    sys.modules.setdefault("win32con", win32con)
    sys.modules.setdefault("win32gui", ModuleType("win32gui"))
