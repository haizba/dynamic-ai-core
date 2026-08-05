from __future__ import annotations

import traceback

import run


def pause() -> None:
    try:
        input("\n按回车键关闭窗口……")
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        run.main()
    except Exception:
        print("\n运行失败：")
        traceback.print_exc()
    finally:
        pause()
