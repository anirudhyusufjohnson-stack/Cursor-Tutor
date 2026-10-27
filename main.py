"""Cursor Tutor - hover over anything on your screen and the app explains it, using AI algorithms
that run on your own computer (no online API, no account, no internet needed)."""
import argparse
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description="Cursor Tutor")
    parser.add_argument("--selftest", action="store_true", help="check all the AI parts on made-up screens and exit")
    parser.add_argument("--tune", action="store_true", help="tune the agent's settings with the genetic algorithm and exit")
    args = parser.parse_args()

    if args.tune:
        import tune

        sys.argv = [sys.argv[0]]
        tune.main()
        return

    if args.selftest:
        import selftest

        sys.exit(selftest.run())

    # Make screen coordinates match real pixels on high-DPI Windows displays.
    if platform.system() == "Windows":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    from ai_tutor import Tutor
    from config import cfg
    from ui import TutorApp

    tutor = Tutor(cfg)
    ok, msg = tutor.check()
    if not ok:  # the app still opens and shows this message too
        print("Note:", msg)
    TutorApp(cfg, tutor).run()


if __name__ == "__main__":
    main()
