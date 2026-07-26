"""Compatibility entry point for the Protocol III agent.

The legacy UI-TARS implementation remains in agent.py, but this default path never imports it.
"""

from web_agent.cli import main


if __name__ == "__main__":
    main()
