"""Repository-root import shim for the editable rl_base package."""

from pathlib import Path

_pkg_dir = Path(__file__).with_name("rl_base")
__path__ = [str(_pkg_dir)]
__file__ = str(_pkg_dir / "__init__.py")
exec(compile((_pkg_dir / "__init__.py").read_text(encoding="utf-8"), __file__, "exec"))
