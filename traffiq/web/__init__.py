"""
TRAFFIQ Web Submodule:
Flask dashboard server and background streaming engine.
"""

from traffiq.web.engine import DashboardEngine
from traffiq.web.server import create_app

__all__ = [
    "create_app",
    "DashboardEngine",
]
