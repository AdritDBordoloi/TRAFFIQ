"""Tier 1: Package Importability & Clean Architecture Tests.
Verifies that all submodules under traffiq can be imported cleanly without syntax errors,
circular dependencies, or missing external dependencies.
"""

import sys
import unittest
import importlib

PACKAGE_MODULES = [
    "traffiq",
    "traffiq.config",
    "traffiq.config.settings",
    "traffiq.database",
    "traffiq.database.connection",
    "traffiq.database.repository",
    "traffiq.database.migration",
    "traffiq.vision",
    "traffiq.vision.camera",
    "traffiq.vision.detector",
    "traffiq.vision.tracker",
    "traffiq.vision.tripwire",
    "traffiq.enforcement",
    "traffiq.enforcement.signal",
    "traffiq.enforcement.violation",
    "traffiq.enforcement.ocr",
    "traffiq.enforcement.challan",
    "traffiq.cli",
    "traffiq.cli.app",
]


class TestPackageImports(unittest.TestCase):
    """Verifies that all modules under traffiq can be imported cleanly."""

    def test_import_root_package(self):
        """Verifies traffiq package root imports and exposes __version__."""
        mod = importlib.import_module("traffiq")
        self.assertIsNotNone(mod)
        self.assertTrue(hasattr(mod, "__version__"), "traffiq root package must define __version__")

    def test_import_config_submodules(self):
        """Verifies traffiq.config and its settings module import cleanly."""
        for mod_name in ["traffiq.config", "traffiq.config.settings"]:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed importing {mod_name}")

    def test_import_database_submodules(self):
        """Verifies traffiq.database submodules import cleanly."""
        for mod_name in [
            "traffiq.database",
            "traffiq.database.connection",
            "traffiq.database.repository",
            "traffiq.database.migration",
        ]:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed importing {mod_name}")

    def test_import_vision_submodules(self):
        """Verifies traffiq.vision submodules import cleanly."""
        for mod_name in [
            "traffiq.vision",
            "traffiq.vision.camera",
            "traffiq.vision.detector",
            "traffiq.vision.tracker",
            "traffiq.vision.tripwire",
        ]:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed importing {mod_name}")

    def test_import_enforcement_submodules(self):
        """Verifies traffiq.enforcement submodules import cleanly."""
        for mod_name in [
            "traffiq.enforcement",
            "traffiq.enforcement.signal",
            "traffiq.enforcement.violation",
            "traffiq.enforcement.ocr",
            "traffiq.enforcement.challan",
        ]:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed importing {mod_name}")

    def test_import_cli_submodules(self):
        """Verifies traffiq.cli and main application launcher import cleanly."""
        for mod_name in ["traffiq.cli", "traffiq.cli.app"]:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed importing {mod_name}")

    def test_no_circular_imports(self):
        """Ensures modules can be reloaded in arbitrary order without circular import deadlocks."""
        # Purge traffiq from sys.modules temporarily to test fresh import sequence
        to_purge = [k for k in sys.modules if k.startswith("traffiq")]
        saved_modules = {k: sys.modules[k] for k in to_purge}
        
        try:
            for k in to_purge:
                del sys.modules[k]

            # Import leaf modules before root packages
            leaf_modules = [
                "traffiq.config.settings",
                "traffiq.database.repository",
                "traffiq.vision.tripwire",
                "traffiq.enforcement.signal",
                "traffiq.cli.app",
            ]
            for mod_name in leaf_modules:
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod)
        finally:
            # Restore module cache
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
