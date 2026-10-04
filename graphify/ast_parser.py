"""
AST Code Parser for Graphify.
Deterministically parses Python source files, extracts modules, classes,
methods, functions, import dependencies, and call references into structured
graph nodes and relationships.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Set, Tuple


class ASTCodeVisitor(ast.NodeVisitor):
    """AST visitor collecting structural code symbols and relations."""

    def __init__(self, module_path: str, rel_filepath: str) -> None:
        self.module_path = module_path
        self.rel_filepath = rel_filepath
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self._current_class: Optional[str] = None
        self._current_scope: str = module_path

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append({
                "source": self.module_path,
                "target": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "import_module",
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source_mod = node.module or ""
        for alias in node.names:
            target_name = f"{source_mod}.{alias.name}" if source_mod else alias.name
            self.imports.append({
                "source": self.module_path,
                "target": target_name,
                "module": source_mod,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "import_from",
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.module_path}.{node.name}"
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._resolve_attr_name(base)}")

        docstring = ast.get_docstring(node) or ""
        self.classes.append({
            "id": class_id,
            "name": node.name,
            "module": self.module_path,
            "file": self.rel_filepath,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "bases": bases,
            "docstring": docstring.strip(),
        })

        prev_class = self._current_class
        prev_scope = self._current_scope
        self._current_class = node.name
        self._current_scope = class_id

        self.generic_visit(node)

        self._current_class = prev_class
        self._current_scope = prev_scope

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        func_name = getattr(node, "name", "")
        lineno = getattr(node, "lineno", 0)
        end_lineno = getattr(node, "end_lineno", lineno)
        is_method = self._current_class is not None

        if is_method:
            func_id = f"{self.module_path}.{self._current_class}.{func_name}"
            parent_id = f"{self.module_path}.{self._current_class}"
        else:
            func_id = f"{self.module_path}.{func_name}"
            parent_id = self.module_path

        args = []
        args_node = getattr(node, "args", None)
        if args_node:
            for arg in args_node.args:
                args.append(arg.arg)

        docstring = ast.get_docstring(node) or ""
        self.functions.append({
            "id": func_id,
            "name": func_name,
            "module": self.module_path,
            "parent_id": parent_id,
            "is_method": is_method,
            "class_name": self._current_class,
            "file": self.rel_filepath,
            "line": lineno,
            "end_line": end_lineno,
            "args": args,
            "docstring": docstring.strip(),
        })

        prev_scope = self._current_scope
        self._current_scope = func_id

        self.generic_visit(node)

        self._current_scope = prev_scope

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._resolve_call_name(node.func)
        if callee:
            self.calls.append({
                "caller": self._current_scope,
                "callee": callee,
                "line": node.lineno,
            })
        self.generic_visit(node)

    def _resolve_attr_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._resolve_attr_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return ""

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._resolve_attr_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None


class ASTCodeParser:
    """Parses an entire project directory and constructs raw AST graphs."""

    EXCLUDED_DIRS = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
        ".agents",
        "graphify-out",
        "violations",
    }

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)

    def parse_repository(self) -> Dict[str, Any]:
        """Scans the repository and returns structured nodes and edges."""
        modules: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        imports: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        docs: List[Dict[str, Any]] = []

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Prune excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.EXCLUDED_DIRS]

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(fpath, self.root_dir).replace("\\", "/")

                if fname.endswith(".py"):
                    mod_info, cls_list, fn_list, imp_list, call_list = self._parse_python_file(fpath, rel_path)
                    if mod_info:
                        modules.append(mod_info)
                        classes.extend(cls_list)
                        functions.extend(fn_list)
                        imports.extend(imp_list)
                        calls.extend(call_list)

                elif fname in ("README.md", "CONTRIBUTING.md", "PROJECT.md", "pyproject.toml", "requirements.txt", "LICENSE"):
                    docs.append({
                        "id": f"doc:{rel_path}",
                        "name": fname,
                        "file": rel_path,
                        "size": os.path.getsize(fpath),
                        "type": "documentation" if fname.endswith(".md") or fname == "LICENSE" else "configuration",
                    })

        return {
            "root_dir": self.root_dir,
            "modules": modules,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "calls": calls,
            "docs": docs,
        }

    def _parse_python_file(
        self, fpath: str, rel_path: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Compute module dot path
        mod_parts = []
        without_ext = os.path.splitext(rel_path)[0]
        for part in without_ext.split("/"):
            if part and part != "__init__":
                mod_parts.append(part)
        module_path = ".".join(mod_parts) if mod_parts else "__init__"

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=fpath)
        except Exception as e:
            return None, [], [], [], []

        docstring = ast.get_docstring(tree) or ""
        line_count = len(content.splitlines())

        module_info = {
            "id": module_path,
            "name": os.path.basename(rel_path),
            "file": rel_path,
            "docstring": docstring.strip(),
            "lines": line_count,
            "size": len(content),
        }

        visitor = ASTCodeVisitor(module_path=module_path, rel_filepath=rel_path)
        visitor.visit(tree)

        return module_info, visitor.classes, visitor.functions, visitor.imports, visitor.calls
