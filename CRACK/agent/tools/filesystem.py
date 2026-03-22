"""
Filesystem tools: read_file, search_repo, list_directory.

These operate on the locally checked-out repository. File paths are sanitized
to prevent directory traversal outside the repo root.
"""

import os
import subprocess
from pathlib import Path
from typing import Callable

from .base import ToolProvider, ToolContext


def _sanitize_path(repo_path: str, relative_path: str) -> str:
    """
    Resolve a relative path against the repo root and ensure it doesn't escape.

    Raises ValueError if the resolved path is outside the repo.
    """
    repo = Path(repo_path).resolve()
    target = (repo / relative_path).resolve()
    try:
        target.relative_to(repo)
    except ValueError:
        raise ValueError(f"Path '{relative_path}' resolves outside the repository.")
    return str(target)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, showing {max_chars}/{len(text)} chars]"


class FilesystemToolProvider(ToolProvider):
    """Provides tools for reading files, searching code, and exploring repo structure."""

    def get_tools(self) -> list[Callable]:
        repo_path = self.ctx.repo_path
        max_chars = self.ctx.max_output_chars

        def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
            """Read a file from the repository. Returns the file contents with line numbers.

            Args:
                path: Relative path to the file within the repository.
                start_line: First line to read (1-indexed, inclusive). Defaults to 1.
                end_line: Last line to read (1-indexed, inclusive). If None, reads to end of file.
            """
            try:
                full_path = _sanitize_path(repo_path, path)
            except ValueError as e:
                return str(e)

            if not os.path.isfile(full_path):
                return f"Error: '{path}' is not a file or does not exist."

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as e:
                return f"Error reading file: {e}"

            total_lines = len(lines)
            start = max(1, start_line) - 1  # Convert to 0-indexed
            end = total_lines if end_line is None else min(end_line, total_lines)
            selected = lines[start:end]

            numbered = [f"{i}: {line.rstrip()}" for i, line in enumerate(selected, start=start + 1)]
            result = "\n".join(numbered)
            header = f"# {path} ({total_lines} lines total, showing {start + 1}-{start + len(selected)})\n"
            return _truncate(header + result, max_chars)

        def search_repo(
            pattern: str,
            file_glob: str | None = None,
            max_results: int = 30,
        ) -> str:
            """Search the repository for a text or regex pattern using ripgrep.

            Args:
                pattern: The search pattern (supports regex).
                file_glob: Optional glob to filter files, e.g. '*.py' or 'src/**/*.ts'.
                max_results: Maximum number of matching lines to return. Defaults to 30.
            """
            cmd = [
                "rg",
                "--line-number",
                "--no-heading",
                "--color=never",
                f"--max-count={max_results}",
                "--max-columns=200",
                "--max-columns-preview",
            ]
            if file_glob:
                cmd.extend(["--glob", file_glob])
            cmd.extend(["--", pattern])
            cmd.append(repo_path)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=repo_path,
                )
            except FileNotFoundError:
                # Fallback to grep if rg is not installed
                cmd = ["grep", "-rn", "--include", file_glob or "*", "--", pattern, repo_path]
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30
                    )
                except Exception as e:
                    return f"Error: search failed: {e}"
            except Exception as e:
                return f"Error: search failed: {e}"

            output = result.stdout.strip()
            if not output:
                return "No matches found."

            # Strip the repo_path prefix from results for cleaner output
            repo_prefix = repo_path.rstrip("/") + "/"
            output = output.replace(repo_prefix, "")
            return _truncate(output, max_chars)

        def list_directory(path: str = ".") -> str:
            """List the contents of a directory in the repository.

            Args:
                path: Relative path to the directory. Defaults to the repo root.
            """
            try:
                full_path = _sanitize_path(repo_path, path)
            except ValueError as e:
                return str(e)

            if not os.path.isdir(full_path):
                return f"Error: '{path}' is not a directory or does not exist."

            try:
                entries = sorted(os.listdir(full_path))
            except Exception as e:
                return f"Error listing directory: {e}"

            lines = []
            for entry in entries:
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    lines.append(f"  {entry}/")
                else:
                    size = os.path.getsize(entry_path)
                    lines.append(f"  {entry} ({size} bytes)")

            header = f"# {path}/ ({len(entries)} entries)\n"
            return _truncate(header + "\n".join(lines), max_chars)

        return [read_file, search_repo, list_directory]
