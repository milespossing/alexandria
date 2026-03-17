"""File discovery — find files to index using fd (respects .gitignore)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def discover_files(
    root: Path,
    follow_symlinks: bool = False,
    ignore_patterns: list[str] | None = None,
) -> list[Path]:
    """Discover files to index using fd.

    Uses fd for gitignore-aware, fast file discovery with semantics similar
    to the fd tool. Falls back to a pure-Python walk if fd is not available.

    Args:
        root: Root directory to search.
        follow_symlinks: Whether to follow symbolic links.
        ignore_patterns: Extra glob patterns to exclude (same syntax as
            ``fd --exclude`` / ``.gitignore``).

    Returns:
        List of absolute file paths.
    """
    extra = ignore_patterns or []
    try:
        return _discover_with_fd(root, follow_symlinks, extra)
    except FileNotFoundError:
        return _discover_fallback(root, follow_symlinks, extra)


def _discover_with_fd(
    root: Path,
    follow_symlinks: bool,
    ignore_patterns: list[str],
) -> list[Path]:
    """Use fd for file discovery."""
    cmd = [
        "fd",
        "--type",
        "f",
        "--hidden",  # include dotfiles like .eslintrc
        "--exclude",
        ".git",  # always exclude .git directory
    ]

    for pattern in ignore_patterns:
        cmd.extend(["--exclude", pattern])

    if follow_symlinks:
        cmd.append("--follow")

    # fd outputs one path per line, relative to the search directory
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    files = []
    for line in result.stdout.strip().splitlines():
        if line:
            files.append(root / line)

    return files


def _discover_fallback(
    root: Path,
    follow_symlinks: bool,
    ignore_patterns: list[str],
) -> list[Path]:
    """Pure-Python fallback when fd is not available.

    Respects .gitignore using pathspec library.
    """
    import pathspec

    # Load .gitignore patterns
    gitignore_path = root / ".gitignore"
    patterns: list[str] = []

    # Always ignore .git
    patterns.append(".git/")

    if gitignore_path.exists():
        patterns.extend(gitignore_path.read_text().splitlines())

    # Append extra ignore patterns from .alexandria.yml
    patterns.extend(ignore_patterns)

    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() and not follow_symlinks:
            continue
        if not path.is_file():
            continue
        # Check against gitignore
        rel = str(path.relative_to(root))
        if spec.match_file(rel):
            continue
        files.append(path)

    return files
