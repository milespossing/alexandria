"""Tests for project config loading and ignore-pattern wiring.

These are unit tests — they do not require Qdrant or Ollama.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from alexandria.config import Config, load_project_config
from alexandria.discovery import discover_files


class TestLoadProjectConfig:
    """Verify .alexandria.yml parsing and merging."""

    def test_no_config_file(self, tmp_path: Path) -> None:
        """When .alexandria.yml is absent, config is returned unchanged."""
        config = Config()
        result = load_project_config(tmp_path, config)
        assert result is config
        assert result.ignore_patterns == []

    def test_empty_config_file(self, tmp_path: Path) -> None:
        """An empty .alexandria.yml should not crash or mutate config."""
        (tmp_path / ".alexandria.yml").write_text("")
        config = Config()
        result = load_project_config(tmp_path, config)
        assert result.ignore_patterns == []

    def test_ignore_patterns_loaded(self, tmp_path: Path) -> None:
        """Ignore patterns from the YAML should be merged into config."""
        (tmp_path / ".alexandria.yml").write_text(
            textwrap.dedent("""\
                ignore:
                  - "*.min.js"
                  - "vendor/"
                  - "dist/**"
            """)
        )
        config = Config()
        load_project_config(tmp_path, config)
        assert config.ignore_patterns == ["*.min.js", "vendor/", "dist/**"]

    def test_non_list_ignore_is_ignored(self, tmp_path: Path) -> None:
        """If 'ignore' is not a list, it should be silently skipped."""
        (tmp_path / ".alexandria.yml").write_text("ignore: true\n")
        config = Config()
        load_project_config(tmp_path, config)
        assert config.ignore_patterns == []

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Malformed YAML should log a warning but not crash."""
        (tmp_path / ".alexandria.yml").write_text(": : : bad yaml [[[")
        config = Config()
        result = load_project_config(tmp_path, config)
        assert result is config
        assert result.ignore_patterns == []


class TestDiscoverFilesIgnorePatterns:
    """Verify that ignore_patterns are respected during file discovery."""

    def _make_repo(self, tmp_path: Path) -> Path:
        """Create a minimal file tree for testing discovery."""
        # Create some files
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def util(): pass")
        (tmp_path / "bundle.min.js").write_text("var x=1;")

        vendor = tmp_path / "vendor"
        vendor.mkdir()
        (vendor / "lib.py").write_text("# vendored")

        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "output.js").write_text("// built")

        return tmp_path

    def test_no_ignore_patterns(self, tmp_path: Path) -> None:
        """Without ignore patterns, all files should be discovered."""
        root = self._make_repo(tmp_path)
        files = discover_files(root, follow_symlinks=False, ignore_patterns=[])
        names = {f.name for f in files}
        assert "app.py" in names
        assert "bundle.min.js" in names
        assert "lib.py" in names
        assert "output.js" in names

    def test_glob_ignore_pattern(self, tmp_path: Path) -> None:
        """A glob pattern like '*.min.js' should exclude matching files."""
        root = self._make_repo(tmp_path)
        files = discover_files(root, follow_symlinks=False, ignore_patterns=["*.min.js"])
        names = {f.name for f in files}
        assert "app.py" in names
        assert "bundle.min.js" not in names

    def test_directory_ignore_pattern(self, tmp_path: Path) -> None:
        """A directory pattern like 'vendor' should exclude its contents."""
        root = self._make_repo(tmp_path)
        files = discover_files(root, follow_symlinks=False, ignore_patterns=["vendor"])
        names = {f.name for f in files}
        assert "app.py" in names
        assert "lib.py" not in names

    def test_multiple_ignore_patterns(self, tmp_path: Path) -> None:
        """Multiple patterns should all be applied."""
        root = self._make_repo(tmp_path)
        files = discover_files(
            root,
            follow_symlinks=False,
            ignore_patterns=["*.min.js", "vendor", "dist"],
        )
        names = {f.name for f in files}
        assert names == {"app.py", "utils.py"}
