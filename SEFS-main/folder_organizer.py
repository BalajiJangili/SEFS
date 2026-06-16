"""Filesystem materialization of semantic clusters for SEFS."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path

import config


class FolderOrganizer:
    """Create and maintain semantic folders that point to files in `_raw`."""

    def __init__(self, root_dir: Path, pause_flag: threading.Event) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.pause_flag = pause_flag
        self.logger = logging.getLogger(self.__class__.__name__)

        self._reserved_dirs = {
            config.RAW_SUBDIR,
            config.INTERNAL_DIR,
            config.UNCATEGORIZED_DIR,
        }
        self._reserved_dirs_lower = {name.lower() for name in self._reserved_dirs}

    def reorganize(self, cluster_dict: dict[str, list[Path]]) -> None:
        """Apply cluster-to-folder mapping to the root directory."""
        self.pause_flag.set()
        try:
            normalized_clusters = self._normalize_cluster_dict(cluster_dict)
            if config.UNCATEGORIZED_DIR not in normalized_clusters:
                normalized_clusters[config.UNCATEGORIZED_DIR] = []

            existing_semantic_dirs = self._list_existing_semantic_dirs()
            target_names = set(normalized_clusters.keys())

            for folder_path in existing_semantic_dirs:
                if folder_path.name not in target_names:
                    self.logger.info("Removing stale semantic folder: %s", folder_path)
                    shutil.rmtree(folder_path, ignore_errors=True)

            for folder_name, files in normalized_clusters.items():
                self._rebuild_folder(folder_name, files)
        finally:
            self.pause_flag.clear()

    def _normalize_cluster_dict(self, cluster_dict: dict[str, list[Path]]) -> dict[str, list[Path]]:
        normalized: dict[str, list[Path]] = {}
        used_names: set[str] = set()

        for folder_name, paths in cluster_dict.items():
            fallback = config.UNCATEGORIZED_DIR if folder_name == config.UNCATEGORIZED_DIR else "Cluster"
            sanitized = config.sanitize_folder_name(folder_name, fallback=fallback)
            if (
                folder_name != config.UNCATEGORIZED_DIR
                and sanitized.lower() in self._reserved_dirs_lower
            ):
                sanitized = "Cluster"
            unique_name = self._dedupe_folder_name(sanitized, used_names)
            used_names.add(unique_name.lower())

            deduped_paths: list[Path] = []
            seen_paths: set[str] = set()
            for path in paths:
                resolved = Path(path).resolve(strict=False)
                key = str(resolved)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                if resolved.exists() and resolved.is_file():
                    deduped_paths.append(resolved)
                else:
                    self.logger.debug("Skipping missing source file during organize: %s", resolved)

            normalized[unique_name] = deduped_paths

        return normalized

    def _list_existing_semantic_dirs(self) -> list[Path]:
        semantic_dirs: list[Path] = []
        for child in self.root_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in self._reserved_dirs:
                continue
            if child.name.startswith("."):
                continue
            semantic_dirs.append(child)
        return semantic_dirs

    def _rebuild_folder(self, folder_name: str, file_paths: list[Path]) -> None:
        target_dir = self.root_dir / folder_name
        temp_dir = self.root_dir / f".{folder_name}.sefs_tmp"

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._populate_folder(temp_dir, file_paths)

            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)

            temp_dir.replace(target_dir)
            self.logger.info("Updated semantic folder: %s (%d files)", target_dir, len(file_paths))
        except Exception:
            self.logger.exception("Failed to rebuild semantic folder %s", folder_name)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _populate_folder(self, folder_dir: Path, file_paths: list[Path]) -> None:
        used_entry_names: set[str] = set()
        for source in file_paths:
            entry_name = self._build_unique_entry_name(source, used_entry_names)
            destination = folder_dir / entry_name
            self._create_link_or_copy(source, destination)

    def _build_unique_entry_name(self, source: Path, used_names: set[str]) -> str:
        base_name = source.name
        if base_name not in used_names:
            used_names.add(base_name)
            return base_name

        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
        stem = source.stem
        suffix = source.suffix
        candidate = f"{stem}_{digest}{suffix}"

        counter = 2
        while candidate in used_names:
            candidate = f"{stem}_{digest}_{counter}{suffix}"
            counter += 1

        used_names.add(candidate)
        return candidate

    def _create_link_or_copy(self, source: Path, destination: Path) -> None:
        source = source.resolve(strict=False)
        if destination.exists() or destination.is_symlink():
            destination.unlink(missing_ok=True)

        try:
            os.symlink(str(source), str(destination))
        except (OSError, NotImplementedError) as exc:
            if os.name == "nt":
                self.logger.warning(
                    "Symlink failed on Windows (%s). Falling back to copy: %s",
                    exc,
                    source,
                )
                shutil.copy2(source, destination)
            else:
                raise

    def _dedupe_folder_name(self, name: str, used_names: set[str]) -> str:
        if name.lower() not in used_names:
            return name

        index = 2
        while True:
            candidate = f"{name}_{index}"
            if candidate.lower() not in used_names:
                return candidate
            index += 1
