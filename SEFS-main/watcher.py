"""Filesystem event watcher for SEFS raw files."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

import config


class _RawEventHandler(FileSystemEventHandler):
    """Internal watchdog handler forwarding events to SEFSWatcher."""

    def __init__(self, watcher: "SEFSWatcher") -> None:
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        self._watcher._handle_created_or_modified(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._watcher._handle_created_or_modified(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._watcher._handle_deleted(event)

    def on_moved(self, event: FileSystemMovedEvent) -> None:
        self._watcher._handle_moved(event)


class SEFSWatcher:
    """Monitor `_raw` for supported file changes and trigger semantic updates."""

    def __init__(
        self,
        root_dir: Path,
        semantic_engine: Any,
        pause_flag: threading.Event,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.raw_dir = self.root_dir / config.RAW_SUBDIR
        self.semantic_engine = semantic_engine
        self.pause_flag = pause_flag

        self.logger = logging.getLogger(self.__class__.__name__)
        # fsevents can be unreliable in restricted macOS environments; polling is stable.
        self.observer = PollingObserver() if sys.platform == "darwin" else Observer()
        self.handler = _RawEventHandler(self)

        self._debounce_lock = threading.Lock()
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._started = False

    def start(self) -> None:
        """Start watching the raw directory recursively."""
        if self._started:
            return

        if not self.raw_dir.exists():
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.observer.schedule(self.handler, str(self.raw_dir), recursive=True)
            self.observer.start()
        except Exception as exc:
            self.logger.warning("Native observer failed (%s). Falling back to polling observer.", exc)
            self.observer = PollingObserver()
            self.observer.schedule(self.handler, str(self.raw_dir), recursive=True)
            self.observer.start()
        self._started = True
        self.logger.info("Watcher started on %s", self.raw_dir)

    def stop(self) -> None:
        """Stop watching and cancel pending debounce timers."""
        if not self._started:
            return

        self.observer.stop()
        self.observer.join(timeout=5)

        with self._debounce_lock:
            timers = list(self._debounce_timers.values())
            self._debounce_timers.clear()

        for timer in timers:
            timer.cancel()

        self._started = False
        self.logger.info("Watcher stopped")

    def _handle_created_or_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or self.pause_flag.is_set():
            return

        path = Path(event.src_path).resolve(strict=False)
        if self._should_ignore(path, require_supported=True):
            return

        self._schedule_debounced_process(path)

    def _handle_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or self.pause_flag.is_set():
            return

        path = Path(event.src_path).resolve(strict=False)
        if self._should_ignore(path, require_supported=True):
            return

        self._cancel_debounce(path)
        try:
            self.semantic_engine.remove_file(path, trigger_recluster=True)
        except Exception:
            self.logger.exception("Failed to remove file from semantic index: %s", path)

    def _handle_moved(self, event: FileSystemMovedEvent) -> None:
        if event.is_directory or self.pause_flag.is_set():
            return

        source = Path(event.src_path).resolve(strict=False)
        destination = Path(event.dest_path).resolve(strict=False)

        if not self._should_ignore(source, require_supported=True):
            self._cancel_debounce(source)
            try:
                self.semantic_engine.remove_file(source, trigger_recluster=False)
            except Exception:
                self.logger.exception("Failed to remove moved source file from index: %s", source)

        if not self._should_ignore(destination, require_supported=True):
            self._schedule_debounced_process(destination)
        else:
            try:
                self.semantic_engine.recluster()
            except Exception:
                self.logger.exception("Failed to trigger recluster after move: %s -> %s", source, destination)

    def _schedule_debounced_process(self, path: Path) -> None:
        key = str(path.resolve(strict=False))
        with self._debounce_lock:
            existing = self._debounce_timers.get(key)
            if existing:
                existing.cancel()

            timer = threading.Timer(
                config.DEBOUNCE_SECONDS,
                self._process_after_debounce,
                args=(key,),
            )
            timer.daemon = True
            self._debounce_timers[key] = timer
            timer.start()

    def _cancel_debounce(self, path: Path) -> None:
        key = str(path.resolve(strict=False))
        with self._debounce_lock:
            timer = self._debounce_timers.pop(key, None)
        if timer:
            timer.cancel()

    def _process_after_debounce(self, path_key: str) -> None:
        with self._debounce_lock:
            self._debounce_timers.pop(path_key, None)

        if self.pause_flag.is_set():
            return

        path = Path(path_key)
        if self._should_ignore(path, require_supported=True):
            return

        if not path.exists() or not path.is_file():
            return

        try:
            self.semantic_engine.process_file(path, trigger_recluster=True)
        except Exception:
            self.logger.exception("Failed to process file after debounce: %s", path)

    def _should_ignore(self, path: Path, require_supported: bool) -> bool:
        try:
            path.relative_to(self.raw_dir)
        except ValueError:
            return True

        if config.is_temporary_file(path):
            return True

        if require_supported and not config.is_supported_file(path):
            return True

        return False
