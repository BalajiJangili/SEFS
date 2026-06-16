"""SEFS entry point: initialize components and run runtime orchestration."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

import config
from dashboard import SEFSDashboard
from extractor import TextExtractor
from folder_organizer import FolderOrganizer
from semantic_engine import SemanticEngine
from watcher import SEFSWatcher


LOGGER = logging.getLogger("sefs")



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SEFS process."""
    parser = argparse.ArgumentParser(description="Semantic Entropy File System (SEFS)")
    parser.add_argument(
        "--root",
        required=True,
        type=str,
        help="Root folder to manage. SEFS watches <root>/_raw.",
    )
    return parser.parse_args()



def configure_logging() -> None:
    """Configure process-wide logging behavior."""
    logging.basicConfig(
        level=config.get_log_level(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def ensure_structure(root_dir: Path) -> tuple[Path, Path, Path]:
    """Create required runtime directories if they do not exist."""
    root_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root_dir / config.RAW_SUBDIR
    internal_dir = root_dir / config.INTERNAL_DIR
    uncategorized_dir = root_dir / config.UNCATEGORIZED_DIR

    raw_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)
    uncategorized_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = internal_dir / config.CONFIG_FILE_NAME
    if not runtime_config_path.exists():
        runtime_config_path.write_text(
            yaml.safe_dump(
                {
                    "root_dir": str(root_dir),
                    "raw_dir": str(raw_dir),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return raw_dir, internal_dir, uncategorized_dir



def acquire_lock(internal_dir: Path) -> Path:
    """Create a lock file to prevent concurrent SEFS instances on one root."""
    lock_path = internal_dir / config.LOCK_FILE_NAME
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"SEFS lock exists at {lock_path}. Another instance may already be running."
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.write(f"started_utc={datetime.now(timezone.utc).isoformat()}\n")

    return lock_path



def release_lock(lock_path: Path | None) -> None:
    """Delete lock file on shutdown."""
    if lock_path and lock_path.exists():
        lock_path.unlink(missing_ok=True)



def iter_supported_files(raw_dir: Path) -> Iterable[Path]:
    """Yield supported files from `_raw`, recursively and deterministically."""
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if config.is_temporary_file(path):
            continue
        if not config.is_supported_file(path):
            continue
        yield path



def register_signal_handlers(stop_event: threading.Event) -> None:
    """Register SIGINT/SIGTERM to trigger graceful process shutdown."""

    def _handle_signal(signum: int, _frame: object) -> None:
        LOGGER.info("Received signal %s. Shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)



def main() -> int:
    """Program entry point for SEFS runtime."""
    configure_logging()
    args = parse_args()

    root_dir = Path(args.root).resolve()
    config.ROOT_DIR = root_dir
    raw_dir, internal_dir, _uncategorized_dir = ensure_structure(root_dir)

    lock_path: Path | None = None
    pause_flag = threading.Event()

    extractor = TextExtractor()
    organizer = FolderOrganizer(root_dir=root_dir, pause_flag=pause_flag)
    engine = SemanticEngine(root_dir=root_dir, extractor=extractor, folder_organizer=organizer)

    def _on_file_renamed(old_path: str, new_path: str) -> None:
        old_file = Path(old_path).resolve(strict=False)
        new_file = Path(new_path).resolve(strict=False)

        # Rename means old id should disappear and new id should be upserted.
        engine.remove_file(old_file, trigger_recluster=False)
        engine.process_file(new_file, trigger_recluster=True)

    def _on_file_deleted(path: str) -> None:
        engine.remove_file(Path(path).resolve(strict=False), trigger_recluster=True)

    dashboard = SEFSDashboard(
        snapshot_provider=engine.get_dashboard_snapshot,
        query_handler=engine.answer_query,
        override_handler=engine.set_manual_override,
        rename_handler=_on_file_renamed,
        delete_handler=_on_file_deleted,
    )
    watcher = SEFSWatcher(root_dir=root_dir, semantic_engine=engine, pause_flag=pause_flag)

    stop_event = threading.Event()

    try:
        lock_path = acquire_lock(internal_dir)

        LOGGER.info("Running initial scan of %s", raw_dir)
        for file_path in iter_supported_files(raw_dir):
            engine.process_file(file_path, trigger_recluster=False)

        engine.recluster()

        watcher.start()
        dashboard.start()
        register_signal_handlers(stop_event)

        LOGGER.info(
            "SEFS running. Monitoring: %s | Dashboard: http://%s:%s",
            raw_dir,
            config.DASHBOARD_HOST,
            config.DASHBOARD_PORT,
        )

        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)

        return 0
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1
    except Exception:
        LOGGER.exception("Fatal error in SEFS runtime")
        return 1
    finally:
        watcher.stop()
        dashboard.stop()
        engine.close()
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
