"""Core semantic indexing, clustering, and dashboard state management for SEFS."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.config import Settings

import config
from extractor import TextExtractor

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - import is environment-dependent
    umap = None


class SemanticEngine:
    """Manage extraction, embeddings, clustering, and view state for SEFS."""

    def __init__(
        self,
        root_dir: Path,
        extractor: TextExtractor,
        folder_organizer: Any,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.extractor = extractor
        self.folder_organizer = folder_organizer

        self.logger = logging.getLogger(self.__class__.__name__)
        self._engine_lock = threading.RLock()
        self._snapshot_lock = threading.Lock()

        self._model: SentenceTransformer | None = None
        self._model_unavailable = False
        self._hashing_vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._uncategorizable: dict[str, dict[str, Any]] = {}
        self._event_history: list[dict[str, Any]] = []
        self._timeline_history: list[dict[str, Any]] = []
        self._manual_overrides: dict[str, str] = {}
        self._cluster_lookup_cache: dict[str, str] = {}
        self._current_cluster_sizes: dict[str, int] = {}

        self.vector_store_dir = self.root_dir / config.INTERNAL_DIR / config.VECTOR_STORE_SUBDIR
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._overrides_path = self.root_dir / config.INTERNAL_DIR / config.OVERRIDES_FILE_NAME

        # Disable telemetry hooks to avoid noisy runtime errors from posthog version mismatches.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
        os.environ.setdefault("POSTHOG_DISABLED", "1")
        self._patch_posthog_capture()
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)
        self._load_manual_overrides()

        self._dashboard_snapshot: dict[str, Any] = {
            "points": [],
            "stats": {
                "total_files": 0,
                "cluster_count": 0,
                "uncategorized_count": 0,
                "last_update_iso": datetime.now(timezone.utc).isoformat(),
            },
            "ambiguities": {},
            "duplicates": [],
            "similarity_edges": [],
            "cluster_summaries": {},
            "recent_events": [],
            "timeline": [],
            "empty_message": "No files indexed yet. Drop supported files into _raw/.",
        }

        self._ollama_enabled = os.getenv(config.OLLAMA_ENABLE_ENV, "0") == "1"
        self._ollama_model = os.getenv(config.OLLAMA_MODEL_ENV, config.DEFAULT_OLLAMA_MODEL)
        self._ollama_timeout = int(
            os.getenv(
                config.OLLAMA_TIMEOUT_ENV,
                str(config.DEFAULT_OLLAMA_TIMEOUT_SECONDS),
            )
        )
        self._ollama_path = self._find_ollama_executable()
        rag_env = os.getenv(config.OLLAMA_RAG_ENABLE_ENV)
        if rag_env is None:
            self._ollama_rag_enabled = self._ollama_path is not None
        else:
            self._ollama_rag_enabled = rag_env == "1"
        self._ollama_rag_model = os.getenv(config.OLLAMA_RAG_MODEL_ENV, self._ollama_model)
        self._ollama_rag_timeout = int(
            os.getenv(
                config.OLLAMA_RAG_TIMEOUT_ENV,
                str(config.DEFAULT_OLLAMA_RAG_TIMEOUT_SECONDS),
            )
        )
        provider = os.getenv(config.RAG_PROVIDER_ENV, config.DEFAULT_RAG_PROVIDER).strip().lower()
        if provider not in {"auto", "ollama", "huggingface"}:
            provider = config.DEFAULT_RAG_PROVIDER
        self._rag_provider = provider
        self._hf_local_only = os.getenv(config.HF_LOCAL_ONLY_ENV, "1") != "0"
        self._hf_rag_model = os.getenv(config.HF_RAG_MODEL_ENV, config.DEFAULT_HF_RAG_MODEL)
        self._hf_rag_max_new_tokens = int(
            os.getenv(
                config.HF_RAG_MAX_NEW_TOKENS_ENV,
                str(config.DEFAULT_HF_RAG_MAX_NEW_TOKENS),
            )
        )
        self._hf_rag_pipeline: Any | None = None
        self._hf_rag_unavailable = False

    def process_file(self, filepath: Path, trigger_recluster: bool = True) -> None:
        """Extract, embed, and upsert a file; then optionally recluster."""
        path = Path(filepath).resolve(strict=False)
        file_id = self._file_id(path)

        if not path.exists() or not path.is_file():
            self.logger.warning("Skipping non-existent file: %s", path)
            return

        extracted_text = self.extractor.extract(path)
        cleaned_text = extracted_text.strip()
        metadata = self._build_metadata(path, cleaned_text)
        old_embedding: np.ndarray | None = None
        new_embedding: np.ndarray | None = None
        old_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
        event_action = "modified"

        with self._engine_lock:
            old_embedding = self._get_existing_embedding(file_id)
            if old_embedding is None:
                event_action = "added"

            # Spec-aligned behavior: do not embed files with insufficient extracted content.
            if len(cleaned_text) < config.MIN_TEXT_LENGTH:
                self._uncategorizable[file_id] = metadata
                try:
                    self._collection.delete(ids=[file_id])
                except Exception:
                    self.logger.debug("No existing embedding to delete for %s", path)
                if old_embedding is None:
                    event_action = "short_text_skipped"
                else:
                    event_action = "short_text_demoted"
            else:
                embedding = self._encode_text(cleaned_text)
                new_embedding = np.asarray(embedding, dtype=float)
                self._collection.upsert(
                    ids=[file_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[cleaned_text],
                )
                self._uncategorizable.pop(file_id, None)

            if trigger_recluster:
                self.recluster()

        if trigger_recluster:
            new_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
            drift = self._semantic_drift(old_embedding, new_embedding)
            self._record_event(
                event_type=event_action,
                file_id=file_id,
                old_cluster=old_cluster,
                new_cluster=new_cluster,
                semantic_drift=drift,
                detail=self._format_semantic_diff_detail(path.name, old_cluster, new_cluster, drift),
            )

    def remove_file(self, filepath: Path, trigger_recluster: bool = True) -> None:
        """Remove a file from all indexes; then optionally recluster."""
        path = Path(filepath).resolve(strict=False)
        file_id = self._file_id(path)
        old_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
        old_embedding: np.ndarray | None = None

        with self._engine_lock:
            old_embedding = self._get_existing_embedding(file_id)
            try:
                self._collection.delete(ids=[file_id])
            except Exception:
                self.logger.debug("Embedding delete ignored for missing id: %s", file_id)
            self._uncategorizable.pop(file_id, None)

            if trigger_recluster:
                self.recluster()

        if trigger_recluster:
            self._record_event(
                event_type="deleted",
                file_id=file_id,
                old_cluster=old_cluster,
                new_cluster="Deleted",
                semantic_drift=self._semantic_drift(old_embedding, None),
                detail=f"{path.name} was removed from the semantic index.",
            )

    def recluster(self) -> None:
        """Recompute clusters, update folders, and publish dashboard snapshot."""
        with self._engine_lock:
            self._prune_missing_uncategorizable()
            self._prune_missing_overrides()

            result = self._collection.get(include=["embeddings", "metadatas", "documents"])
            ids = result.get("ids") or []
            embeddings_list = result.get("embeddings") or []
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []

            embeddings = np.array(embeddings_list, dtype=float) if embeddings_list else np.empty((0, 0))
            embedded_count = len(ids)

            labels = np.full(embedded_count, -1, dtype=int)
            label_to_folder: dict[int, str] = {}
            normalized_embeddings = (
                self._normalize_embeddings(embeddings) if embedded_count > 0 else embeddings
            )

            if embedded_count >= 3:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
                        min_samples=config.HDBSCAN_MIN_SAMPLES,
                        metric=config.HDBSCAN_METRIC,
                        cluster_selection_method="leaf",
                        allow_single_cluster=True,
                    )
                    labels = clusterer.fit_predict(normalized_embeddings)
                except Exception as exc:
                    self.logger.warning("HDBSCAN failed; falling back to Uncategorized: %s", exc)
                    labels = np.full(embedded_count, -1, dtype=int)
            elif embedded_count == 2:
                pair_similarity = float(np.dot(normalized_embeddings[0], normalized_embeddings[1]))
                if pair_similarity >= config.PAIR_CLUSTER_SIMILARITY_THRESHOLD:
                    labels = np.array([0, 0], dtype=int)

            if embedded_count >= 2 and np.all(labels == -1):
                labels = self._cluster_by_similarity_fallback(
                    normalized_embeddings,
                    threshold=config.SIMILARITY_FALLBACK_THRESHOLD,
                )
            if embedded_count >= 2 and np.all(labels == -1):
                labels = self._cluster_by_tfidf_fallback(
                    [str(doc or "") for doc in documents],
                    threshold=config.TFIDF_FALLBACK_THRESHOLD,
                )
            if embedded_count >= 2 and np.all(labels == -1):
                labels = self._force_best_pair_cluster(
                    normalized_embeddings,
                    min_similarity=config.FORCED_PAIR_MIN_SIMILARITY,
                )

            used_names = {
                config.UNCATEGORIZED_DIR.lower(),
                config.RAW_SUBDIR.lower(),
                config.INTERNAL_DIR.lower(),
            }
            unique_labels = sorted({int(label) for label in labels.tolist() if int(label) != -1})

            for label in unique_labels:
                cluster_texts = [
                    str(documents[idx])
                    for idx, entry_label in enumerate(labels.tolist())
                    if int(entry_label) == label and idx < len(documents) and documents[idx]
                ]
                proposed = self.generate_cluster_name(cluster_texts, cluster_index=label)
                unique_name = self._dedupe_folder_name(proposed, used_names)
                used_names.add(unique_name.lower())
                label_to_folder[label] = unique_name

            if embeddings.shape[0] > 0:
                coords = self.get_2d_coordinates(embeddings)
            else:
                coords = np.empty((0, 2), dtype=float)

            cluster_dict: dict[str, list[Path]] = {config.UNCATEGORIZED_DIR: []}
            points: list[dict[str, Any]] = []
            embedded_folder_by_id: dict[str, str] = {}
            cluster_texts_by_folder: dict[str, list[str]] = {}

            for idx, file_id in enumerate(ids):
                path = Path(file_id)
                metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                label = int(labels[idx]) if idx < len(labels) else -1
                folder = label_to_folder.get(label, config.UNCATEGORIZED_DIR)
                if file_id in self._manual_overrides:
                    folder = config.sanitize_folder_name(
                        self._manual_overrides[file_id],
                        fallback=folder,
                    )

                cluster_dict.setdefault(folder, []).append(path)

                x_value = float(coords[idx][0]) if idx < len(coords) else float(idx)
                y_value = float(coords[idx][1]) if idx < len(coords) else 0.0

                points.append(
                    {
                        "path": file_id,
                        "filename": str(metadata.get("filename", path.name)),
                        "cluster": folder,
                        "size": int(metadata.get("size", 0)),
                        "mtime": float(metadata.get("mtime", 0.0)),
                        "mtime_iso": str(metadata.get("mtime_iso", "")),
                        "snippet": str(metadata.get("snippet", "")),
                        "x": x_value,
                        "y": y_value,
                        "ambiguous": False,
                        "ambiguity_choices": [],
                        "manual_override": file_id in self._manual_overrides,
                    }
                )
                embedded_folder_by_id[file_id] = folder

                if idx < len(documents):
                    text = str(documents[idx] or "").strip()
                    if text:
                        cluster_texts_by_folder.setdefault(folder, []).append(text)

            uncategorized_ids = sorted(self._uncategorizable.keys())
            uncategorized_coords = self._build_uncategorized_coordinates(
                count=len(uncategorized_ids),
                reference=coords,
            )
            for idx, file_id in enumerate(uncategorized_ids):
                path = Path(file_id)
                if not path.exists():
                    continue

                metadata = self._uncategorizable[file_id]
                folder = config.UNCATEGORIZED_DIR
                if file_id in self._manual_overrides:
                    folder = config.sanitize_folder_name(
                        self._manual_overrides[file_id],
                        fallback=config.UNCATEGORIZED_DIR,
                    )
                cluster_dict.setdefault(folder, []).append(path)

                snippet_text = str(metadata.get("snippet", "")).strip()
                if snippet_text:
                    cluster_texts_by_folder.setdefault(folder, []).append(snippet_text)

                points.append(
                    {
                        "path": file_id,
                        "filename": str(metadata.get("filename", path.name)),
                        "cluster": folder,
                        "size": int(metadata.get("size", 0)),
                        "mtime": float(metadata.get("mtime", 0.0)),
                        "mtime_iso": str(metadata.get("mtime_iso", "")),
                        "snippet": str(metadata.get("snippet", "")),
                        "x": float(uncategorized_coords[idx][0]),
                        "y": float(uncategorized_coords[idx][1]),
                        "ambiguous": False,
                        "ambiguity_choices": [],
                        "manual_override": file_id in self._manual_overrides,
                    }
                )

            cluster_dict = {
                folder: self._dedupe_paths(paths)
                for folder, paths in cluster_dict.items()
                if paths or folder == config.UNCATEGORIZED_DIR
            }

            try:
                self.folder_organizer.reorganize(cluster_dict)
            except Exception:
                self.logger.exception("Folder reorganization failed")

            ambiguities = self._detect_ambiguities(
                ids=ids,
                normalized_embeddings=normalized_embeddings,
                assigned_folders=embedded_folder_by_id,
            )
            for point in points:
                file_id = str(point.get("path", ""))
                ambiguity = ambiguities.get(file_id)
                if ambiguity:
                    point["ambiguous"] = True
                    point["ambiguity_choices"] = list(ambiguity.get("choices", []))

            points.sort(key=lambda item: (item["cluster"], item["filename"].lower()))
            self._cluster_lookup_cache = {
                str(point.get("path", "")): str(point.get("cluster", config.UNCATEGORIZED_DIR))
                for point in points
            }
            self._current_cluster_sizes = {
                folder: len(files)
                for folder, files in cluster_dict.items()
                if len(files) > 0
            }

            duplicates = self._compute_duplicate_pairs(
                ids=ids,
                normalized_embeddings=normalized_embeddings,
                cluster_lookup=self._cluster_lookup_cache,
            )
            similarity_edges = self._compute_similarity_edges(
                ids=ids,
                normalized_embeddings=normalized_embeddings,
            )
            cluster_summaries = self._build_cluster_summaries(
                cluster_texts_by_folder=cluster_texts_by_folder,
                cluster_dict=cluster_dict,
            )
            cluster_options = sorted(
                {
                    folder
                    for folder, count in self._current_cluster_sizes.items()
                    if folder != config.UNCATEGORIZED_DIR and count > 0
                }
            )

            semantic_cluster_count = len(
                [
                    name
                    for name, files in cluster_dict.items()
                    if name != config.UNCATEGORIZED_DIR and len(files) > 0
                ]
            )
            uncategorized_count = sum(
                1 for point in points if point["cluster"] == config.UNCATEGORIZED_DIR
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            timeline_point = {
                "time_iso": now_iso,
                "total_files": len(points),
                "cluster_count": semantic_cluster_count,
                "uncategorized_count": uncategorized_count,
                "cluster_sizes": dict(self._current_cluster_sizes),
            }
            self._timeline_history.append(timeline_point)
            if len(self._timeline_history) > config.MAX_TIMELINE_POINTS:
                self._timeline_history = self._timeline_history[-config.MAX_TIMELINE_POINTS :]

            snapshot = {
                "points": points,
                "stats": {
                    "total_files": len(points),
                    "cluster_count": semantic_cluster_count,
                    "uncategorized_count": uncategorized_count,
                    "last_update_iso": now_iso,
                },
                "ambiguities": ambiguities,
                "duplicates": duplicates,
                "similarity_edges": similarity_edges,
                "cluster_summaries": cluster_summaries,
                "cluster_options": cluster_options,
                "recent_events": self._event_history[-config.MAX_RECENT_EVENTS_IN_SNAPSHOT :],
                "timeline": self._timeline_history[-config.MAX_TIMELINE_POINTS :],
                "empty_message": "No files indexed yet. Drop supported files into _raw/.",
            }
            with self._snapshot_lock:
                self._dashboard_snapshot = snapshot

            self.logger.info(
                "Recluster complete: embedded=%d uncategorizable=%d semantic_clusters=%d uncategorized=%d",
                embedded_count,
                len(self._uncategorizable),
                semantic_cluster_count,
                uncategorized_count,
            )

    def get_2d_coordinates(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce high-dimensional embeddings to stable 2D coordinates."""
        sample_count = embeddings.shape[0]
        if sample_count == 0:
            return np.empty((0, 2), dtype=float)
        if sample_count == 1:
            return np.array([[0.0, 0.0]], dtype=float)
        if sample_count == 2:
            return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

        if umap is not None:
            try:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, sample_count - 1),
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=42,
                )
                return np.asarray(reducer.fit_transform(embeddings), dtype=float)
            except Exception as exc:
                self.logger.warning("UMAP failed; trying t-SNE fallback: %s", exc)

        try:
            perplexity = max(1, min(30, sample_count - 1))
            reducer = TSNE(
                n_components=2,
                metric="euclidean",
                perplexity=perplexity,
                init="random",
                learning_rate="auto",
                random_state=42,
            )
            return np.asarray(reducer.fit_transform(embeddings), dtype=float)
        except Exception as exc:
            self.logger.warning("t-SNE failed; using deterministic fallback layout: %s", exc)
            x = np.arange(sample_count, dtype=float)
            y = np.zeros(sample_count, dtype=float)
            return np.column_stack((x, y))

    def generate_cluster_name(self, document_texts: list[str], cluster_index: int) -> str:
        """Generate a folder name from cluster content."""
        texts = [text for text in document_texts if text.strip()]
        fallback = f"Cluster_{cluster_index}"

        ollama_name = self._generate_cluster_name_with_ollama(texts, fallback)
        if ollama_name:
            return config.sanitize_folder_name(ollama_name, fallback=fallback)

        if not texts:
            return config.sanitize_folder_name(fallback, fallback=fallback)

        try:
            vectorizer = TfidfVectorizer(max_features=3, stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.sum(axis=0)).ravel()

            sorted_indices = scores.argsort()[::-1]
            keywords = [
                str(feature_names[idx])
                for idx in sorted_indices
                if idx < len(feature_names) and float(scores[idx]) > 0
            ]
            keywords = keywords[:3]

            if not keywords:
                return config.sanitize_folder_name(fallback, fallback=fallback)

            title = "_".join(word.capitalize() for word in keywords)
            return config.sanitize_folder_name(title, fallback=fallback)
        except Exception as exc:
            self.logger.warning("TF-IDF cluster naming failed for cluster %s: %s", cluster_index, exc)
            return config.sanitize_folder_name(fallback, fallback=fallback)

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the latest dashboard snapshot."""
        with self._snapshot_lock:
            return copy.deepcopy(self._dashboard_snapshot)

    def set_manual_override(self, filepath: str, cluster_name: str | None) -> tuple[bool, str]:
        """Set or clear a manual cluster override for a file path."""
        file_id = self._file_id(Path(filepath))
        normalized_cluster = (cluster_name or "").strip()
        old_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
        if not Path(file_id).exists() and file_id not in self._cluster_lookup_cache:
            return False, "File is not currently indexed."

        with self._engine_lock:
            if not normalized_cluster or normalized_cluster.lower() in {"auto", "__auto__"}:
                if file_id in self._manual_overrides:
                    self._manual_overrides.pop(file_id, None)
                    self._save_manual_overrides()
                    self.recluster()
                    new_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
                    self._record_event(
                        event_type="override_cleared",
                        file_id=file_id,
                        old_cluster=old_cluster,
                        new_cluster=new_cluster,
                        semantic_drift=0.0,
                        detail=f"Manual override cleared for {Path(file_id).name}.",
                    )
                    return True, "Manual override cleared."
                return True, "No manual override was set."

            target = config.sanitize_folder_name(normalized_cluster, fallback="Manual_Cluster")
            self._manual_overrides[file_id] = target
            self._save_manual_overrides()
            self.recluster()
            new_cluster = self._cluster_lookup_cache.get(file_id, config.UNCATEGORIZED_DIR)
            self._record_event(
                event_type="override_set",
                file_id=file_id,
                old_cluster=old_cluster,
                new_cluster=new_cluster,
                semantic_drift=0.0,
                detail=f"Manual override set for {Path(file_id).name} -> {target}.",
            )
            return True, f"Manual override set to '{target}'."

    def semantic_search(self, query: str, top_k: int = config.DEFAULT_SEARCH_TOP_K) -> list[dict[str, Any]]:
        """Return top-k semantic matches from indexed documents."""
        normalized_query = query.strip()
        if not normalized_query:
            return []

        limit = max(1, min(int(top_k), config.MAX_SEARCH_TOP_K))
        candidate_limit = max(limit, limit * config.SEARCH_CANDIDATE_MULTIPLIER)
        candidate_limit = min(candidate_limit, config.SEARCH_MAX_CANDIDATES)
        query_embedding = self._encode_text(normalized_query)
        fallback_embedding = self._model_unavailable or self._model is None

        with self._engine_lock:
            try:
                result = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=candidate_limit,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as exc:
                self.logger.warning("Semantic search failed: %s", exc)
                return []

        ids_rows = result.get("ids") or [[]]
        metadata_rows = result.get("metadatas") or [[]]
        document_rows = result.get("documents") or [[]]
        distance_rows = result.get("distances") or [[]]

        ids = ids_rows[0] if ids_rows else []
        metadatas = metadata_rows[0] if metadata_rows else []
        documents = document_rows[0] if document_rows else []
        distances = distance_rows[0] if distance_rows else []
        cluster_lookup = self._build_cluster_lookup()
        lexical_scores, any_token_match = self._compute_lexical_scores(
            normalized_query,
            documents,
            metadatas,
        )

        if fallback_embedding:
            semantic_weight = config.SEARCH_FALLBACK_SEMANTIC_WEIGHT
            lexical_weight = config.SEARCH_FALLBACK_LEXICAL_WEIGHT
        else:
            semantic_weight = config.SEARCH_SEMANTIC_WEIGHT
            lexical_weight = config.SEARCH_LEXICAL_WEIGHT

        matches: list[dict[str, Any]] = []
        for index, file_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document = str(documents[index]) if index < len(documents) and documents[index] else ""
            distance = float(distances[index]) if index < len(distances) else 0.0
            path = Path(str(file_id))
            lexical_entry = lexical_scores[index] if index < len(lexical_scores) else {}
            lexical_score = float(lexical_entry.get("score", 0.0))
            token_hits = int(lexical_entry.get("token_hits", 0))
            semantic_score = float(1.0 / (1.0 + max(0.0, distance)))
            combined_score = (semantic_weight * semantic_score) + (lexical_weight * lexical_score)

            document_length = len(document.strip())
            if document_length < config.MIN_TEXT_LENGTH:
                continue

            if any_token_match and token_hits == 0 and lexical_score < config.SEARCH_MIN_LEXICAL_SCORE:
                continue

            if (
                combined_score < config.SEARCH_MIN_COMBINED_SCORE
                and lexical_score < config.SEARCH_MIN_LEXICAL_SCORE
            ):
                continue

            matches.append(
                {
                    "path": str(file_id),
                    "filename": str(metadata.get("filename", path.name)),
                    "cluster": cluster_lookup.get(str(file_id), config.UNCATEGORIZED_DIR),
                    "distance": distance,
                    "relevance": combined_score,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "token_hits": token_hits,
                    "snippet": str(metadata.get("snippet", document[: config.SNIPPET_LENGTH])),
                    "document": document,
                }
            )

        matches.sort(
            key=lambda item: (
                float(item.get("relevance", 0.0)),
                float(item.get("lexical_score", 0.0)),
                float(item.get("semantic_score", 0.0)),
            ),
            reverse=True,
        )
        return matches[:limit]

    def answer_query(self, query: str, top_k: int = config.DEFAULT_SEARCH_TOP_K) -> dict[str, Any]:
        """Run full RAG retrieval + generation over indexed files."""
        normalized_query = query.strip()
        if not normalized_query:
            return {
                "query": query,
                "answer": "Enter a query to search indexed files.",
                "results": [],
                "used_ollama": False,
                "generator": "none",
                "error": None,
            }

        matches = self.semantic_search(normalized_query, top_k=top_k)
        if not matches:
            return {
                "query": normalized_query,
                "answer": "No indexed documents matched this query yet.",
                "results": [],
                "used_ollama": False,
                "generator": "none",
                "error": None,
            }

        context_docs = matches[: config.RAG_CONTEXT_DOCS]
        answer, generator, error = self._generate_rag_answer(normalized_query, context_docs)
        return {
            "query": normalized_query,
            "answer": answer,
            "results": matches,
            "used_ollama": generator == "ollama",
            "generator": generator,
            "error": error,
        }

    def close(self) -> None:
        """Release in-memory resources used by the engine."""
        with self._engine_lock:
            self._model = None

    def _file_id(self, path: Path) -> str:
        return str(path.resolve(strict=False))

    def _encode_text(self, text: str) -> list[float]:
        model = self._get_model()
        if model is None:
            return self._fallback_text_embedding(text)

        try:
            embedding = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
            if isinstance(embedding, np.ndarray):
                return embedding.astype(float).tolist()
            return [float(value) for value in embedding]
        except Exception as exc:
            self.logger.warning("Embedding model inference failed. Using fallback embedding: %s", exc)
            return self._fallback_text_embedding(text)

    def _get_model(self) -> SentenceTransformer | None:
        with self._engine_lock:
            if self._model_unavailable:
                return None
            if self._model is None:
                self.logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
                if self._hf_local_only and not self._is_hf_model_cached(config.EMBEDDING_MODEL):
                    self._model_unavailable = True
                    self.logger.warning(
                        "Embedding model '%s' is not cached locally. Using hashing fallback. "
                        "To fetch it once, run with %s=0.",
                        config.EMBEDDING_MODEL,
                        config.HF_LOCAL_ONLY_ENV,
                    )
                    return None
                if self._hf_local_only:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    self._model = SentenceTransformer(config.EMBEDDING_MODEL)
                except Exception as exc:
                    self._model_unavailable = True
                    self.logger.warning(
                        "Failed to load embedding model (%s). Falling back to local hashing embeddings.",
                        exc,
                    )
                    return None
            return self._model

    def _fallback_text_embedding(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(float), 384)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            return vector.tolist()

        matrix = self._hashing_vectorizer.transform([cleaned])
        vector = matrix.toarray().astype(float).ravel()
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def _build_metadata(self, path: Path, text: str) -> dict[str, Any]:
        try:
            stat = path.stat()
            size = int(stat.st_size)
            mtime = float(stat.st_mtime)
        except OSError:
            size = 0
            mtime = 0.0

        snippet = text[: config.SNIPPET_LENGTH]
        return {
            "filename": path.name,
            "size": size,
            "mtime": mtime,
            "mtime_iso": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            if mtime > 0
            else "",
            "snippet": snippet,
            "cluster": config.UNCATEGORIZED_DIR,
        }

    def _compute_lexical_scores(
        self,
        query: str,
        documents: list[Any],
        metadatas: list[Any],
    ) -> tuple[list[dict[str, float | int]], bool]:
        query_terms = self._tokenize_search_text(query, for_query=True)
        query_lower = query.lower().strip()

        search_texts: list[str] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            search_texts.append(self._build_search_text(str(document or ""), metadata))

        tfidf_scores = np.zeros(len(search_texts), dtype=float)
        if search_texts:
            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                matrix = vectorizer.fit_transform([query] + search_texts)
                if matrix.shape[1] > 0:
                    tfidf_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            except Exception:
                tfidf_scores = np.zeros(len(search_texts), dtype=float)

        lexical_entries: list[dict[str, float | int]] = []
        any_token_match = False

        for index, text in enumerate(search_texts):
            doc_terms = self._tokenize_search_text(text)
            token_hits = len(query_terms.intersection(doc_terms)) if query_terms else 0
            any_token_match = any_token_match or token_hits > 0
            token_coverage = (token_hits / len(query_terms)) if query_terms else 0.0

            tfidf_score = float(tfidf_scores[index]) if index < len(tfidf_scores) else 0.0
            phrase_bonus = 1.0 if query_lower and query_lower in text.lower() else 0.0
            lexical_score = min(
                1.0,
                max(0.0, (0.65 * tfidf_score) + (0.25 * token_coverage) + (0.10 * phrase_bonus)),
            )

            lexical_entries.append({"score": lexical_score, "token_hits": token_hits})

        return lexical_entries, any_token_match

    def _build_search_text(self, document: str, metadata: dict[str, Any]) -> str:
        filename = str(metadata.get("filename", ""))
        snippet = str(metadata.get("snippet", ""))
        parts = [filename, document, snippet]
        return " ".join(part for part in parts if part).strip()

    def _tokenize_search_text(self, text: str, for_query: bool = False) -> set[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        tokens: set[str] = set()

        for raw_token in raw_tokens:
            token = self._normalize_search_token(raw_token)
            if len(token) < config.SEARCH_MIN_TOKEN_LENGTH:
                continue
            if token in ENGLISH_STOP_WORDS:
                continue
            if for_query and token in config.SEARCH_QUERY_NOISE_TOKENS:
                continue
            tokens.add(token)
        return tokens

    def _normalize_search_token(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _prune_missing_uncategorizable(self) -> None:
        missing = [file_id for file_id in self._uncategorizable if not Path(file_id).exists()]
        for file_id in missing:
            self._uncategorizable.pop(file_id, None)

    def _prune_missing_overrides(self) -> None:
        missing = [file_id for file_id in self._manual_overrides if not Path(file_id).exists()]
        if not missing:
            return
        for file_id in missing:
            self._manual_overrides.pop(file_id, None)
        self._save_manual_overrides()

    def _detect_ambiguities(
        self,
        ids: list[str],
        normalized_embeddings: np.ndarray,
        assigned_folders: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        ambiguities: dict[str, dict[str, Any]] = {}
        if normalized_embeddings.size == 0 or len(ids) < 3:
            return ambiguities

        folder_to_indices: dict[str, list[int]] = {}
        for idx, file_id in enumerate(ids):
            folder = assigned_folders.get(file_id, config.UNCATEGORIZED_DIR)
            if folder == config.UNCATEGORIZED_DIR:
                continue
            folder_to_indices.setdefault(folder, []).append(idx)

        if len(folder_to_indices) < 2:
            return ambiguities

        centroids: dict[str, np.ndarray] = {}
        for folder, indices in folder_to_indices.items():
            cluster_vectors = normalized_embeddings[indices]
            if cluster_vectors.size == 0:
                continue
            centroid = np.mean(cluster_vectors, axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            centroids[folder] = centroid

        if len(centroids) < 2:
            return ambiguities

        for idx, file_id in enumerate(ids):
            vector = normalized_embeddings[idx]
            scored: list[tuple[str, float]] = []
            for folder, centroid in centroids.items():
                similarity = float(np.dot(vector, centroid))
                scored.append((folder, similarity))
            scored.sort(key=lambda item: item[1], reverse=True)
            if len(scored) < 2:
                continue

            first_folder, first_score = scored[0]
            second_folder, second_score = scored[1]
            delta = first_score - second_score

            if first_score < config.AMBIGUITY_MIN_SIMILARITY:
                continue
            if delta > config.AMBIGUITY_DELTA_THRESHOLD:
                continue

            choices = [first_folder, second_folder]
            ambiguities[file_id] = {
                "current_cluster": assigned_folders.get(file_id, config.UNCATEGORIZED_DIR),
                "choices": choices,
                "top_similarity": first_score,
                "second_similarity": second_score,
                "delta": delta,
            }
        return ambiguities

    def _compute_duplicate_pairs(
        self,
        ids: list[str],
        normalized_embeddings: np.ndarray,
        cluster_lookup: dict[str, str],
    ) -> list[dict[str, Any]]:
        duplicates: list[dict[str, Any]] = []
        sample_count = len(ids)
        if sample_count < 2 or normalized_embeddings.size == 0:
            return duplicates

        similarities = np.matmul(normalized_embeddings, normalized_embeddings.T)
        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                similarity = float(similarities[left, right])
                if similarity < config.DUPLICATE_SIMILARITY_THRESHOLD:
                    continue
                path_left = ids[left]
                path_right = ids[right]
                duplicates.append(
                    {
                        "path_a": path_left,
                        "path_b": path_right,
                        "filename_a": Path(path_left).name,
                        "filename_b": Path(path_right).name,
                        "cluster_a": cluster_lookup.get(path_left, config.UNCATEGORIZED_DIR),
                        "cluster_b": cluster_lookup.get(path_right, config.UNCATEGORIZED_DIR),
                        "similarity": similarity,
                    }
                )

        duplicates.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
        return duplicates[: config.MAX_DUPLICATE_PAIRS]

    def _compute_similarity_edges(
        self,
        ids: list[str],
        normalized_embeddings: np.ndarray,
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        sample_count = len(ids)
        if sample_count < 2 or normalized_embeddings.size == 0:
            return edges

        similarities = np.matmul(normalized_embeddings, normalized_embeddings.T)
        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                similarity = float(similarities[left, right])
                if similarity < config.EDGE_SIMILARITY_THRESHOLD:
                    continue
                edges.append(
                    {
                        "source": ids[left],
                        "target": ids[right],
                        "similarity": similarity,
                    }
                )

        edges.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
        return edges[: config.MAX_SIMILARITY_EDGES]

    def _build_cluster_summaries(
        self,
        cluster_texts_by_folder: dict[str, list[str]],
        cluster_dict: dict[str, list[Path]],
    ) -> dict[str, str]:
        summaries: dict[str, str] = {}
        for folder, paths in cluster_dict.items():
            if not paths:
                continue
            texts = cluster_texts_by_folder.get(folder, [])
            summaries[folder] = self._generate_cluster_summary(folder, texts, len(paths))
        return summaries

    def _generate_cluster_summary(self, folder: str, texts: list[str], file_count: int) -> str:
        if not texts:
            if file_count == 1:
                return "Contains one file with limited extractable text."
            return f"Contains {file_count} files with limited extractable text."

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=max(1, config.CLUSTER_SUMMARY_KEYWORDS),
            )
            matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.sum(axis=0)).ravel()
            sorted_indices = scores.argsort()[::-1]
            keywords = [
                str(feature_names[idx])
                for idx in sorted_indices
                if idx < len(feature_names) and float(scores[idx]) > 0
            ][: config.CLUSTER_SUMMARY_KEYWORDS]
        except Exception:
            keywords = []

        if not keywords:
            return f"{file_count} files focused around the '{folder}' topic."
        topic_text = ", ".join(keywords)
        return f"{file_count} files focused on {topic_text}."

    def _dedupe_folder_name(self, name: str, used_names: set[str]) -> str:
        candidate = config.sanitize_folder_name(name, fallback="Cluster")
        if candidate.lower() not in used_names:
            return candidate

        suffix = 2
        while True:
            amended = f"{candidate}_{suffix}"
            if amended.lower() not in used_names:
                return amended
            suffix += 1

    def _dedupe_paths(self, paths: list[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(Path(key))
        return deduped

    def _build_uncategorized_coordinates(
        self,
        count: int,
        reference: np.ndarray,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0, 2), dtype=float)

        if reference.size == 0:
            start_x = -1.0
            start_y = 0.0
        else:
            start_x = float(np.min(reference[:, 0])) - 1.0
            start_y = float(np.min(reference[:, 1]))

        coords = np.zeros((count, 2), dtype=float)
        for idx in range(count):
            coords[idx][0] = start_x
            coords[idx][1] = start_y + (idx * 0.2)
        return coords

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _cluster_by_similarity_fallback(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        sample_count = embeddings.shape[0]
        labels = np.full(sample_count, -1, dtype=int)
        if sample_count < 2:
            return labels

        parents = list(range(sample_count))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parents[root_b] = root_a

        similarities = np.matmul(embeddings, embeddings.T)
        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                if float(similarities[left, right]) >= threshold:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in range(sample_count):
            root = find(index)
            groups.setdefault(root, []).append(index)

        next_label = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            for member in members:
                labels[member] = next_label
            next_label += 1

        return labels

    def _force_best_pair_cluster(
        self,
        embeddings: np.ndarray,
        min_similarity: float,
    ) -> np.ndarray:
        sample_count = embeddings.shape[0]
        labels = np.full(sample_count, -1, dtype=int)
        if sample_count < 2:
            return labels

        similarities = np.matmul(embeddings, embeddings.T)
        np.fill_diagonal(similarities, -1.0)

        best_index = np.unravel_index(np.argmax(similarities), similarities.shape)
        best_similarity = float(similarities[best_index])
        if best_similarity < min_similarity:
            return labels

        left, right = int(best_index[0]), int(best_index[1])
        labels[left] = 0
        labels[right] = 0
        return labels

    def _cluster_by_tfidf_fallback(
        self,
        documents: list[str],
        threshold: float,
    ) -> np.ndarray:
        sample_count = len(documents)
        labels = np.full(sample_count, -1, dtype=int)
        if sample_count < 2:
            return labels

        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(documents)
            if matrix.shape[1] == 0:
                return labels
            similarities = cosine_similarity(matrix)
        except Exception:
            return labels

        parents = list(range(sample_count))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parents[root_b] = root_a

        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                if float(similarities[left, right]) >= threshold:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in range(sample_count):
            root = find(index)
            groups.setdefault(root, []).append(index)

        next_label = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            for member in members:
                labels[member] = next_label
            next_label += 1

        return labels

    def _generate_cluster_name_with_ollama(
        self,
        texts: list[str],
        fallback: str,
    ) -> str | None:
        if not self._ollama_enabled or not self._ollama_path or not texts:
            return None

        excerpt = "\n".join(text[:300] for text in texts[:4])
        prompt = (
            "Generate a concise 2-3 word topic label for these documents. "
            "Return only the label.\n\n"
            f"{excerpt}"
        )

        try:
            completed = subprocess.run(
                [self._ollama_path, "run", self._ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=self._ollama_timeout,
                check=False,
            )
            if completed.returncode != 0:
                self.logger.debug("Ollama naming failed with return code %s", completed.returncode)
                return None

            output = completed.stdout.strip().splitlines()
            if not output:
                return None

            candidate = output[0].strip(" \t\n\r\"'")
            sanitized = config.sanitize_folder_name(candidate, fallback=fallback)
            return sanitized if sanitized else None
        except Exception as exc:
            self.logger.debug("Ollama naming unavailable, using TF-IDF: %s", exc)
            return None

    def _generate_rag_answer(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> tuple[str, str, str | None]:
        providers: list[str]
        if self._rag_provider == "auto":
            providers = ["ollama", "huggingface"]
        elif self._rag_provider == "ollama":
            providers = ["ollama"]
        else:
            providers = ["huggingface"]

        errors: list[str] = []
        for provider in providers:
            if provider == "ollama":
                answer, error = self._generate_rag_with_ollama(query, contexts)
            else:
                answer, error = self._generate_rag_with_huggingface(query, contexts)

            if answer:
                return answer, provider, None
            if error:
                errors.append(error)

        if errors:
            return "", "none", " | ".join(errors)
        return "", "none", "No RAG generator is currently available."

    def _generate_rag_with_ollama(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        if not self._ollama_rag_enabled:
            return None, (
                f"Ollama RAG is disabled. Set {config.OLLAMA_RAG_ENABLE_ENV}=1 "
                "or set SEFS_RAG_PROVIDER=huggingface."
            )
        if not self._ollama_path:
            return None, (
                "Ollama executable was not found in PATH or common install locations. "
                "Set SEFS_RAG_PROVIDER=huggingface for an alternate local generator."
            )

        prompt = self._render_rag_prompt(query, contexts)
        if not prompt:
            return None, "No retrievable context was available for generation."

        answer, error, timed_out = self._run_ollama_prompt(prompt, self._ollama_rag_timeout)
        if answer:
            return answer, None

        if timed_out:
            retry_timeout = max(self._ollama_rag_timeout * 2, 30)
            retry_prompt = self._render_rag_prompt(
                query,
                contexts,
                max_docs=min(2, len(contexts)),
                max_context_chars=min(1_500, config.RAG_MAX_CONTEXT_CHARS),
            )
            if retry_prompt:
                retry_answer, retry_error, _ = self._run_ollama_prompt(retry_prompt, retry_timeout)
                if retry_answer:
                    return retry_answer, None
                if retry_error:
                    return (
                        None,
                        (
                            f"Ollama timed out at {self._ollama_rag_timeout}s and retry failed at "
                            f"{retry_timeout}s. Set {config.OLLAMA_RAG_TIMEOUT_ENV}=60 or higher."
                        ),
                    )

        return None, error

    def _generate_rag_with_huggingface(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        prompt = self._render_rag_prompt(query, contexts)
        if not prompt:
            return None, "No retrievable context was available for generation."

        pipeline_obj, error = self._get_hf_rag_pipeline()
        if pipeline_obj is None:
            return None, error or "HuggingFace RAG pipeline is unavailable."

        try:
            outputs = pipeline_obj(
                prompt,
                max_new_tokens=self._hf_rag_max_new_tokens,
                do_sample=False,
            )
            if not outputs:
                return None, "HuggingFace generator returned no outputs."

            first = outputs[0]
            if isinstance(first, dict):
                answer = str(first.get("generated_text", "")).strip()
            else:
                answer = str(first).strip()

            if not answer:
                return None, "HuggingFace generator returned an empty answer."
            return answer, None
        except Exception as exc:
            self.logger.debug("HuggingFace RAG generation failed: %s", exc)
            return None, f"HuggingFace generation error: {exc}"

    def _get_hf_rag_pipeline(self) -> tuple[Any | None, str | None]:
        with self._engine_lock:
            if self._hf_rag_unavailable:
                return None, "HuggingFace RAG pipeline previously failed to initialize."
            if self._hf_rag_pipeline is not None:
                return self._hf_rag_pipeline, None

            if self._hf_local_only and not self._is_hf_model_cached(self._hf_rag_model):
                self._hf_rag_unavailable = True
                return (
                    None,
                    (
                        f"HuggingFace model '{self._hf_rag_model}' is not cached locally. "
                        f"Download once with {config.HF_LOCAL_ONLY_ENV}=0 or use Ollama provider."
                    ),
                )

            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:
                self._hf_rag_unavailable = True
                return None, f"transformers pipeline import failed: {exc}"

            if self._hf_local_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            try:
                self._hf_rag_pipeline = pipeline(
                    "text2text-generation",
                    model=self._hf_rag_model,
                    tokenizer=self._hf_rag_model,
                )
                return self._hf_rag_pipeline, None
            except Exception as exc:
                self._hf_rag_unavailable = True
                return None, f"Unable to load HuggingFace model '{self._hf_rag_model}': {exc}"

    def _render_rag_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        max_docs: int | None = None,
        max_context_chars: int | None = None,
    ) -> str:
        context_sections: list[str] = []
        context_budget = max_context_chars if max_context_chars is not None else config.RAG_MAX_CONTEXT_CHARS
        selected_contexts = contexts if max_docs is None else contexts[:max_docs]
        for index, item in enumerate(selected_contexts, start=1):
            text = str(item.get("document", "")).strip()
            if not text:
                text = str(item.get("snippet", "")).strip()
            if not text:
                continue

            section = f"[{index}] {item.get('filename', 'unknown')}\n{text}\n"
            if len(section) > context_budget:
                section = section[:context_budget]
            context_sections.append(section)
            context_budget -= len(section)
            if context_budget <= 0:
                break

        if not context_sections:
            return ""

        return (
            "You are answering questions using only the provided document context. "
            "If context is insufficient, say so clearly. "
            "Cite supporting documents as [1], [2], etc.\n\n"
            f"Query: {query}\n\n"
            "Context:\n"
            f"{''.join(context_sections)}\n"
            "Answer:"
        )

    def _run_ollama_prompt(
        self,
        prompt: str,
        timeout_seconds: int,
    ) -> tuple[str | None, str | None, bool]:
        try:
            completed = subprocess.run(
                [self._ollama_path, "run", self._ollama_rag_model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                self.logger.debug("Ollama RAG generation failed with code %s", completed.returncode)
                stderr = (completed.stderr or "").strip()
                return None, f"Ollama generation failed ({completed.returncode}): {stderr or 'unknown error'}", False
            answer = completed.stdout.strip()
            if not answer:
                return None, "Ollama returned an empty response.", False
            return answer, None, False
        except subprocess.TimeoutExpired:
            return None, f"Ollama generation timed out after {timeout_seconds} seconds.", True
        except Exception as exc:
            self.logger.debug("Ollama RAG generation failed: %s", exc)
            return None, f"Ollama generation error: {exc}", False

    def _find_ollama_executable(self) -> str | None:
        candidate = shutil.which("ollama")
        if candidate:
            return candidate

        fallback_paths = [
            "/opt/homebrew/bin/ollama",
            "/usr/local/bin/ollama",
            str(Path.home() / ".local/bin/ollama"),
        ]
        for path in fallback_paths:
            candidate_path = Path(path)
            if candidate_path.exists() and os.access(candidate_path, os.X_OK):
                return str(candidate_path)
        return None

    def _patch_posthog_capture(self) -> None:
        try:
            import posthog  # type: ignore
        except Exception:
            return

        def _noop_capture(*_args: Any, **_kwargs: Any) -> None:
            return None

        try:
            posthog.capture = _noop_capture  # type: ignore[assignment]
            if hasattr(posthog, "disabled"):
                posthog.disabled = True  # type: ignore[attr-defined]
        except Exception:
            return

    def _is_hf_model_cached(self, model_name: str) -> bool:
        normalized = model_name.strip()
        if not normalized:
            return False

        model_candidates = [normalized]
        if "/" not in normalized:
            # SentenceTransformers commonly resolves short ids via this namespace.
            model_candidates.append(f"sentence-transformers/{normalized}")

        model_dir_names = [f"models--{candidate.replace('/', '--')}" for candidate in model_candidates]
        for cache_root in self._get_hf_cache_roots():
            for model_dir_name in model_dir_names:
                model_root = cache_root / model_dir_name
                snapshots_dir = model_root / "snapshots"
                if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                    return True

        # Legacy SentenceTransformers cache layout.
        legacy_roots: list[Path] = []
        st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
        if st_home:
            legacy_roots.append(Path(st_home).expanduser())
        legacy_roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")

        legacy_names: set[str] = set()
        for candidate in model_candidates:
            short_name = candidate.split("/")[-1]
            legacy_names.add(short_name)
            legacy_names.add(candidate.replace("/", "_"))

        for legacy_root in legacy_roots:
            for legacy_name in legacy_names:
                candidate_dir = legacy_root / legacy_name
                if candidate_dir.exists() and any(candidate_dir.iterdir()):
                    return True
        return False

    def _get_hf_cache_roots(self) -> list[Path]:
        roots: list[Path] = []
        env_candidates = [
            os.getenv("HUGGINGFACE_HUB_CACHE"),
            os.getenv("HF_HOME"),
            os.getenv("TRANSFORMERS_CACHE"),
        ]
        for candidate in env_candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            roots.append(path)
            roots.append(path / "hub")

        roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        deduped: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(root.resolve(strict=False))
        return deduped

    def _load_manual_overrides(self) -> None:
        """Load manual cluster override map from disk if present."""
        self._manual_overrides = {}
        if not self._overrides_path.exists():
            return

        try:
            payload = json.loads(self._overrides_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed reading overrides file %s: %s", self._overrides_path, exc)
            return

        if not isinstance(payload, dict):
            return

        overrides: dict[str, str] = {}
        for key, value in payload.items():
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if not normalized_key or not normalized_value:
                continue
            overrides[normalized_key] = normalized_value
        self._manual_overrides = overrides

    def _save_manual_overrides(self) -> None:
        """Persist manual override map to disk."""
        try:
            self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._manual_overrides, indent=2, sort_keys=True)
            self._overrides_path.write_text(payload, encoding="utf-8")
        except Exception as exc:
            self.logger.warning("Failed writing overrides file %s: %s", self._overrides_path, exc)

    def _get_existing_embedding(self, file_id: str) -> np.ndarray | None:
        """Fetch an existing embedding vector for a file id if stored."""
        try:
            existing = self._collection.get(ids=[file_id], include=["embeddings"])
        except Exception:
            return None

        embeddings = existing.get("embeddings") or []
        if not embeddings:
            return None

        first = embeddings[0]
        if first is None:
            return None
        try:
            vector = np.asarray(first, dtype=float)
        except Exception:
            return None
        if vector.size == 0:
            return None
        return vector

    def _semantic_drift(self, old_embedding: np.ndarray | None, new_embedding: np.ndarray | None) -> float:
        """Return normalized semantic drift in [0, 1] based on cosine shift."""
        if old_embedding is None and new_embedding is None:
            return 0.0
        if old_embedding is None or new_embedding is None:
            return 1.0

        old = np.asarray(old_embedding, dtype=float).ravel()
        new = np.asarray(new_embedding, dtype=float).ravel()
        if old.size == 0 or new.size == 0:
            return 1.0

        old_norm = float(np.linalg.norm(old))
        new_norm = float(np.linalg.norm(new))
        if old_norm == 0.0 or new_norm == 0.0:
            return 1.0

        similarity = float(np.dot(old / old_norm, new / new_norm))
        similarity = max(-1.0, min(1.0, similarity))
        drift = (1.0 - similarity) / 2.0
        return float(max(0.0, min(1.0, drift)))

    def _format_semantic_diff_detail(
        self,
        filename: str,
        old_cluster: str,
        new_cluster: str,
        semantic_drift: float,
    ) -> str:
        """Render a readable semantic drift message for UI/event feed."""
        if old_cluster == new_cluster:
            return (
                f"{filename} was reprocessed in '{new_cluster}'. "
                f"Semantic drift: {semantic_drift:.2f}."
            )
        return (
            f"{filename} shifted from '{old_cluster}' to '{new_cluster}'. "
            f"Semantic drift: {semantic_drift:.2f}."
        )

    def _record_event(
        self,
        event_type: str,
        file_id: str,
        old_cluster: str,
        new_cluster: str,
        semantic_drift: float,
        detail: str,
    ) -> None:
        """Record a bounded event/timeline history entry and refresh snapshot fields."""
        timestamp = datetime.now(timezone.utc).isoformat()
        event = {
            "time_iso": timestamp,
            "type": event_type,
            "path": file_id,
            "filename": Path(file_id).name,
            "old_cluster": old_cluster,
            "new_cluster": new_cluster,
            "semantic_drift": float(semantic_drift),
            "detail": detail,
        }
        self._event_history.append(event)
        if len(self._event_history) > config.MAX_EVENT_HISTORY:
            self._event_history = self._event_history[-config.MAX_EVENT_HISTORY :]

        with self._snapshot_lock:
            stats = dict(self._dashboard_snapshot.get("stats", {}))
            timeline_point = {
                "time_iso": timestamp,
                "total_files": int(stats.get("total_files", 0)),
                "cluster_count": int(stats.get("cluster_count", 0)),
                "uncategorized_count": int(stats.get("uncategorized_count", 0)),
                "cluster_sizes": dict(self._current_cluster_sizes),
            }
            self._timeline_history.append(timeline_point)
            if len(self._timeline_history) > config.MAX_TIMELINE_POINTS:
                self._timeline_history = self._timeline_history[-config.MAX_TIMELINE_POINTS :]

            self._dashboard_snapshot["recent_events"] = self._event_history[
                -config.MAX_RECENT_EVENTS_IN_SNAPSHOT :
            ]
            self._dashboard_snapshot["timeline"] = self._timeline_history[
                -config.MAX_TIMELINE_POINTS :
            ]

    def _build_cluster_lookup(self) -> dict[str, str]:
        with self._snapshot_lock:
            points = list(self._dashboard_snapshot.get("points", []))
        return {str(point.get("path", "")): str(point.get("cluster", config.UNCATEGORIZED_DIR)) for point in points}
