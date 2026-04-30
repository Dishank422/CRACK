"""
Embedding-based semantic search tool for the agent-based code reviewer.

Uses Nomic CodeRankEmbed for code embeddings, Chonkie for AST-based chunking,
and FAISS IndexIDMap(IndexFlatIP) for vector storage. Supports incremental
updates via a cache directory (.crack-embeddings/) to avoid re-embedding
unchanged files on each CI run.

Cache structure:
    .crack-embeddings/
        index.faiss      - FAISS IndexIDMap wrapping IndexFlatIP
        metadata.json    - {next_id, chunks: {id: {file, start_line, end_line, text_hash}},
                            file_to_ids: {file: [id, ...]}}
"""

import hashlib
import json
import logging
import os
from typing import Callable

from .base import ToolProvider, ToolContext

CACHE_DIR = os.environ.get("CRACK_EMBEDDINGS_DIR", ".crack-embeddings")
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 512  # tokens

# File extensions worth embedding (skip binaries, lockfiles, etc.)
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".m", ".mm", ".lua", ".pl", ".pm", ".ex", ".exs", ".erl", ".hs",
    ".ml", ".mli", ".fs", ".fsx", ".clj", ".cljs", ".vim", ".el", ".jl",
    ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss", ".less",
    ".md", ".rst", ".txt", ".cfg", ".ini", ".conf",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", "venv", ".venv", "env", ".env", "dist", "build", ".eggs",
    "vendor", "third_party", ".crack-embeddings",
}


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _should_index_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in CODE_EXTENSIONS


def _collect_files(repo_path: str) -> list[str]:
    """Collect all indexable files as relative paths."""
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_path)
            if _should_index_file(rel):
                files.append(rel)
    return files


_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".cs": "c_sharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".lua": "lua", ".hs": "haskell",
    ".ml": "ocaml", ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".jl": "julia", ".r": "r",
}

_chunker_cache: dict = {}


def _get_chunker(language: str):
    """Get or create a cached CodeChunker for the given language."""
    if language not in _chunker_cache:
        from chonkie import CodeChunker
        _chunker_cache[language] = CodeChunker(language=language, chunk_size=CHUNK_SIZE)
    return _chunker_cache[language]


def _chunk_file(repo_path: str, rel_path: str) -> list[dict]:
    """Chunk a single file using Chonkie's CodeChunker with AST-based splitting."""
    full_path = os.path.join(repo_path, rel_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    if not content.strip():
        return []

    try:
        ext = os.path.splitext(rel_path)[1].lower()
        language = _LANG_MAP.get(ext)

        if language:
            chunker = _get_chunker(language)
            chunks = chunker.chunk(content)
        else:
            chunks = None
    except Exception:
        chunks = None

    if chunks:
        result = []
        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if not text.strip():
                continue
            # Approximate line numbers from character offsets
            start_idx = getattr(chunk, "start_index", None)
            end_idx = getattr(chunk, "end_index", None)
            if start_idx is not None:
                start_line = content[:start_idx].count("\n") + 1
            else:
                start_line = 1
            if end_idx is not None:
                end_line = content[:end_idx].count("\n") + 1
            else:
                end_line = content.count("\n") + 1
            result.append({
                "file": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "text": text,
                "text_hash": _text_hash(text),
            })
        return result

    # Fallback: fixed-size line-based chunking
    lines = content.splitlines(keepends=True)
    chunk_lines = 50
    result = []
    for i in range(0, len(lines), chunk_lines):
        chunk_text = "".join(lines[i : i + chunk_lines])
        if chunk_text.strip():
            result.append({
                "file": rel_path,
                "start_line": i + 1,
                "end_line": min(i + chunk_lines, len(lines)),
                "text": chunk_text,
                "text_hash": _text_hash(chunk_text),
            })
    return result


def _get_cache_dir(repo_path: str) -> str:
    """Get the cache directory path, respecting CRACK_EMBEDDINGS_DIR."""
    if os.path.isabs(CACHE_DIR):
        return CACHE_DIR
    return os.path.join(repo_path, CACHE_DIR)


def _load_cache(repo_path: str):
    """Load cached FAISS index and metadata if they exist."""
    import faiss

    cache_dir = _get_cache_dir(repo_path)
    index_path = os.path.join(cache_dir, INDEX_FILE)
    meta_path = os.path.join(cache_dir, METADATA_FILE)

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        logging.info(
            f"Loaded embedding cache: {index.ntotal} vectors, "
            f"{len(metadata.get('file_to_ids', {}))} files"
        )
        return index, metadata
    except Exception as e:
        logging.warning(f"Failed to load embedding cache: {e}")
        return None, None


def _save_cache(repo_path: str, index, metadata: dict) -> None:
    """Save FAISS index and metadata to cache."""
    import faiss

    cache_dir = _get_cache_dir(repo_path)
    os.makedirs(cache_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(cache_dir, INDEX_FILE))
    with open(os.path.join(cache_dir, METADATA_FILE), "w") as f:
        json.dump(metadata, f)
    logging.info(f"Saved embedding cache: {index.ntotal} vectors")


def _new_index():
    """Create a new FAISS IndexIDMap wrapping IndexFlatIP."""
    import faiss

    flat = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(flat)


def _embed_texts(model, texts: list[str]):
    """Embed a list of texts using fastembed."""
    import numpy as np

    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    total = len(texts)
    results = []
    log_interval = max(1, total // 5)
    for i, emb in enumerate(model.embed(texts, batch_size=64)):
        results.append(emb)
        done = i + 1
        if done % log_interval == 0 or done == total:
            logging.info(f"Embedding: {done}/{total} chunks")
    return np.array(results, dtype=np.float32)


class EmbeddingToolProvider(ToolProvider):
    """
    Provides semantic_search tool backed by code embeddings.

    On initialize():
    - Loads the embedding model
    - Loads or builds the FAISS index (with incremental cache updates)

    The tool is only registered if initialization succeeds. If dependencies
    are missing or embedding fails, this provider gracefully degrades to
    providing no tools.
    """

    def __init__(self, ctx: ToolContext):
        super().__init__(ctx)
        self._index = None  # faiss.Index, set during initialize()
        self._metadata: dict | None = None
        self._model = None
        self._ready = False

    def initialize(self) -> None:
        try:
            import faiss  # noqa: F811
            import numpy as np  # noqa: F811
            from fastembed import TextEmbedding
        except ImportError as e:
            logging.warning(
                f"Semantic search dependencies not available ({e}); disabled."
            )
            return

        repo_path = self.ctx.repo_path
        changed_files = {f["path"] for f in (self.ctx.changed_files or [])}

        logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            self._model = TextEmbedding(model_name=EMBEDDING_MODEL)
        except Exception as e:
            logging.warning(f"Failed to load embedding model: {e}; semantic search disabled.")
            return

        # Try loading cache — invalidate if model changed
        index, metadata = _load_cache(repo_path)
        if metadata is not None and metadata.get("model") != EMBEDDING_MODEL:
            logging.info(
                f"Cache model mismatch: {metadata.get('model')} != {EMBEDDING_MODEL}. "
                "Rebuilding index."
            )
            index, metadata = None, None

        if index is not None and metadata is not None:
            # Incremental update: remove chunks for changed/deleted files, re-embed them
            file_to_ids = metadata.get("file_to_ids", {})
            chunks = metadata.get("chunks", {})
            next_id = metadata.get("next_id", 0)

            # Collect IDs to remove
            ids_to_remove = []
            for f in changed_files:
                for id_val in file_to_ids.pop(f, []):
                    ids_to_remove.append(id_val)
                    chunks.pop(str(id_val), None)

            if ids_to_remove:
                index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
                logging.info(f"Removed {len(ids_to_remove)} stale chunks from cache")

            # Re-chunk and embed changed files (not deleted ones)
            files_to_embed = [
                f for f in changed_files
                if os.path.exists(os.path.join(repo_path, f))
            ]
        else:
            # Cold start: embed everything
            index = _new_index()
            metadata = {"next_id": 0, "model": EMBEDDING_MODEL, "chunks": {}, "file_to_ids": {}}
            chunks = metadata["chunks"]
            file_to_ids = metadata["file_to_ids"]
            next_id = 0
            files_to_embed = _collect_files(repo_path)
            logging.info(f"Cold start: indexing {len(files_to_embed)} files")

        # Chunk and embed new/changed files
        if files_to_embed:
            all_new_chunks = []
            total = len(files_to_embed)
            for i, f in enumerate(files_to_embed):
                if total > 10 and (i + 1) % max(1, total // 10) == 0:
                    logging.info(f"Chunking files: {i + 1}/{total}")
                all_new_chunks.extend(_chunk_file(repo_path, f))

            if all_new_chunks:
                texts = [c["text"] for c in all_new_chunks]
                embeddings = _embed_texts(self._model, texts)

                new_ids = np.arange(next_id, next_id + len(all_new_chunks), dtype=np.int64)
                index.add_with_ids(embeddings, new_ids)

                for i, chunk in enumerate(all_new_chunks):
                    id_val = int(new_ids[i])
                    chunks[str(id_val)] = {
                        "file": chunk["file"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "text_hash": chunk["text_hash"],
                    }
                    file_to_ids.setdefault(chunk["file"], []).append(id_val)

                next_id += len(all_new_chunks)
                logging.info(f"Embedded {len(all_new_chunks)} new chunks")

        metadata["next_id"] = next_id
        metadata["model"] = EMBEDDING_MODEL
        metadata["chunks"] = chunks
        metadata["file_to_ids"] = file_to_ids

        self._index = index
        self._metadata = metadata
        self._ready = True

        # Save updated cache
        _save_cache(repo_path, index, metadata)

        logging.info(
            f"Embedding index ready: {index.ntotal} vectors, "
            f"{len(file_to_ids)} files"
        )

    def get_tools(self) -> list[Callable]:
        if not self._ready:
            return []

        index = self._index
        metadata = self._metadata
        model = self._model
        max_chars = self.ctx.max_output_chars
        repo_path = self.ctx.repo_path

        def semantic_search(query: str, top_k: int = 10) -> str:
            """Search the codebase using semantic similarity. Finds code that is
            conceptually related to your query, even if it doesn't contain the
            exact keywords. Useful for finding similar patterns, related functions,
            or code that implements a concept described in natural language.

            Use this when grep/ripgrep wouldn't work because you don't know the
            exact identifiers or keywords to search for.

            Args:
                query: Natural language description or code snippet to search for.
                top_k: Number of results to return. Defaults to 10.
            """
            query_embedding = _embed_texts(model, [query])
            k = min(top_k, index.ntotal)
            if k == 0:
                return "No code indexed for semantic search."

            scores, ids = index.search(query_embedding, k)
            chunks = metadata["chunks"]

            results = []
            for score, id_val in zip(scores[0], ids[0]):
                if id_val == -1:
                    continue
                chunk_meta = chunks.get(str(int(id_val)))
                if not chunk_meta:
                    continue
                results.append(
                    f"[{chunk_meta['file']}:{chunk_meta['start_line']}-"
                    f"{chunk_meta['end_line']}] (score: {score:.3f})"
                )

            if not results:
                return "No relevant results found."

            output = f"# Semantic search: {len(results)} results\n" + "\n".join(results)
            if len(output) > max_chars:
                output = output[:max_chars] + "\n... [truncated]"
            return output

        return [semantic_search]
