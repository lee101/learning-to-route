import hashlib
import os
from pathlib import Path

import numpy as np


def _find_pybed_model() -> Path | None:
    env = os.environ.get("LTR_PYBED_MODEL")
    if env and Path(env).exists():
        return Path(env)
    try:
        import pybed

        p = Path(pybed.__file__).resolve().parents[1] / "model"
        if p.exists():
            return p
    except Exception:
        pass
    return None


class Embedder:
    def __init__(self, backend: str = "auto"):
        self.backend = None
        self._model = None
        if backend in ("auto", "pybed"):
            try:
                from pybed import EmbedModel

                model_dir = _find_pybed_model()
                if model_dir is None:
                    raise FileNotFoundError("pybed model dir not found")
                self._model = EmbedModel.from_dir(model_dir)
                self.backend = "pybed"
            except Exception:
                if backend == "pybed":
                    raise
        if self.backend is None and backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    "sentence-transformers/static-retrieval-mrl-en-v1",
                    truncate_dim=512,
                )
                self.backend = "st"
            except Exception:
                if backend == "st":
                    raise
        if self.backend is None:
            self.backend = "hash"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "pybed":
            vecs = self._model.embed_batch(texts).astype(np.float32)
        elif self.backend == "st":
            vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        else:
            vecs = np.stack([self._hash_embed(t) for t in texts])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @staticmethod
    def _hash_embed(text: str, dim: int = 512) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            vec[h % dim] += 1.0 if (h >> 32) % 2 else -1.0
        return vec
