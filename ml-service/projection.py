
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.manifold import TSNE

def compute_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    st = SentenceTransformer(model_name)
    emb = st.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def project_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(embeddings)

def project_tsne(embeddings: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random", learning_rate="auto")
    return tsne.fit_transform(embeddings)

def project_texts(texts: List[str], labels: Optional[List[str]] = None, method: str = "umap", **kwargs) -> Dict:
    emb = compute_embeddings(texts)
    if method.lower() == "tsne":
        coords = project_tsne(emb, perplexity=float(kwargs.get("perplexity", 30.0)), random_state=int(kwargs.get("random_state", 42)))
    else:
        coords = project_umap(emb, n_neighbors=int(kwargs.get("n_neighbors", 15)), min_dist=float(kwargs.get("min_dist", 0.1)), random_state=int(kwargs.get("random_state", 42)))
    payload = {"x": coords[:,0].tolist(), "y": coords[:,1].tolist()}
    if labels is not None:
        payload["labels"] = labels
    return payload
