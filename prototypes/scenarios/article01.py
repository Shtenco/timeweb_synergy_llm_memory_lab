"""Small synthetic retrieval trace; it does not run an LLM."""

from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path


def _load_stand():
    root = Path(__file__).resolve().parents[2]
    source = root / "evidence" / "v6" / "HIAI_v6" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from corpora import projects
    from hiai_bench import Index

    return projects, Index


def run(seed: int = 42) -> dict:
    projects, Index = _load_stand()
    documents, questions, meta = projects(seed=seed)
    question = next(q for q in questions if q["id"] == "P-03-retention")
    index = Index(documents)
    retrieval = {}
    for method in ("bm25", "dense_lsa", "graph"):
        found = index.retrieve(question["question"], method, 3)
        retrieval[method] = {
            "found_ids": found,
            "gold_ids": question["gold"],
            "all_support_found": set(question["gold"]).issubset(found),
        }
    return {
        "evidence_type": "synthetic",
        "corpus_version": "projects_v2",
        "corpus_sha256": hashlib.sha256(json.dumps(
            sorted(documents, key=lambda row: row["id"]), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "dataset_metadata": meta,
        "query_id": question["id"],
        "retrieval": retrieval,
        "limitations": [
            "Only document retrieval is measured; no LLM generation or answer correctness is measured.",
            "The bundled corpus is synthetic and differs from the archived v5 corpus.",
            "LSA is a TF-IDF/SVD representation, not a pretrained embedding model.",
        ],
    }
