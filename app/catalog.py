"""
Catalog store + retrieval.

Design choice: TF-IDF (character+word n-grams) over "name + test_type_label"
instead of a neural embedding index. Reasoning (see APPROACH.md):
  - Zero external embedding-API calls -> no extra latency/quota inside the
    30s-per-call budget, and no cold-start model download on free hosting.
  - Assessment names are short, technical, and keyword-dense ("Java 8 (New)",
    "SQL Server", "OPQ32r") so lexical overlap is already a strong signal.
  - The LLM re-ranks/selects from the shortlist TF-IDF returns, so we get
    semantic understanding at the *selection* step without paying for it at
    the *retrieval* step for all 258 items on every turn.
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"

TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


class Catalog:
    def __init__(self, path: Path = DATA_PATH):
        self.items: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        for it in self.items:
            it["search_text"] = f"{it['name']} {it['test_type_label']} {it['test_type_code']}"
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w.+#-]*\b",  # keep tokens like "c++", "c#", ".net"
        )
        self._matrix = self._vectorizer.fit_transform([it["search_text"] for it in self.items])
        self._url_index = {it["url"]: it for it in self.items}
        self._name_index = {it["name"].lower(): it for it in self.items}

    def size(self) -> int:
        return len(self.items)

    def by_url(self, url: str) -> Dict[str, Any] | None:
        return self._url_index.get(url)

    def search(self, query: str, top_k: int = 40) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = []
        for i in ranked[: top_k * 2]:
            if sims[i] <= 0:
                continue
            results.append(self.items[i])
            if len(results) >= top_k:
                break
        return results

    def fuzzy_lookup(self, name_fragment: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Used for /compare grounding: 'OPQ' -> OPQ32r, 'GSA' -> Global Skills Assessment, etc."""
        frag = name_fragment.lower().strip()
        if not frag:
            return []
        scored = []
        for it in self.items:
            name_l = it["name"].lower()
            if frag in name_l:
                score = 1.0
            else:
                score = SequenceMatcher(None, frag, name_l).ratio()
            if score > 0.35:
                scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def as_prompt_rows(self, items: List[Dict[str, Any]]) -> str:
        lines = []
        for it in items:
            lines.append(
                f"- name: {it['name']} | url: {it['url']} | type: {it['test_type_code']} "
                f"({it['test_type_label']}) | remote_testing: {it['remote_testing']} | "
                f"adaptive_irt: {it['adaptive_irt']}"
            )
        return "\n".join(lines)


_catalog_singleton: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton
