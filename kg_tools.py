"""MMKG tool implementations — Streamlit Cloud 배포용.

로컬 Stable Diffusion 대신 HuggingFace Inference API를 사용합니다.
데이터 파일은 data/ 폴더에서 읽습니다.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"

_OBJ_SYMBOL_PATH  = DATA_DIR / "object_symbol_mapping.csv"
_SUBJ_SYMBOL_PATH = DATA_DIR / "subject_symbol_correlation.csv"
_RAW_PATH         = DATA_DIR / "minhwa_raw_combined.csv"
_KG_PAIRS_PATH    = DATA_DIR / "symbol_pairs_input_check4.csv"
_GRAPHML_PATH     = DATA_DIR / "minhwa_kg.graphml"

GENERIC    = {"한국전통민화", "민화", "전통", "기타", ""}
NON_SYMBOL = {"그림", "도상", "동물", "식물", "화훼", "화목", "문자", "장식", "행사",
              "교육", "생활", "용도", "한지", "비단", "일필", "공필", "채색", "수묵"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _split(val: str | None) -> list[str]:
    if not val:
        return []
    return [t.strip() for t in val.replace(";", ",").split(",")
            if t.strip() and t.strip() not in GENERIC]


# ── 1. 모티프 → 상징 ──────────────────────────────────────────────────────────
def get_motif_symbols(motif: str, top_k: int = 10) -> dict[str, Any]:
    rows    = _read_csv(_OBJ_SYMBOL_PATH)
    kg_rows = _read_csv(_KG_PAIRS_PATH)

    obj_symbols: list[dict] = []
    for row in rows:
        if motif.strip() in (row.get("Visual Object (객체)", "") or "").strip():
            sym = (row.get("Symbolic Meaning (상징)") or "").strip()
            cnt = int(float(row.get("Co-occurrence", "0") or 0))
            if sym and sym not in NON_SYMBOL:
                obj_symbols.append({"symbol": sym, "co_occurrence": cnt,
                                    "source": "object_symbol_mapping"})

    kg_symbols: list[str] = []
    for row in kg_rows:
        m = row.get("motif", "").replace("motif:", "").strip()
        if motif.strip() in m:
            s = row.get("symbol", "").replace("symbol:", "").strip()
            if s and s not in NON_SYMBOL:
                kg_symbols.append(s)

    obj_symbols.sort(key=lambda x: x["co_occurrence"], reverse=True)
    return {
        "motif": motif,
        "object_symbol_mapping": obj_symbols[:top_k],
        "kg_pairs": list(set(kg_symbols))[:top_k],
        "note": f"공출현 {len(obj_symbols)}건, KG쌍 {len(set(kg_symbols))}건",
    }


# ── 2. 화목 → 상징 ────────────────────────────────────────────────────────────
def get_genre_symbols(genre: str, top_k: int = 10) -> dict[str, Any]:
    rows = _read_csv(_SUBJ_SYMBOL_PATH)
    results: list[dict] = []
    for row in rows:
        g = (row.get("Genre (화목)") or "").strip()
        if genre.strip() in g or g in genre.strip():
            sym  = (row.get("Symbol (상징어)") or "").strip()
            freq = int(float(row.get("Frequency (빈도)", "0") or 0))
            ratio = float(row.get("Ratio (%)", "0") or 0)
            if sym and sym not in NON_SYMBOL:
                results.append({"symbol": sym, "frequency": freq, "ratio_pct": ratio})
    results.sort(key=lambda x: x["frequency"], reverse=True)
    return {
        "genre": genre,
        "symbols": results[:top_k],
        "note": f"장르 '{genre}'에서 {len(results)}개 상징 상관관계 발견",
    }


# ── 3. 작품 검색 ─────────────────────────────────────────────────────────────
def search_artworks(query: str, field: str = "any", limit: int = 10) -> dict[str, Any]:
    rows = _read_csv(_RAW_PATH)
    search_fields = (
        ["subject_type", "objects", "keywords", "purpose", "era", "theme_type", "painting_type"]
        if field == "any" else [field]
    )
    matched: list[dict] = []
    for row in rows:
        for f in search_fields:
            if query.lower() in (row.get(f) or "").lower():
                matched.append({
                    "file_name":    row.get("file_name", ""),
                    "subject_type": row.get("subject_type", ""),
                    "objects":      row.get("objects", ""),
                    "keywords":     row.get("keywords", ""),
                    "purpose":      row.get("purpose", ""),
                    "era":          row.get("era", ""),
                })
                break
    return {"query": query, "field": field, "total_found": len(matched), "results": matched[:limit]}


# ── 4. 작품 상징 추론 ─────────────────────────────────────────────────────────
def infer_artwork_symbols(file_name: str, top_k: int = 5) -> dict[str, Any]:
    rows = _read_csv(_RAW_PATH)
    row  = next((r for r in rows if r.get("file_name", "").strip() == file_name.strip()), None)
    if row is None:
        return {"error": f"'{file_name}' 작품을 데이터셋에서 찾을 수 없습니다.", "file_name": file_name}

    obj_data  = _read_csv(_OBJ_SYMBOL_PATH)
    subj_data = _read_csv(_SUBJ_SYMBOL_PATH)
    kg_data   = _read_csv(_KG_PAIRS_PATH)

    obj_to_syms: dict[str, list[dict]] = defaultdict(list)
    max_co = 1
    for r in obj_data:
        obj = (r.get("Visual Object (객체)") or "").strip()
        sym = (r.get("Symbolic Meaning (상징)") or "").strip()
        cnt = int(float(r.get("Co-occurrence", "0") or 0))
        max_co = max(max_co, cnt)
        if obj and sym and sym not in NON_SYMBOL:
            obj_to_syms[obj].append({"symbol": sym, "count": cnt})

    subj_to_syms: dict[str, list[dict]] = defaultdict(list)
    for r in subj_data:
        g   = (r.get("Genre (화목)") or "").strip()
        sym = (r.get("Symbol (상징어)") or "").strip()
        freq  = int(float(r.get("Frequency (빈도)", "0") or 0))
        ratio = float(r.get("Ratio (%)", "0") or 0)
        if g and sym and sym not in NON_SYMBOL:
            subj_to_syms[g].append({"symbol": sym, "frequency": freq, "ratio": ratio})

    kg_motif_to_syms: dict[str, set[str]] = defaultdict(set)
    for r in kg_data:
        m = r.get("motif", "").replace("motif:", "").strip()
        s = r.get("symbol", "").replace("symbol:", "").strip()
        if m and s and s not in NON_SYMBOL:
            kg_motif_to_syms[m].add(s)

    candidates: dict[str, list[dict]] = defaultdict(list)
    objects      = _split(row.get("objects"))
    keywords     = _split(row.get("keywords"))
    subject_type = (row.get("subject_type") or "").strip()
    blocked      = set(objects) | set(_split(row.get("subject_type"))) | set(_split(row.get("theme_type")))

    for kw in keywords:
        if kw not in blocked and kw not in NON_SYMBOL:
            candidates[kw].append({"source": "keyword", "detail": f"keyword: {kw}", "weight": 1.0})

    for obj in objects:
        for item in obj_to_syms.get(obj, []):
            w = 0.55 + 0.35 * (item["count"] / max_co)
            candidates[item["symbol"]].append({
                "source": "object_symbol_mapping",
                "detail": f"{obj}->{item['symbol']} co={item['count']}",
                "weight": round(w, 4),
            })
        for sym in kg_motif_to_syms.get(obj, set()):
            candidates[sym].append({
                "source": "kg_pair",
                "detail": f"{obj}->{sym} (KG triple)",
                "weight": 0.72,
            })

    for item in subj_to_syms.get(subject_type, [])[:8]:
        w = min(0.68, 0.25 + item["ratio"] / 100)
        candidates[item["symbol"]].append({
            "source": "genre_correlation",
            "detail": f"{subject_type}->{item['symbol']} ratio={item['ratio']}%",
            "weight": round(w, 4),
        })

    def noisy_or(items: list[dict]) -> float:
        p = 1.0
        for it in items:
            p *= 1.0 - min(max(it["weight"], 0.0), 0.99)
        return round(1.0 - p, 4)

    ranked = sorted(
        [{"symbol": s, "confidence": noisy_or(ev), "evidence": ev}
         for s, ev in candidates.items()],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:top_k]

    return {
        "file_name": file_name,
        "subject_type": subject_type,
        "objects": objects,
        "keywords": keywords,
        "inferred_symbols": ranked,
        "mmkg_sources_used": ["object_symbol_mapping", "subject_symbol_correlation",
                              "kg_pairs", "keywords"],
    }


# ── 5. KG 탐색 ───────────────────────────────────────────────────────────────
def query_kg_neighborhood(node_name: str, max_hops: int = 1) -> dict[str, Any]:
    try:
        import networkx as nx
        if not _GRAPHML_PATH.exists():
            return {"error": "minhwa_kg.graphml 파일 없음", "node": node_name}
        G = nx.read_graphml(str(_GRAPHML_PATH))
    except Exception as e:
        return {"error": str(e), "node": node_name}

    matched_nodes = [n for n in G.nodes
                     if node_name.lower() in str(n).lower()
                     or node_name.lower() in str(G.nodes[n].get("label", "")).lower()]
    if not matched_nodes:
        return {"node": node_name, "found": False, "neighbors": [],
                "note": "KG에서 매칭 노드를 찾지 못했습니다."}

    seed = matched_nodes[0]
    neighbors, visited, frontier = [], {seed}, [seed]
    for _ in range(max_hops):
        next_frontier = []
        for n in frontier:
            for nbr in list(G.successors(n)) + list(G.predecessors(n)):
                if nbr not in visited:
                    edge_data = G.get_edge_data(n, nbr) or G.get_edge_data(nbr, n) or {}
                    neighbors.append({
                        "from": n, "to": nbr,
                        "relation": edge_data.get("label", edge_data.get("type", "related")),
                    })
                    visited.add(nbr)
                    next_frontier.append(nbr)
        frontier = next_frontier

    return {
        "node": node_name,
        "matched_node_id": seed,
        "found": True,
        "hop": max_hops,
        "neighbors": neighbors[:30],
        "total_neighbors": len(neighbors),
    }


# ── 6. 이미지 생성 — HuggingFace Inference API ───────────────────────────────
_MINHWA_STYLE_PREFIX = (
    "Korean traditional folk painting (민화 Minhwa), "
    "flat perspective, vibrant mineral pigments, decorative patterns, "
    "auspicious symbolism, Joseon dynasty art style, "
    "bold outlines, folk art aesthetic, hanji paper texture"
)
_NEGATIVE_PROMPT = (
    "photograph, realistic, 3d render, western art, modern, text, watermark, "
    "blurry, low quality, dark, violent"
)

_SYMBOL_EN: dict[str, str] = {
    "장수": "longevity", "부귀": "wealth and prosperity", "길상": "auspicious luck",
    "벽사": "warding off evil spirits", "다산": "fertility", "출세": "success",
    "번성": "prosperity", "절개": "integrity", "청렴": "purity", "화목": "harmony",
    "풍요": "abundance", "불로장생": "eternal youth and longevity",
    "학": "crane", "모란": "peony", "잉어": "carp", "호랑이": "tiger",
    "소나무": "pine tree", "대나무": "bamboo", "매화": "plum blossom",
    "연꽃": "lotus", "국화": "chrysanthemum", "봉황": "phoenix",
    "거북": "turtle", "사슴": "deer", "불로초": "herb of immortality",
    "까치": "magpie", "복숭아": "peach",
}


def _ko_to_en(term: str) -> str:
    return _SYMBOL_EN.get(term.strip(), term.strip())


def _build_prompt(motifs: list[str], symbols: list[str], genre: str) -> str:
    motif_en  = ", ".join(_ko_to_en(m) for m in motifs  if m)
    symbol_en = ", ".join(_ko_to_en(s) for s in symbols if s)
    genre_en  = _ko_to_en(genre) if genre else "folk painting"

    parts = [_MINHWA_STYLE_PREFIX]
    if genre:
        parts.append(f"{genre_en} genre")
    if motif_en:
        parts.append(f"featuring {motif_en}")
    if symbol_en:
        parts.append(f"symbolizing {symbol_en}")
    return ", ".join(parts)


def generate_minhwa_image(
    motifs: list[str],
    symbols: list[str],
    genre: str = "",
    output_filename: str = "",
    width: int = 512,
    height: int = 512,
    hf_token: str = "",
    **_kwargs,
) -> dict[str, Any]:
    """HuggingFace Inference API로 민화 스타일 이미지 생성."""
    import requests
    import datetime

    if not hf_token:
        return {"error": "HuggingFace token이 필요합니다. Streamlit Cloud → Settings → Secrets에 HF_TOKEN을 추가하세요."}

    prompt = _build_prompt(motifs, symbols, genre)
    api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": _NEGATIVE_PROMPT,
            "width": width,
            "height": height,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    except requests.Timeout:
        return {"error": "HF API 응답 시간 초과 (120초). 잠시 후 다시 시도하세요."}

    if resp.status_code == 503:
        return {"error": "모델 로딩 중입니다 (콜드 스타트). 20~30초 후 다시 시도하세요."}
    if resp.status_code != 200:
        return {"error": f"HF API 오류 {resp.status_code}: {resp.text[:300]}"}

    from PIL import Image as _PIL
    try:
        image = _PIL.open(io.BytesIO(resp.content))
    except Exception as e:
        return {"error": f"이미지 파싱 실패: {e}"}

    # In-memory bytes for Streamlit display
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return {
        "image_bytes": buf,
        "prompt_used": prompt,
        "negative_prompt": _NEGATIVE_PROMPT,
        "model": "runwayml/stable-diffusion-v1-5 (HF API)",
        "device": "HuggingFace Cloud",
        "size": f"{width}x{height}",
        "motifs_input": motifs,
        "symbols_input": symbols,
        "genre_input": genre,
    }
