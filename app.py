"""한국전통민화 MMKG 에이전트 — Streamlit Cloud 배포용

실행:  streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kg_tools import (
    get_motif_symbols,
    get_genre_symbols,
    search_artworks,
    infer_artwork_symbols,
    generate_minhwa_image,
    query_kg_neighborhood,
)

DATA_DIR = Path(__file__).parent / "data"

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="한국전통민화 MMKG 에이전트",
    page_icon="🖌️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f7f1e4; }
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
.metric-card {
    background: #fff8ee; border: 1px solid #d8cbb7;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;
}
.badge-supported { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:12px; }
.badge-plausible { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 8px; font-size:12px; }
.badge-review    { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:12px; }
h1 { color: #1F1B16; }
h2, h3 { color: #243B53; }
</style>
""", unsafe_allow_html=True)

# HF token (선택사항 — 현재 이미지 생성은 pollinations.ai 사용으로 불필요)
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖌️ 민화 MMKG 에이전트")
    st.caption("Multi-Modal Knowledge Graph 기반\n한국전통민화 상징성 추론·검증·이미지 생성")
    st.divider()

    st.markdown("### 데이터 정보")
    raw_path = DATA_DIR / "minhwa_raw_combined.csv"
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
        st.metric("총 작품 수", f"{len(df_raw):,}건")
        st.metric("화목(장르) 수", f"{df_raw['subject_type'].nunique()}종")
    else:
        st.warning("minhwa_raw_combined.csv 없음")

    st.divider()
    st.markdown("### MMKG 증거 소스")
    st.markdown("""
- `object_symbol_mapping.csv` — 모티프↔상징 공출현
- `subject_symbol_correlation.csv` — 화목↔상징 빈도
- `symbol_pairs_input_check4.csv` — KG 트리플
- `minhwa_kg.graphml` — 지식 그래프
""")
    st.divider()
    st.success("이미지 생성: FLUX (pollinations.ai) ✓")

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 상징 추론 (Inference)",
    "📚 작품 검색 (Search)",
    "🎨 이미지 생성 (Generate)",
    "🕸️ KG 탐색 (Graph)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 상징 추론
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("상징 추론 — MMKG 근거 기반")
    st.caption("모티프·화목 기준 상징 조회 또는 특정 작품 MMKG 추론을 실행합니다.")

    mode = st.radio("추론 방식 선택", ["모티프 조회", "화목(장르) 조회", "작품 파일명으로 추론"],
                    horizontal=True)
    st.divider()

    if mode == "모티프 조회":
        col1, col2 = st.columns([3, 1])
        with col1:
            motif = st.text_input("모티프(객체) 이름", placeholder="예: 학, 모란, 호랑이, 잉어")
        with col2:
            top_k = st.number_input("상위 N개", min_value=3, max_value=20, value=10)

        if st.button("조회", key="motif_btn") and motif:
            with st.spinner("MMKG 조회 중…"):
                result = get_motif_symbols(motif, top_k=top_k)

            st.markdown(f"#### `{motif}` 상징 조회 결과")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**공출현 기반**")
                if result["object_symbol_mapping"]:
                    df = pd.DataFrame(result["object_symbol_mapping"])
                    fig = px.bar(df, x="symbol", y="co_occurrence",
                                 color_discrete_sequence=["#B7432D"],
                                 labels={"symbol": "상징어", "co_occurrence": "공출현 빈도"})
                    fig.update_layout(height=300, margin=dict(t=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("공출현 데이터 없음")
            with col_b:
                st.markdown("**KG 트리플 기반**")
                if result["kg_pairs"]:
                    for sym in result["kg_pairs"]:
                        st.markdown(f"- {sym}")
                else:
                    st.info("KG 트리플 없음")
            st.caption(result["note"])

    elif mode == "화목(장르) 조회":
        GENRES = ["화조도", "십장생도", "문자도", "호작도", "어해도", "책거리", "산수도",
                  "화훼도", "설화화", "민속화", "혼성도"]
        col1, col2 = st.columns([3, 1])
        with col1:
            genre = st.selectbox("화목(장르) 선택", GENRES)
            custom = st.text_input("직접 입력 (선택사항)", placeholder="")
            if custom:
                genre = custom
        with col2:
            top_k = st.number_input("상위 N개", min_value=3, max_value=20, value=10, key="genre_topk")

        if st.button("조회", key="genre_btn"):
            with st.spinner("MMKG 조회 중…"):
                result = get_genre_symbols(genre, top_k=top_k)

            if result["symbols"]:
                df = pd.DataFrame(result["symbols"])
                st.markdown(f"#### `{genre}` — 상위 {top_k}개 상징")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig = px.bar(df, x="symbol", y="frequency",
                                 color="ratio_pct", color_continuous_scale="Oranges",
                                 labels={"symbol": "상징어", "frequency": "빈도", "ratio_pct": "비율(%)"},
                                 hover_data=["ratio_pct"])
                    fig.update_layout(height=340, margin=dict(t=20))
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    st.dataframe(df.rename(columns={
                        "symbol": "상징어", "frequency": "빈도", "ratio_pct": "비율(%)"}),
                        use_container_width=True, hide_index=True)
            else:
                st.warning("해당 화목의 데이터가 없습니다.")

    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            file_name = st.text_input("작품 파일명", placeholder="예: AH_0148_1, HJ_0251")
        with col2:
            top_k = st.number_input("상위 N개", min_value=1, max_value=10, value=5, key="infer_topk")

        if st.button("추론 실행", key="infer_btn") and file_name:
            with st.spinner("MMKG 상징 추론 중…"):
                result = infer_artwork_symbols(file_name.strip(), top_k=top_k)

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(f"#### 작품 `{file_name}` 추론 결과")
                m1, m2, m3 = st.columns(3)
                m1.metric("화목", result.get("subject_type") or "미분류")
                m2.metric("등장 객체 수", len(result.get("objects", [])))
                m3.metric("사용 증거 소스", len(result.get("mmkg_sources_used", [])))
                st.markdown("**등장 객체:**  " + "  ·  ".join(result.get("objects", [])))
                st.divider()

                for sym in result.get("inferred_symbols", []):
                    conf = sym["confidence"]
                    if conf >= 0.78:
                        status, badge = "supported", "badge-supported"
                    elif conf >= 0.55:
                        status, badge = "plausible", "badge-plausible"
                    else:
                        status, badge = "needs_review", "badge-review"
                    sources = list({ev["source"] for ev in sym["evidence"]})
                    with st.expander(
                        f"**{sym['symbol']}**  ·  신뢰도 {conf:.3f}  ·  소스 {len(sources)}개",
                        expanded=conf >= 0.9
                    ):
                        st.progress(min(conf, 1.0))
                        st.markdown(f'<span class="{badge}">{status}</span>', unsafe_allow_html=True)
                        st.markdown("**증거:**")
                        for ev in sym["evidence"]:
                            st.markdown(f"- `{ev['source']}` — {ev['detail']} (weight={ev['weight']})")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 작품 검색
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("작품 검색")
    st.caption("민화 원천 데이터에서 키워드로 작품을 검색합니다.")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        query = st.text_input("검색어", placeholder="예: 호랑이, 장수, 부귀")
    with col2:
        field = st.selectbox("검색 필드",
                             ["any", "subject_type", "objects", "keywords", "purpose", "era"],
                             format_func=lambda x: {
                                 "any": "전체", "subject_type": "화목(장르)",
                                 "objects": "등장 객체", "keywords": "키워드·상징",
                                 "purpose": "용도", "era": "시대",
                             }.get(x, x))
    with col3:
        limit = st.number_input("결과 수", min_value=5, max_value=50, value=20)

    if st.button("검색", key="search_btn") and query:
        with st.spinner("검색 중…"):
            result = search_artworks(query, field=field, limit=limit)

        st.markdown(f"**총 {result['total_found']:,}건 발견** (상위 {limit}건 표시)")
        if result["results"]:
            df = pd.DataFrame(result["results"]).rename(columns={
                "file_name": "파일명", "subject_type": "화목", "objects": "객체",
                "keywords": "키워드", "purpose": "용도", "era": "시대",
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("#### 화목 분포")
            genre_counts = df["화목"].value_counts().reset_index()
            genre_counts.columns = ["화목", "건수"]
            fig = px.pie(genre_counts, names="화목", values="건수",
                         color_discrete_sequence=px.colors.qualitative.Antique)
            fig.update_layout(height=320, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("검색 결과가 없습니다.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 이미지 생성
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("민화 스타일 이미지 생성")
    st.caption("MMKG 데이터로 근거를 확인한 뒤 FLUX 모델(pollinations.ai)로 이미지를 생성합니다. API 키 불필요.")

    st.info("워크플로: **상징 근거 확인 → 모티프·상징 선택 → 이미지 생성**")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1단계: 모티프 선택")
        MOTIF_OPTIONS = ["학", "소나무", "모란", "불로초", "해", "구름", "괴석", "대나무",
                         "매화", "국화", "복숭아", "사슴", "거북", "잉어", "호랑이", "까치"]
        selected_motifs = st.multiselect("시각 모티프 선택", MOTIF_OPTIONS,
                                         default=["학", "소나무", "불로초"])
        custom_motifs = st.text_input("직접 추가 (쉼표 구분)", placeholder="예: 동자, 구름")
        if custom_motifs:
            selected_motifs += [m.strip() for m in custom_motifs.split(",") if m.strip()]

        st.markdown("#### 2단계: 상징 선택")
        SYMBOL_OPTIONS = ["불로장생", "장수", "길상", "절개", "부귀", "출세", "벽사",
                          "화목", "다산", "풍요", "번영", "재복"]
        selected_symbols = st.multiselect("표현할 상징", SYMBOL_OPTIONS,
                                          default=["불로장생", "장수"])
        custom_symbols = st.text_input("직접 추가 (쉼표 구분)", placeholder="예: 청렴, 상서로움",
                                       key="sym_custom")
        if custom_symbols:
            selected_symbols += [s.strip() for s in custom_symbols.split(",") if s.strip()]

    with col2:
        st.markdown("#### 3단계: 화목 및 옵션")
        GENRES = ["화조도", "십장생도", "문자도", "호작도", "어해도", "책거리", "산수도",
                  "화훼도", "혼성도", ""]
        genre = st.selectbox("화목 선택 (선택사항)", GENRES,
                             format_func=lambda x: x if x else "없음")
        img_size = st.select_slider("이미지 크기", options=[256, 512, 768], value=512)

        st.divider()
        st.markdown("#### MMKG 근거 미리 확인")
        if selected_motifs and st.button("근거 데이터 조회", key="preview_btn"):
            with st.spinner("MMKG 조회 중…"):
                for m in selected_motifs[:3]:
                    r = get_motif_symbols(m, top_k=5)
                    pairs = r.get("kg_pairs", [])
                    if pairs:
                        st.markdown(f"**{m}** → KG 연결: " + ", ".join(str(p) for p in pairs[:5]))
                    else:
                        st.markdown(f"**{m}** → KG 데이터 없음 (csv 기반 추론 사용)")

    st.divider()

    if st.button("🎨 이미지 생성", key="gen_btn", type="primary",
                 disabled=not (selected_motifs and selected_symbols)):
        with st.spinner("FLUX 모델로 이미지 생성 중… (약 20~40초)"):
            result = generate_minhwa_image(
                motifs=selected_motifs,
                symbols=selected_symbols,
                genre=genre,
                width=img_size,
                height=img_size,
            )

        if "error" in result:
            st.error(result["error"])
        else:
            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(result["image_bytes"], caption="생성된 민화 스타일 이미지",
                         use_container_width=True)
            with col_info:
                st.markdown("**생성 정보**")
                st.markdown(f"- 모티프: {', '.join(result['motifs_input'])}")
                st.markdown(f"- 상징: {', '.join(result['symbols_input'])}")
                st.markdown(f"- 화목: {result['genre_input'] or '미지정'}")
                st.markdown(f"- 모델: `{result['model']}`")
                st.markdown(f"- 크기: {result['size']}")
                st.divider()
                st.markdown("**사용된 프롬프트:**")
                st.code(result["prompt_used"], language=None)

            # 다운로드 버튼
            result["image_bytes"].seek(0)
            st.download_button(
                label="이미지 다운로드",
                data=result["image_bytes"],
                file_name="minhwa_generated.png",
                mime="image/png",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KG 탐색
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("지식 그래프 탐색")
    st.caption("minhwa_kg.graphml에서 노드 인근 관계를 탐색합니다.")

    col1, col2 = st.columns([3, 1])
    with col1:
        node_query = st.text_input("노드 이름", placeholder="예: 학, 잉어, 장수, 불로장생")
    with col2:
        hops = st.number_input("탐색 홉 수", min_value=1, max_value=2, value=1)

    if st.button("탐색", key="kg_btn") and node_query:
        with st.spinner("KG 탐색 중…"):
            result = query_kg_neighborhood(node_query, max_hops=hops)

        if not result.get("found"):
            st.warning(f"'{node_query}' 노드를 KG에서 찾지 못했습니다.")
        else:
            st.success(f"노드 `{result['matched_node_id']}` 발견 — 인접 {result['total_neighbors']}개")
            neighbors = result.get("neighbors", [])
            if neighbors:
                df_n = pd.DataFrame(neighbors)
                st.dataframe(df_n.rename(columns={"from": "출발", "to": "도착", "relation": "관계"}),
                             use_container_width=True, hide_index=True)
                rel_counts = df_n["relation"].value_counts().head(10).reset_index()
                rel_counts.columns = ["관계", "건수"]
                fig = px.bar(rel_counts, x="건수", y="관계", orientation="h",
                             color_discrete_sequence=["#243B53"])
                fig.update_layout(height=300, margin=dict(t=10), yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
