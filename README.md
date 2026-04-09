# AI·ICT 기술 스캐닝 파이프라인

논문/특허 메타데이터를 결합해 전처리(소문자, 불용어 제거, 토큰화, 간단 표제어화), TF-IDF 기반 토픽 도출, 토픽 유사도/클러스터링을 수행합니다.

## 실행

```bash
python tech_scanning.py \
  --papers data/sample_papers.csv \
  --patents data/sample_patents.csv \
  --n-topics 10 \
  --n-clusters 4 \
  --out outputs/tech_topics.md
```

## 입력 스키마

- `papers.csv`: `title`, `abstract`, `keywords`(선택), `year`(선택)
- `patents.csv`: `title`, `abstract`, `CPC`/`IPC`(선택), `year`(선택)

## 출력

- 마크다운 표: 기술 토픽 + 기술 클러스터 (`outputs/tech_topics.md`)
- 토픽 유사도 행렬 (`outputs/tech_topics.similarity.csv`)
- 토픽 유사도 히트맵 SVG (`outputs/tech_topics_heatmap.svg`)
- 기술 클러스터 맵 SVG (`outputs/tech_topics_clusters.svg`)
- 대시보드 HTML (`outputs/tech_topics_dashboard.html`)
