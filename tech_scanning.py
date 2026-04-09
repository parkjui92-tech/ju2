#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "across", "using", "based",
    "method", "system", "models", "model", "data", "analysis", "approach", "learning", "deep",
    "network", "neural", "ai", "ict", "on", "of", "to", "in", "a", "an", "is", "are",
}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_year(value: str) -> int | None:
    if not value:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None


def preprocess(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
    out = []
    for tok in tokens:
        for s in ("ing", "ed", "es", "s"):
            if tok.endswith(s) and len(tok) > len(s) + 2:
                tok = tok[: -len(s)]
                break
        if tok not in STOPWORDS:
            out.append(tok)
    return out


def load_docs(papers_path: Path, patents_path: Path):
    papers = read_csv(papers_path)
    patents = read_csv(patents_path)
    docs = []
    for row in papers:
        text = f"{row.get('title','')} {row.get('abstract','')} {row.get('keywords','')}"
        year = extract_year(row.get("year") or row.get("publication_year") or row.get("date", ""))
        docs.append({"source": "paper", "text": text, "tokens": preprocess(text), "year": year})
    for row in patents:
        kw = f"{row.get('keywords','')} {row.get('CPC','')} {row.get('IPC','')}"
        text = f"{row.get('title','')} {row.get('abstract','')} {kw}"
        year = extract_year(row.get("year") or row.get("filing_date") or row.get("date", ""))
        docs.append({"source": "patent", "text": text, "tokens": preprocess(text), "year": year})
    return docs


def build_vocab(docs, max_terms=1200):
    counts = Counter()
    for d in docs:
        counts.update(set(d["tokens"]))
    return [w for w, _ in counts.most_common(max_terms)]


def tfidf_vectors(docs, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    df = [0] * len(vocab)
    tfs = []
    for d in docs:
        c = Counter(d["tokens"])
        tf = [0.0] * len(vocab)
        for w, v in c.items():
            if w in idx:
                tf[idx[w]] = v
        for j, v in enumerate(tf):
            if v > 0:
                df[j] += 1
        total = max(1, sum(c.values()))
        tfs.append([v / total for v in tf])
    idf = [math.log((1 + n) / (1 + d)) + 1 for d in df]
    return [[t[j] * idf[j] for j in range(len(vocab))] for t in tfs]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def kmeans(vectors, k, iters=30):
    rnd = random.Random(42)
    k = min(k, len(vectors))
    centers = [vectors[i][:] for i in rnd.sample(range(len(vectors)), k)]
    labels = [0] * len(vectors)
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            sims = [cosine(v, c) for c in centers]
            nl = max(range(k), key=lambda x: sims[x])
            if labels[i] != nl:
                labels[i] = nl
                changed = True
        groups = [[] for _ in range(k)]
        for l, v in zip(labels, vectors):
            groups[l].append(v)
        for j in range(k):
            if groups[j]:
                centers[j] = [sum(vals) / len(vals) for vals in zip(*groups[j])]
        if not changed:
            break
    return labels, centers


def infer_trend(years):
    vals = sorted(y for y in years if y)
    if len(vals) < 2:
        return "판단 불가"
    counts = Counter(vals)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs) + 1e-9
    slope = num / den
    if slope > 0.2:
        return "증가"
    if slope < -0.2:
        return "감소"
    return "정체"


def infer_industry(words):
    text = " ".join(words)
    rules = {
        "Healthcare/Bio": ["medical", "hospital", "drug", "diagnosi", "genom"],
        "Manufacturing/Robotics": ["factory", "industrial", "robot", "maintenance"],
        "Mobility/Automotive": ["vehicle", "autonomou", "driv", "traffic"],
        "Telecom/Network": ["telecom", "5g", "wireles", "slice"],
        "Security": ["security", "attack", "intrusion", "privacy", "encrypt"],
    }
    result = [k for k, v in rules.items() if any(t in text for t in v)]
    return ", ".join(result) if result else "범용 ICT"


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def score_to_color(score: float) -> str:
    score = max(0.0, min(1.0, score))
    red = int(255 * (1 - score))
    green = int(255 * (1 - score))
    return f"rgb({red},{green},255)"


def write_similarity_svg(sim_matrix, topic_ids, path: Path):
    n = len(topic_ids)
    cell = 52
    margin_left = 110
    margin_top = 90
    width = margin_left + n * cell + 30
    height = margin_top + n * cell + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="12" y="24" font-size="16" font-family="Arial" font-weight="bold">Topic Similarity Heatmap</text>',
    ]
    for i, tid in enumerate(topic_ids):
        x = margin_left + i * cell + cell / 2
        y = margin_top - 10
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="11" font-family="Arial">T{tid}</text>')
        lx = margin_left - 12
        ly = margin_top + i * cell + cell / 2 + 4
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="end" font-size="11" font-family="Arial">T{tid}</text>')
    for i in range(n):
        for j in range(n):
            score = sim_matrix[i][j]
            x = margin_left + j * cell
            y = margin_top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{score_to_color(score)}" stroke="#ddd"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" font-size="10" font-family="Arial">{score:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_cluster_svg(cluster_map, topic_keywords, path: Path):
    cluster_ids = sorted(cluster_map)
    topics = sorted({t for ts in cluster_map.values() for t in ts})
    width = 980
    height = max(420, 120 + max(len(cluster_ids), len(topics)) * 52)
    cx = 180
    tx = 640
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-size="18" font-family="Arial" font-weight="bold">Technology Cluster Map</text>',
        '<text x="20" y="52" font-size="12" font-family="Arial">Edges connect clusters to included topics.</text>',
    ]
    cluster_pos, topic_pos = {}, {}
    for idx, cid in enumerate(cluster_ids):
        y = 100 + idx * 70
        cluster_pos[cid] = (cx, y)
        parts.append(f'<circle cx="{cx}" cy="{y}" r="24" fill="#2d7ff9" opacity="0.9"/>')
        parts.append(f'<text x="{cx}" y="{y+4}" text-anchor="middle" font-size="12" fill="white" font-family="Arial">C{cid}</text>')
    for idx, tid in enumerate(topics):
        y = 90 + idx * 52
        topic_pos[tid] = (tx, y)
        kw = ", ".join(topic_keywords[tid][:3])
        parts.append(f'<rect x="{tx-32}" y="{y-16}" width="64" height="32" rx="8" fill="#f2f5fb" stroke="#7a8aa0"/>')
        parts.append(f'<text x="{tx}" y="{y-2}" text-anchor="middle" font-size="11" font-family="Arial">Topic {tid}</text>')
        parts.append(f'<text x="{tx+52}" y="{y+4}" font-size="10" fill="#444" font-family="Arial">{kw}</text>')
    for cid, tids in cluster_map.items():
        x1, y1 = cluster_pos[cid]
        for tid in tids:
            x2, y2 = topic_pos[tid]
            parts.append(f'<line x1="{x1+24}" y1="{y1}" x2="{x2-32}" y2="{y2}" stroke="#9bb7e5" stroke-width="1.5"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_dashboard_html(topic_rows, cluster_rows, heatmap_path: Path, cluster_svg_path: Path, out_path: Path):
    topic_html = "\n".join(["<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in topic_rows])
    cluster_html = "\n".join(["<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cluster_rows])
    html = f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AI·ICT 기술 스캐닝 대시보드</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #1f2937; }}
    .cards {{ display:grid; grid-template-columns: repeat(3, minmax(180px,1fr)); gap:12px; margin-bottom: 20px; }}
    .card {{ background:white; border-radius:10px; padding:14px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
    h1,h2 {{ margin: 8px 0 12px; }}
    .panel {{ background:white; border-radius:12px; padding:16px; margin-bottom:16px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
    table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
    th,td {{ border:1px solid #e5e7eb; padding:8px; text-align:left; vertical-align:top; }}
    th {{ background:#f1f5f9; }}
    img {{ width: 100%; max-width: 980px; border:1px solid #e5e7eb; border-radius:8px; background:white; }}
    .muted {{ color:#64748b; font-size:12px; }}
  </style>
</head>
<body>
  <h1>AI·ICT 기술 스캐닝 대시보드</h1>
  <div class=\"cards\">
    <div class=\"card\"><div class=\"muted\">기술 토픽 수</div><div><b>{len(topic_rows)}</b></div></div>
    <div class=\"card\"><div class=\"muted\">기술 클러스터 수</div><div><b>{len(cluster_rows)}</b></div></div>
    <div class=\"card\"><div class=\"muted\">생성 산출물</div><div><b>Markdown + CSV + SVG + HTML</b></div></div>
  </div>

  <section class=\"panel\">
    <h2>기술 토픽</h2>
    <table>
      <thead><tr><th>기술 토픽</th><th>대표 키워드</th><th>논문/특허 증가 추세</th><th>주요 응용 산업</th></tr></thead>
      <tbody>{topic_html}</tbody>
    </table>
  </section>

  <section class=\"panel\">
    <h2>기술 클러스터</h2>
    <table>
      <thead><tr><th>기술 클러스터</th><th>포함 기술 토픽</th><th>대표 기술</th><th>주요 산업</th></tr></thead>
      <tbody>{cluster_html}</tbody>
    </table>
  </section>

  <section class=\"panel\">
    <h2>토픽 유사도 히트맵</h2>
    <img src=\"{heatmap_path.name}\" alt=\"topic similarity heatmap\" />
  </section>

  <section class=\"panel\">
    <h2>기술 클러스터 맵</h2>
    <img src=\"{cluster_svg_path.name}\" alt=\"technology cluster map\" />
  </section>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--papers", type=Path, required=True)
    p.add_argument("--patents", type=Path, required=True)
    p.add_argument("--n-topics", type=int, default=12)
    p.add_argument("--n-clusters", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("outputs/tech_topics.md"))
    args = p.parse_args()

    docs = load_docs(args.papers, args.patents)
    vocab = build_vocab(docs)
    vectors = tfidf_vectors(docs, vocab)

    labels, centers = kmeans(vectors, args.n_topics)
    topic_docs = defaultdict(list)
    for i, l in enumerate(labels):
        topic_docs[l].append(i)

    topic_rows = []
    topic_keywords = {}
    topic_vectors = []
    for t in sorted(topic_docs):
        c = centers[t]
        top = sorted(range(len(c)), key=lambda i: c[i], reverse=True)[:10]
        words = [vocab[i] for i in top]
        topic_keywords[t] = words
        years = [docs[i]["year"] for i in topic_docs[t]]
        topic_rows.append([f"Topic {t}", ", ".join(words[:8]), infer_trend(years), infer_industry(words)])
        topic_vectors.append(c)

    tlabels, _ = kmeans(topic_vectors, min(args.n_clusters, len(topic_vectors)))
    cluster_map = defaultdict(list)
    for idx, cl in enumerate(tlabels):
        topic_id = sorted(topic_docs)[idx]
        cluster_map[cl].append(topic_id)

    cluster_rows = []
    for cl in sorted(cluster_map):
        tids = cluster_map[cl]
        rep = topic_keywords[tids[0]][:3]
        inds = sorted({infer_industry(topic_keywords[t]) for t in tids})
        cluster_rows.append([f"Cluster {cl}", ", ".join(f"Topic {t}" for t in tids), ", ".join(rep), "; ".join(inds)])

    keys = sorted(topic_docs)
    sim_matrix = [[cosine(topic_vectors[i], topic_vectors[j]) for j in range(len(keys))] for i in range(len(keys))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sim_path = args.out.with_suffix(".similarity.csv")
    with sim_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic"] + [f"Topic {k}" for k in keys])
        for i, k1 in enumerate(keys):
            w.writerow([f"Topic {k1}"] + [f"{x:.4f}" for x in sim_matrix[i]])

    heatmap_path = args.out.with_name(args.out.stem + "_heatmap.svg")
    cluster_svg_path = args.out.with_name(args.out.stem + "_clusters.svg")
    dashboard_path = args.out.with_name(args.out.stem + "_dashboard.html")
    write_similarity_svg(sim_matrix, keys, heatmap_path)
    write_cluster_svg(cluster_map, topic_keywords, cluster_svg_path)
    write_dashboard_html(topic_rows, cluster_rows, heatmap_path, cluster_svg_path, dashboard_path)

    with args.out.open("w", encoding="utf-8") as f:
        f.write("# AI·ICT 기술 스캐닝 결과\n\n")
        f.write("## 기술 토픽\n\n")
        f.write(markdown_table(["기술 토픽", "대표 키워드", "논문/특허 증가 추세", "주요 응용 산업"], topic_rows))
        f.write("\n\n## 기술 클러스터\n\n")
        f.write(markdown_table(["기술 클러스터", "포함 기술 토픽", "대표 기술", "주요 산업"], cluster_rows))
        f.write("\n\n## 시각화\n\n")
        f.write(f"- Topic Similarity Heatmap: `{heatmap_path}`\n")
        f.write(f"- Technology Cluster Map: `{cluster_svg_path}`\n")
        f.write(f"- Dashboard HTML: `{dashboard_path}`\n")

    print(f"완료: {args.out}")


if __name__ == "__main__":
    main()
