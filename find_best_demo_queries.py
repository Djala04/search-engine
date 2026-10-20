import json
import csv
from pathlib import Path

DATASETS = {
    "HotpotQA": {
        "json": "hotpotqa_demo_results_top10.json",
        "tsv": "hotpotqa_test.tsv",
    },
    "FiQA-2018": {
        "json": "fiqa_demo_results.json",
        "tsv": "fiqa_test.tsv",
    },
    "TREC-COVID": {
        "json": "trec-covid_demo_results.json",
        "tsv": "trec-covid_test.tsv",
    },
}

def load_qrels(tsv_path):
    qrels = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(qid, set()).add(str(did))
    return qrels

def mrr_at_k(results, relevant_doc_ids, k=10):
    for rank, item in enumerate(results[:k], start=1):
        if str(item.get("doc_id")) in relevant_doc_ids:
            return 1.0 / rank
    return 0.0

all_winners = []

for ds_name, paths in DATASETS.items():
    with open(paths["json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    qrels = load_qrels(paths["tsv"])

    dataset_winners = []

    for query_text, payload in data.items():
        qid = payload.get("query_id")
        relevant = qrels.get(str(qid), set()) or qrels.get(qid, set())
        if not relevant:
            continue

        bm25 = mrr_at_k(payload.get("bm25", []), relevant, k=10)
        colbert = mrr_at_k(payload.get("colbert", []), relevant, k=10)
        rrf = mrr_at_k(payload.get("rrf", []), relevant, k=10)

        # strict win: RRF must beat both
        if rrf > bm25 and rrf > colbert:
            dataset_winners.append({
                "dataset": ds_name,
                "query": query_text,
                "bm25": bm25,
                "colbert": colbert,
                "rrf": rrf,
                "gain_vs_best_base": rrf - max(bm25, colbert),
            })

    dataset_winners.sort(
        key=lambda x: (x["gain_vs_best_base"], x["rrf"]),
        reverse=True
    )

    all_winners.extend(dataset_winners)

    print(f"\n{'='*110}")
    print(f"{ds_name} — queries where RRF beats BOTH BM25 and ColBERT")
    print(f"{'='*110}")
    print(f"Found: {len(dataset_winners)}\n")

    for item in dataset_winners[:15]:
        print("-" * 110)
        print(item["query"])
        print(f"BM25 MRR@10:    {item['bm25']:.3f}")
        print(f"ColBERT MRR@10: {item['colbert']:.3f}")
        print(f"RRF MRR@10:     {item['rrf']:.3f}")
        print(f"Gain vs best base: {item['gain_vs_best_base']:.3f}")

print(f"\n{'#'*110}")
print("TOP WINNERS ACROSS ALL 3 DATASETS")
print(f"{'#'*110}")

all_winners.sort(
    key=lambda x: (x["gain_vs_best_base"], x["rrf"]),
    reverse=True
)

for item in all_winners[:20]:
    print("-" * 110)
    print(f"[{item['dataset']}] {item['query']}")
    print(f"BM25 MRR@10:    {item['bm25']:.3f}")
    print(f"ColBERT MRR@10: {item['colbert']:.3f}")
    print(f"RRF MRR@10:     {item['rrf']:.3f}")
    print(f"Gain vs best base: {item['gain_vs_best_base']:.3f}")