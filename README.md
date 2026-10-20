# Hybrid Search Engine

A hybrid information retrieval system combining **BM25** (Elasticsearch) and **ColBERTv2** with **Reciprocal Rank Fusion (RRF)**, evaluated zero-shot on three BEIR datasets: TREC-COVID (biomedical), HotpotQA (multi-hop QA) and FiQA (finance).

**Research question:** Can combining lexical and semantic retrieval improve robustness across domains compared to sparse or dense methods alone?

## Key Findings

- ColBERT consistently outperformed BM25 on all three datasets.
- RRF hybrid gave the best nDCG@10 and Recall@100 on HotpotQA, but did not beat ColBERT alone on FiQA or TREC-COVID.
- Dense retrieval generalised well across domains without any fine-tuning.

## How It Works

1. **Indexing:** documents are cleaned, then indexed in Elasticsearch (BM25) and encoded with ColBERTv2 (via RAGatouille).
2. **Retrieval:** BM25 and ColBERT each rank documents independently.
3. **Fusion:** RRF merges the two rankings by rank position into one final list.

# Live Demo

**Try it here: https://hybridsearchengine.streamlit.app/**

A Streamlit app compares BM25, ColBERT and RRF results side by side, with MRR@3 computed against ground-truth judgements.

```bash
streamlit run app.py
```

## Authors

Saif Ali Khan Mangan · Marharyta Dzhafarova · Haadi Muhammad Iftikhar



