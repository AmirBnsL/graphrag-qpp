"""Dataset loaders for LongMemEval-S, MuSiQue, and HotpotQA."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class QuerySample:
    """Normalized query and per-query candidate corpus representation."""

    query_id: str
    dataset: str                  # "longmemeval_s" | "musique" | "hotpotqa"
    question_type: str            # e.g., "single-session-assistant", "2hop", "bridge"
    question: str
    gold_answers: list[str]
    gold_evidence_ids: list[str]  # Supporting passage/session IDs
    corpus_passages: list[dict[str, Any]] = field(default_factory=list)


def _deterministic_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def load_hotpotqa_slice(max_queries: int = 250, seed: int = 42) -> list[QuerySample]:
    """Load a stratified slice of HotpotQA (distractor validation split) spanning bridge & comparison."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    bridge_rows = [r for r in ds if r.get("type") == "bridge"]
    comp_rows = [r for r in ds if r.get("type") == "comparison"]

    half = max_queries // 2
    selected = bridge_rows[: max_queries - half] + comp_rows[:half]

    samples: list[QuerySample] = []
    for row in selected:
        qid = f"hotpot_{row['id']}"
        supp_titles = set(row["supporting_facts"]["title"])
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]

        passages: list[dict[str, Any]] = []
        gold_ids: list[str] = []
        for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
            pid = f"{qid}_p{idx}"
            clean_sents = [s.strip() for s in sents if s and s.strip()]
            text = f"{title}: " + " ".join(clean_sents)
            passages.append(
                {
                    "passage_id": pid,
                    "title": title,
                    "text": text,
                    "sentences": clean_sents,
                    "timestamp": None,
                }
            )
            if title in supp_titles:
                gold_ids.append(pid)

        samples.append(
            QuerySample(
                query_id=qid,
                dataset="hotpotqa",
                question_type=f"hotpot_{row.get('type', 'bridge')}",
                question=row["question"],
                gold_answers=[row["answer"]],
                gold_evidence_ids=gold_ids,
                corpus_passages=passages,
            )
        )
    return samples


def load_musique_slice(max_queries: int = 250, seed: int = 42) -> list[QuerySample]:
    """Load a stratified slice of MuSiQue-Ans across 2-hop, 3-hop, and 4-hop questions."""
    try:
        ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    except Exception:
        ds = load_dataset("bdsaglam/musique", split="validation")

    by_hop: dict[str, list[dict[str, Any]]] = {"2hop": [], "3hop": [], "4hop": []}
    for row in ds:
        if not row.get("answerable", True):
            continue
        raw_id = str(row.get("id", ""))
        hop_tag = raw_id.split("_")[0] if "hop" in raw_id else "2hop"
        hop_key = hop_tag if hop_tag in by_hop else "2hop"
        by_hop[hop_key].append(row)

    # Stratified allocation: 50% 2-hop, 30% 3-hop, 20% 4-hop
    n2 = int(max_queries * 0.50)
    n3 = int(max_queries * 0.30)
    n4 = max_queries - n2 - n3
    selected = by_hop["2hop"][:n2] + by_hop["3hop"][:n3] + by_hop["4hop"][:n4]
    if len(selected) < max_queries:
        remainder = [r for rows in by_hop.values() for r in rows if r not in selected]
        selected.extend(remainder[: max_queries - len(selected)])

    samples: list[QuerySample] = []
    for row in selected[:max_queries]:
        qid = f"musique_{row['id']}"
        raw_id = str(row.get("id", ""))
        hop_type = raw_id.split("_")[0] if "hop" in raw_id else "multihop"
        paragraphs = row.get("paragraphs", [])

        passages: list[dict[str, Any]] = []
        gold_ids: list[str] = []
        for p in paragraphs:
            idx = p.get("idx", len(passages))
            pid = f"{qid}_p{idx}"
            title = p.get("title", f"Doc {idx}")
            body = p.get("paragraph_text", "").strip()
            text = f"{title}: {body}"
            passages.append(
                {
                    "passage_id": pid,
                    "title": title,
                    "text": text,
                    "sentences": [s.strip() for s in body.split(". ") if s.strip()],
                    "timestamp": None,
                }
            )
            if p.get("is_supporting", False):
                gold_ids.append(pid)

        ans_list = [str(row["answer"])]
        for alias in row.get("answer_aliases", []) or []:
            if alias and str(alias) not in ans_list:
                ans_list.append(str(alias))

        samples.append(
            QuerySample(
                query_id=qid,
                dataset="musique",
                question_type=f"musique_{hop_type}",
                question=row["question"],
                gold_answers=ans_list,
                gold_evidence_ids=gold_ids,
                corpus_passages=passages,
            )
        )
    return samples


def load_longmemeval_slice(
    max_queries: int = 500,
    cache_dir: str = "/content/artifacts/raw_data",
) -> list[QuerySample]:
    """Load LongMemEval-S (500 conversational memory questions across all 6 categories)."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_json = cache_path / "longmemeval_s.json"

    raw_records: list[dict[str, Any]] = []
    if local_json.exists():
        with open(local_json, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
    else:
        # Try HuggingFace datasets or direct HuggingFace Hub JSON download
        urls = [
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json",
        ]
        for url in urls:
            try:
                urllib.request.urlretrieve(url, local_json)
                with open(local_json, "r", encoding="utf-8") as f:
                    raw_records = json.load(f)
                if raw_records:
                    break
            except Exception:
                continue

        if not raw_records:
            ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="train")
            raw_records = list(ds)

    samples: list[QuerySample] = []
    for idx, row in enumerate(raw_records[:max_queries]):
        qid = f"lme_{row.get('question_id', idx)}"
        qtype = str(row.get("question_type", "multi-session"))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        haystack_sessions = row.get("haystack_sessions", [])
        haystack_dates = row.get("haystack_dates", [])
        haystack_ids = row.get("haystack_session_ids", [])
        answer_session_ids = set(str(x) for x in (row.get("answer_session_ids", []) or []))

        passages: list[dict[str, Any]] = []
        gold_ids: list[str] = []

        for s_idx, session in enumerate(haystack_sessions):
            sess_id = str(haystack_ids[s_idx]) if s_idx < len(haystack_ids) else f"sess_{s_idx}"
            sess_date = str(haystack_dates[s_idx]) if s_idx < len(haystack_dates) else ""
            pid = f"{qid}_{sess_id}"

            turns_text: list[str] = []
            has_answer_turn = False
            if isinstance(session, list):
                for turn in session:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user").capitalize()
                        content = str(turn.get("content", "")).strip()
                        if turn.get("has_answer", False):
                            has_answer_turn = True
                        if content:
                            turns_text.append(f"[{role}]: {content}")
                    elif isinstance(turn, str):
                        turns_text.append(turn.strip())
            elif isinstance(session, str):
                turns_text.append(session.strip())

            # Chunk long sessions into ~1200-char conversational windows so retrieval budget is matched
            full_session_str = "\n".join(turns_text)
            header = f"Session ({sess_date}): " if sess_date else "Session: "
            if len(full_session_str) <= 1600:
                passages.append(
                    {
                        "passage_id": pid,
                        "title": f"Session {sess_id} ({sess_date})",
                        "text": header + full_session_str,
                        "sentences": turns_text[:25],
                        "timestamp": sess_date,
                    }
                )
                if sess_id in answer_session_ids or has_answer_turn:
                    gold_ids.append(pid)
            else:
                # Split into turn groups of 6 turns
                step = 6
                for c_idx in range(0, max(1, len(turns_text)), step):
                    chunk_turns = turns_text[c_idx : c_idx + step]
                    chunk_pid = f"{pid}_c{c_idx // step}"
                    chunk_body = "\n".join(chunk_turns)
                    passages.append(
                        {
                            "passage_id": chunk_pid,
                            "title": f"Session {sess_id} Part {c_idx // step + 1} ({sess_date})",
                            "text": header + chunk_body,
                            "sentences": chunk_turns,
                            "timestamp": sess_date,
                        }
                    )
                    if sess_id in answer_session_ids or has_answer_turn:
                        gold_ids.append(chunk_pid)

        # Fallback evidence matching if answer_session_ids wasn't populated
        if not gold_ids and answer:
            ans_lower = answer.lower().strip()
            for p in passages:
                if ans_lower and ans_lower in p["text"].lower():
                    gold_ids.append(p["passage_id"])

        samples.append(
            QuerySample(
                query_id=qid,
                dataset="longmemeval_s",
                question_type=qtype,
                question=question,
                gold_answers=[answer],
                gold_evidence_ids=gold_ids,
                corpus_passages=passages,
            )
        )
    return samples
