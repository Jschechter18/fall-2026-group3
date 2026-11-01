"""Load compact question catalogs and compute dataset-wide local statistics."""
from collections import Counter, defaultdict
import json

from mas_sae.evaluation.scoring import normalize_answer
from mas_sae.experiments.artifacts import sha256_json

from ..deduplication import normalize_question
from .similarity import _template


def _load_catalog(path):
    records, seen = [], set()
    exact, templates, titles = defaultdict(list), defaultdict(list), defaultdict(list)
    answers, hops, openings = Counter(), Counter(), Counter()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if (not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"].strip()
                    or not isinstance(value.get("question"), str) or not value["question"].strip()
                    or not isinstance(value.get("answer"), str) or not value["answer"].strip()):
                raise ValueError(f"{path}:{line_number}: expected id, question, and answer strings")
            if value["id"] in seen:
                raise ValueError(f"Duplicate record ID: {value['id']}")
            seen.add(value["id"])
            steps = value.get("question_decomposition")
            paragraphs = value.get("paragraphs")
            if (not isinstance(steps, list) or not isinstance(paragraphs, list)
                    or any(not isinstance(s, dict) or not isinstance(s.get("question"), str)
                           or not isinstance(s.get("answer"), str) for s in steps)
                    or any(not isinstance(p, dict) or not isinstance(p.get("title"), str) for p in paragraphs)):
                raise ValueError(f"{path}:{line_number}: invalid decomposition or paragraph metadata")
            record = {"record_id": value["id"], "question": value["question"], "answer": value["answer"],
                      "hop_type": value["id"].split("__", 1)[0], "candidate_sha256": sha256_json(value),
                      "steps": [{"question": s["question"], "answer": s["answer"]} for s in steps],
                      "support_titles": [p["title"] for p in paragraphs if p.get("is_supporting") is True]}
            records.append(record)
            exact[normalize_question(record["question"])].append(record["record_id"])
            templates[_template(record)].append(record["record_id"])
            for title in set(record["support_titles"]):
                if title.strip():
                    titles[normalize_question(title)].append(record["record_id"])
            answers[normalize_answer(record["answer"])] += 1
            hops[record["hop_type"]] += 1
            openings[" ".join(normalize_question(record["question"]).split()[:3])] += 1
    stats = {"records": len(records), "by_hop_type": dict(hops),
             "unique_normalized_answers": len(answers), "top_answers": answers.most_common(20),
             "top_question_openings": openings.most_common(20),
             "repeated_support_titles": [{"title": title, "count": len(ids), "record_ids": ids}
                                         for title, ids in sorted(titles.items(), key=lambda p: (-len(p[1]), p[0]))
                                         if len(ids) > 1][:50]}
    return records, stats, exact, templates
