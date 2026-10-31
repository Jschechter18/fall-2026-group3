"""Heuristic templates and bounded approximate question-neighbor retrieval."""
from collections import Counter, defaultdict
import hashlib
import random
import re

from ..deduplication import normalize_question


def _template(record):
    text = normalize_question(record["question"])
    entities = set(record["support_titles"])
    entities.update(step["answer"] for step in record["steps"])
    for entity in sorted(entities, key=len, reverse=True):
        if not entity.strip():
            continue
        normalized = normalize_question(entity)
        if len(normalized) >= 3 and not normalized.isnumeric():
            text = re.sub(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", "<entity>", text)
    return re.sub(r"\b\d+\b", "<number>", text)


def _shingles(question):
    words = re.findall(r"\w+", normalize_question(question))
    features = set(words) | {" ".join(pair) for pair in zip(words, words[1:])}
    return {int.from_bytes(hashlib.blake2s(feature.encode(), digest_size=4).digest(), "big") for feature in features}


def _neighbor_pairs(records, *, min_similarity=.5, neighbors=5, bucket_limit=128, retrieval_stats=None):
    """Bounded MinHash/LSH retrieval; approximate and explicitly not exhaustive."""
    prime = 4294967291
    rng = random.Random(42)
    coefficients = [(rng.randrange(1, prime), rng.randrange(prime)) for _ in range(64)]
    features = [_shingles(record["question"]) for record in records]
    buckets = defaultdict(list)
    signatures = []
    truncated = 0
    for index, values in enumerate(features):
        signature = tuple(min((a * value + b) % prime for value in values) if values else 0
                          for a, b in coefficients)
        signatures.append(signature)
        for band in range(16):
            key = (band, signature[band * 4:band * 4 + 4])
            if len(buckets[key]) < bucket_limit:
                buckets[key].append(index)
            else:
                truncated += 1
    if retrieval_stats is not None:
        retrieval_stats["lsh_bucket_entries_omitted"] = truncated
    for index, signature in enumerate(signatures):
        collisions = Counter()
        for band in range(16):
            collisions.update(j for j in buckets[(band, signature[band*4:band*4+4])] if j != index)
        candidates = sorted(collisions, key=lambda j: (-collisions[j], j))[:64]
        ranked = []
        for other in candidates:
            union = features[index] | features[other]
            score = len(features[index] & features[other]) / len(union) if union else 0
            if score >= min_similarity:
                ranked.append((score, other))
        for score, other in sorted(ranked, reverse=True)[:neighbors]:
            yield min(index, other), max(index, other), score, truncated
