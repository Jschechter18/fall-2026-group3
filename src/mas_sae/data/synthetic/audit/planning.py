"""Plan bounded catalog and neighbor comparisons within prompt budgets."""
from collections import defaultdict
import json

from ..deduplication import normalize_question
from .review import _prompt
from .similarity import _neighbor_pairs, _template


def _make_batches(records, batch_size, max_prompt_chars, max_neighbor_batches):
    """Catalog covers all fitting records; extra packs bring cross-batch neighbors together."""
    ordered = sorted(range(len(records)), key=lambda i: (_template(records[i]), normalize_question(records[i]["question"])))
    sizes = [len(json.dumps(record, ensure_ascii=False)) for record in records]
    overhead = len(_prompt([], "neighbors_99999999")) + 10
    packets, current, chars = [], [], overhead
    for index in ordered:
        if current and (len(current) >= batch_size or chars + sizes[index] + 2 > max_prompt_chars):
            packets.append(current); current, chars = [], overhead
        current.append(index); chars += sizes[index] + 2
    if current:
        packets.append(current)
    shortest = sorted(range(len(records)), key=lambda i: sizes[i])
    skipped = set()
    catalog = []
    for packet in packets:
        if len(packet) == 1 and len(records) > 1:
            partner = next((i for i in shortest if i != packet[0]
                            and overhead + sizes[i] + sizes[packet[0]] + 4 <= max_prompt_chars), None)
            if partner is not None:
                packet = packet + [partner]
        if len(packet) < 2 or len(_prompt([records[i] for i in packet], "catalog_99999999")) > max_prompt_chars:
            skipped.update(packet)
        else:
            catalog.append(packet)
    catalog_membership = defaultdict(set)
    jobs = []
    for number, packet in enumerate(catalog, 1):
        batch_id = f"catalog_{number:08d}"
        for index in packet: catalog_membership[index].add(batch_id)
        jobs.append({"batch_id": batch_id, "kind": "catalog", "indices": packet})
    cross_pairs, seen = [], set()
    retrieval_stats = {}
    for left, right, score, truncated in _neighbor_pairs(records, retrieval_stats=retrieval_stats):
        if (left, right) in seen or catalog_membership.get(left, set()) & catalog_membership.get(right, set()):
            continue
        seen.add((left, right)); cross_pairs.append((left, right, score))
    cross_pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    neighbor_packets, current, chars = [], [], overhead
    for left, right, score in cross_pairs:
        new = [i for i in (left, right) if i not in current]
        if current and (len(current) + len(new) > batch_size
                        or chars + sum(sizes[i] + 2 for i in new) > max_prompt_chars):
            neighbor_packets.append(current); current, chars = [], overhead
            new = [left, right]
        if overhead + sum(sizes[i] + 2 for i in (left, right)) > max_prompt_chars:
            continue
        current.extend(new); chars += sum(sizes[i] + 2 for i in new)
    if current: neighbor_packets.append(current)
    for number, packet in enumerate(neighbor_packets[:max_neighbor_batches], 1):
        jobs.append({"batch_id": f"neighbors_{number:08d}", "kind": "neighbors", "indices": packet})
    coverage = {
        "local_records_scanned": len(records), "catalog_batches_planned": len(catalog),
        "catalog_records_planned": len(catalog_membership),
        "records_without_model_comparison": [records[i]["record_id"] for i in sorted(skipped - set(catalog_membership))],
        "cross_batch_candidate_pairs": len(cross_pairs), "neighbor_batches_available": len(neighbor_packets),
        "neighbor_batches_planned": min(max_neighbor_batches, len(neighbor_packets)),
        "neighbor_batches_omitted_by_budget": max(0, len(neighbor_packets) - max_neighbor_batches),
        "lsh_bucket_entries_omitted": retrieval_stats.get("lsh_bucket_entries_omitted", 0),
        "retrieval": "64 MinHash values / 16 LSH bands; Jaccard >= 0.5; top 5 retrieved neighbors/record",
        "retrieval_is_exhaustive": False,
    }
    return jobs, coverage
