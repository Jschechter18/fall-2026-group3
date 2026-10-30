"""Compatibility imports for the dataset-wide redundancy audit."""
from .audit import AUDIT_VERSION, audit_dataset
from .audit.catalog import _load_catalog
from .audit.planning import _make_batches
from .audit.review import OBSERVATIONS, RELATIONSHIPS, SYSTEM, _parse_review, _prompt, _review_batch
from .audit.similarity import _neighbor_pairs, _shingles, _template

__all__ = ["AUDIT_VERSION", "audit_dataset"]
