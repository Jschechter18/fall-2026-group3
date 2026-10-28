# Audit of distributed collection commit 147ccca

Verdict: the basic activation concatenation/index reconstruction is sound for
the tested fixtures, but this commit is not ready for a long distributed run.
This audit leaves runtime behavior unchanged and annotates the assumptions
about synthetic inputs, which are not available yet.

## Findings, in priority order

1. **P1: candidates workers cannot collect.**
   `scripts/collect_activations.py:154` supplies no candidate orders, and its
   distributed call at line 186 never passes them. `collect_examples` requires
   them for the candidates presentation. Existing v3 full candidates configs
   therefore fail before question generation. Freeze candidate order in the
   manifest before chunking, persist it in packages, and pass it to the existing
   collector. Reuse the existing candidate-order logic, extending its hop input
   handling if necessary; assigning orders independently per chunk would change
   the experiment when chunk sizes change.

2. **P1: experiment compatibility is optional and actual provenance is lost.**
   At `scripts/collect_activations.py:156`, an empty compatibility dictionary
   bypasses validation. A nonempty dictionary checks only caller-selected
   top-level fields. The actual resolved configuration and progress identity
   are constructed but not persisted in the portable chunk. Finalization accepts
   an independently supplied resolved configuration without comparing it to the
   workers. Workers using different model revisions, generation settings, or code
   can consequently produce chunks that appear compatible. Reuse the existing
   provenance builders to persist and verify a required experiment identity,
   while retaining hardware separately as worker metadata.

3. **P1: stored hashes are not verified through the whole pipeline.**
   `plan_chunks` at line 178 does not check that the frozen manifest belongs to
   the supplied master dataset or that its payload hashes match. Workers check
   question IDs but never recompute payload/membership hashes. Finalization at
   line 366 calculates current file hashes without comparing them to an expected
   accepted manifest, and does not verify recorded seeds against the plan.
   Changed questions or modified local records can therefore be accepted with
   the original identity labels. Verify each transition using existing hash
   helpers and carry the accepted file manifest into local finalization.
   A checksum of the downloaded bytes alone cannot prove they represent the
   intended question payload or collection configuration.

4. **P1: historical partitions and downstream consumers are not integrated.**
   The freeze CLI expects JSON/YAML with a `questions` list; existing committed
   partitions are CSV and use unprefixed IDs. There is no migration path for those
   assignments. Existing `production.read_partitions` also derives hop groups
   from IDs and rejects the new namespaced IDs. `interaction_paths` reads train
   and validation, whereas the new finalizer writes test records separately.
   Finalization does not produce the existing behavior package/index manifest.
   Generic tensor loading is compatible, but claiming full SAE/probe/causal
   integration is premature. Add a validated adapter preserving historical
   assignments/exposure and exercise the actual downstream entry points.

5. **P1: distributed resume is absent.**
   The worker rejects `--resume` and collects the complete work package before
   writing a checkpoint. Interrupting a package loses all of its question work.
   It also processes one package per invocation, so each package reloads models.
   Reuse the existing checkpoint loop inside the distributed worker and allow
   multiple assigned packages with one model load. Explicitly distinguish a
   resumable local attempt from retrying a whole chunk on another instance.

6. **P2: split validation duplicates and weakens existing code.**
   `_weighted_assignment` checks names and total but accepts negative values
   and uses different rounding from the existing allocation helper. Frozen old
   partition values are not validated either; an unknown value can disappear
   from a work plan. Reuse existing proportion validation and `allocate_counts`
   where their contract fits, and validate every frozen assignment. This is a
   direct recycle-code opportunity with correctness benefits.

7. **P2: final publication can leave a partial final run.**
   Finalization renames activations before results, then writes the final
   summary. A failure between these operations leaves a final directory that
   blocks retries, while cleanup removes only staging directories. Add a final
   completion marker that consumers require and a defined recovery path; two
   directory renames do not constitute a transaction. Work-package uploads can
   also overwrite the same input key, and explicit attempt IDs can be reused.
   Enforce immutability/conflict detection at those write boundaries.

8. **P2: final metadata loses information needed for analysis.**
   Finalization replaces actual hop groups with the literal `explicit` and does
   not attach the frozen source/partition metadata to final question manifests.
   `collection_report` is not wired into collection or finalization. Summary
   `run_name` is also left as the staging name. Carry real frozen metadata into
   final outputs and generate source/partition counts without redefining the
   existing disagreement eligibility policy.

## Validation and limitations

- The relevant regression invocation passed 186 test executions (the progress
  test module was inadvertently listed twice). `git diff --check` passed.
- Existing distributed tests compare serial and multichunk activation values
  and all four indexes. This supports the core merge algorithm.
- Five additional temporary audit tests reproduced: missing candidate orders,
  planning changed payloads with stale partitions, accepting altered seed
  records in finalization, rejection of namespaced partition rows by the old
  reader, and acceptance of negative proportions. These tests assert the
  observed defects; passing them confirms a defect, not correct behavior.
- Existing S3 tests use an in-memory fake; no live S3, EC2, paid model jobs,
  or real synthetic release were exercised.
- Per-site finalization still loads all sites in each chunk container. Tensor
  concatenation and round-trip verification create additional copies; capacity
  planning needs actual sizes before a large merge.
- Recommended repair order: candidate/identity validation; historical and
  downstream adapters; resume; publication and metadata; then a small two-EC2
  S3 smoke run before scaling.

<!-- REVIEWING NOW -->

## Simplification after review

The experimental identity-generation, source fingerprinting, and strict
code/environment compatibility gates were removed at the user's request.
Use the agreed collection YAML and run ID operationally, and review scientific
changes before mixing results. Provenance is recorded rather than enforced.
Lightweight master/partition linkage, question-seed checks, existing merge
alignment checks, and retained S3 download checksums remain. This deliberately
does not implement the audit's proposed comprehensive compatibility enforcement.

## Candidate-order wiring

Fixed: freeze now assigns candidate order using explicit partition/hop groups,
preserves existing orders on extension, and carries them through chunk membership
and work packages. Workers pass them to the existing collector and record them
in manifest provenance. Missing orders fail before model loading; finalization
checks candidates records against planned orders. Tests exercise mixed-source
fixtures, chunk-size independence, extensions, and the actual worker entry point
with model execution mocked. No synthetic release or paid model run was used.

## Partition and downstream integration

Implemented: historical CSV assignments are mapped by dataset source/original
ID, with optional historical exposure metadata and no reassignment of known
questions. Finalization carries explicit origin/hop/exposure metadata and writes
permanent membership, collected-row manifests, source hashes, and a coverage
report separating exclusions and reserved intervention questions. Explicit
partitioned-layout readers include test files and preserve legacy defaults.
Existing behavior-package and activation-export CLIs accept the new layout;
SAE and probe loaders are exercised against the actual merged tensors.

Claude label generation remains a separate ticket. Distributed full runs do not
use the historical layer-scan merge path. Old work plans without question
metadata retain their old outputs and need regeneration for the new tables.
