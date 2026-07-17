# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Group-aware multi-reward convexity analysis
- **Date**: 2026-07-17
- **Symptom**: Multi-reward training analysis produced empty or misleading curves, discarded groups with an all-NaN inapplicable reward, and approximated convexified hypervolume incorrectly in three or more dimensions.
- **Root Cause**: The reader applied an all-dimensions finite mask before prompt grouping, while the high-dimensional hull-volume routine omitted box vertices with multiple coordinates on the reference planes.
- **Fix**: `reward_pareto_analysis/reward_logs.py` partitions complete prompt groups by their exact available-reward tuple and rejects partial missingness; `plots.py` computes exact discrete and convexified dominated volumes plus LP-based convex-supported ratios; `analyze.py` dispatches full-dimensional combination-specific PNG/PDF/CSV outputs instead of reward-pair projections.
- **Lesson**: Reward applicability must be resolved at group granularity before forming objective vectors, and a D-dimensional dominated box has `2**D` vertices. Never silently clip a negative convexification gap because it hides a geometric implementation error.
- **Related Constraint**: N/A

### Pareto convexity ratio plots compressed by theoretical bounds
- **Date**: 2026-07-17
- **Symptom**: Relative convexification HV gap and convex-supported Pareto ratio curves were difficult to read because their y-axes always spanned the full theoretical interval `[0, 1]`, even when the plotted mean, median, and IQR occupied a narrow subrange.
- **Root Cause**: The plotting specification used fixed theoretical bounds as display limits instead of deriving display limits from the finite aggregate statistics actually rendered.
- **Fix**: `reward_pareto_analysis/plots.py` computes adaptive y-limits from the per-step mean, median, and IQR with padding and a minimum span; when a rendered statistic reaches the theoretical upper boundary, the display adds slight headroom without relaxing domain validation. The overview also contains qualitative high/low interpretation guidance for every metric, and each publication figure is emitted as both PDF and text-preserving SVG. Regression tests cover adaptive ratio-axis scaling and both vector artifact sets.
- **Lesson**: A metric's theoretical domain is not necessarily a useful plotting range. Validate mathematical bounds while scaling the visible axis to every rendered statistic, including uncertainty bands rather than only the central curve, and leave display headroom when data touches a hard boundary so valid values do not look clipped.
- **Related Constraint**: N/A

### Pareto analysis aggregation scaled quadratically with training length
- **Date**: 2026-07-17
- **Symptom**: Projecting the reward analysis to 800 training steps showed avoidable aggregation latency after the geometric metrics had finished computing.
- **Root Cause**: Every metric and rendered panel repeatedly scanned all group rows once per step, making summary aggregation quadratic in the number of steps; geometric step computations were also capped at four processes on a larger CPU host.
- **Fix**: `reward_pareto_analysis/plots.py` groups rows by step once and reuses cached summaries across the overview and vector figures; step-level computations now accept a configurable worker limit, with automatic sizing capped at 16. `analyze.py` and `default.yaml` expose the setting as `compute.max_workers`.
- **Lesson**: For long-run analysis, derive all per-step metric summaries from one grouping pass and cache them for every output consumer. Independent CPU-heavy step computations should expose bounded concurrency instead of embedding a small fixed worker cap.
- **Related Constraint**: N/A

### Offline reward analysis inherited image-workflow requirements
- **Date**: 2026-07-17
- **Symptom**: The default Pareto analysis configuration silently enabled three data sources, omitted the `rewards_analysis` section, and required reward-model configuration even when only reading saved reward pickles.
- **Root Cause**: Image rescoring, checkpoint evaluation, and saved-reward analysis shared unconditional defaults and validation despite having different runtime dependencies.
- **Fix**: `reward_pareto_analysis/analyze.py` defaults image-based sources to disabled, requires `rewards` only when an image source is enabled, and avoids prompt and model-loading work in rewards-only mode; `default.yaml` explicitly enables only saved-reward analysis, and the README documents a minimal configuration.
- **Lesson**: Multi-source offline tools should validate dependencies per enabled source. Lightweight, read-only analysis should be the explicit safe default, while GPU/model workflows require opt-in configuration.
- **Related Constraint**: N/A

### Convex-support LP returned an unknown HiGHS status
- **Date**: 2026-07-17
- **Symptom**: Four-dimensional Pareto analysis intermittently failed with HiGHS Status 15 (`model_status is Unknown; primal_status is Infeasible`) even though selecting the target point itself guarantees LP feasibility.
- **Root Cause**: The convex-combination primal used one variable per Pareto point and contained an explicitly feasible but highly degenerate one-hot solution. SciPy 1.15.3's bundled HiGHS could return an indeterminate numerical status for this formulation even after its reward coordinates were centered and scaled.
- **Fix**: `reward_pareto_analysis/plots.py` now solves the equivalent reward-weight dual with one variable per reward dimension, validates successful solutions, and retries indeterminate dual-simplex results with HiGHS interior point. Error logs name the current Pareto metric rather than the removed convexity-depth metric. Regression tests cover the reported 4-D group and solver fallback under SciPy 1.15.3.
- **Lesson**: Algebraic scaling cannot eliminate structural degeneracy. For supported-front classification, solve the direct weighted-sum feasibility problem in objective space, validate solver output, and reserve an algorithmically distinct fallback for genuinely indeterminate statuses.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
