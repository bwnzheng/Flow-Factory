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

### GRPO-Guard bypassed sample metadata and source propagation
- **Date**: 2026-07-27
- **Symptom**: Multi-source GRPO-Guard failed with `GenEval reward requires 'metadata' containing 'include'`, and specialist rewards could run on the wrong dataset.
- **Root Cause**: `GRPOGuardTrainer.sample()` called `adapter.inference()` directly instead of the standard `generate_samples()` / `sample_batch()` pipeline, so generated samples never received dataset metadata, source, or source ID.
- **Fix**: `trainers/grpo.py` now delegates GRPO-Guard sampling to `generate_samples()` while requesting the `next_latents_mean` callback; a regression test verifies the delegation contract.
- **Lesson**: Algorithm-specific sampling parameters belong in `generate_samples()` keyword arguments. Reward-based trainers must not bypass `sample_batch()`, because it owns metadata injection, source-aware reward routing, and sample bookkeeping.
- **Related Constraint**: #6

### Convex-support LP returned an unknown HiGHS status
- **Date**: 2026-07-17
- **Symptom**: Four-dimensional Pareto analysis intermittently failed with HiGHS Status 15 (`model_status is Unknown; primal_status is Infeasible`) even though selecting the target point itself guarantees LP feasibility.
- **Root Cause**: The convex-combination primal used one variable per Pareto point and contained an explicitly feasible but highly degenerate one-hot solution. SciPy 1.15.3's bundled HiGHS could return an indeterminate numerical status for this formulation even after its reward coordinates were centered and scaled.
- **Fix**: `reward_pareto_analysis/plots.py` now solves the equivalent reward-weight dual with one variable per reward dimension, validates successful solutions, and retries indeterminate dual-simplex results with HiGHS interior point. Error logs name the current Pareto metric rather than the removed convexity-depth metric. Regression tests cover the reported 4-D group and solver fallback under SciPy 1.15.3.
- **Lesson**: Algebraic scaling cannot eliminate structural degeneracy. For supported-front classification, solve the direct weighted-sum feasibility problem in objective space, validate solver output, and reserve an algorithmically distinct fallback for genuinely indeterminate statuses.
- **Related Constraint**: N/A

### Crossover offspring mixed incompatible trajectory metadata
- **Date**: 2026-07-28
- **Symptom**: Crossover GRPO-Guard offspring could fail while stacking trajectories, use parent statistics for an intervention transition, or optimize mixed parent/child batches with incompatible shared index maps. Runtime also failed on missing trainer attributes `_max_sde` and `_num_steps`; `_adapter` would have failed next. Multi-generation evolution could reach `BaseSample._stack_values()` with CPU parents and NPU children and fail because `torch.stack` saw two devices. Crossover NFT denoised offspring outside its EMA sampling policy and applied training-only crossover metadata during evaluation.
- **Root Cause**: Child construction treated the crossover latent as a post-step latent, inherited sparse parent mappings and arbitrary parent conditioning, and did not preserve the selected-parent lineage or the base trainer's sampling context. Trainer callbacks copied GA-internal attribute names and assumed that training-only `sample()` state existed when the shared evaluation path called `sample_batch()` directly. Template batching also stacked samples before moving them to a common device, which is invalid once CPU offload and device-resident offspring coexist.
- **Fix**: `trainers/crossover/genetic_algorithm.py` now preserves deterministic parent lineage, propagates the full adapter conditioning batch, applies source-aware reward weights, and selects a fixed-size population with configurable survivor scoring. Before stacking template conditioning, it filters and moves every sample dictionary to the target device without mutating the population. Its cross-rank statistics are explicitly packed as `torch.float32` before reduction for HCCL/NPU compatibility. `trainers/crossover_grpo_guard.py` builds dense identity-mapped trajectories, recomputes old-policy statistics at the actual intervention boundary, derives step counts from `training_args`, and uses the trainer's public `adapter`. Both crossover trainers use an explicit, consumed training-rollout marker so evaluation and disabled crossover bypass training-only post-processing. `trainers/crossover_nft.py` also evolves offspring inside the NFT sampling context. Crossover configurations require rank-local complete groups and reject unsupported groupwise rewards.
- **Lesson**: An intentional off-policy intervention still needs internally consistent transition tuples. Whenever parent and offspring samples share an optimize batch, their shared trajectory maps must have the same semantics, offspring generation must use the same model-policy context and full conditioning contract as parent generation, and distributed statistic tensors must use a backend-supported explicit dtype rather than inheriting Python or NumPy defaults. Trainer callbacks must use attributes guaranteed by the trainer hierarchy—not similarly named state owned by a helper object—and `sample_batch()` overrides need an explicit training/evaluation contract rather than depending on caller side effects or indirect signals. With CPU offload, normalize tensor devices per sample before any collation operation; moving an already-stacked batch is too late.
- **Related Constraint**: #6, #7

### Crossover genetic selection ignored configured advantage aggregation
- **Date**: 2026-07-30
- **Symptom**: Crossover parent and survivor selection always used GDPO-style per-reward normalization even when `train.advantage_aggregation` was configured as `sum`; constant-reward populations also retained a non-zero genetic selection score.
- **Root Cause**: The genetic-algorithm refactor introduced a private GDPO-only advantage implementation instead of preserving the trainer's configured aggregation contract and the formal processor's zero-mean normalization behavior.
- **Fix**: `trainers/crossover/genetic_algorithm.py` now validates and applies `advantage_aggregation` for both parent and merged-population survivor ranking: `sum` aggregates weighted raw rewards before group normalization, while `gdpo` normalizes each reward before weighting. Both crossover trainers log the resolved aggregation, and regression tests cover aggregation-specific results, source-aware weights, invalid values, and zero-variance groups.
- **Lesson**: Any helper that performs intermediate policy-data selection must honor the same user-facing objective configuration as the final optimizer. Distributed helpers may reproduce the communication-free local-group ordering, but must not call collective-bearing processors from rank-asynchronous per-group loops.
- **Related Constraint**: #6

### Pareto filtering mishandled equal leading coordinates
- **Date**: 2026-08-03
- **Symptom**: Pareto-first crossover selection removed one of two identical reward vectors, and could retain a dominated candidate when its dominator had the same first reward coordinate but appeared later in input order.
- **Root Cause**: `compute_pareto_mask` omitted the required strict-improvement check and used a one-direction scan whose first-coordinate sort did not establish an order among equal leading coordinates.
- **Fix**: `trainers/crossover/pareto.py` now applies the direct `O(N²d)` dominance definition over finite candidates; regression tests cover duplicate vectors and domination with equal leading coordinates.
- **Lesson**: Pareto dominance is a strict partial order, and sorting by one objective only permits one-direction pruning when ties in that objective are handled explicitly. A direct definition is safer for the small genetic candidate pools used here.
- **Related Constraint**: N/A

### Nested GA diagnostics contained non-JSON NumPy scalars
- **Date**: 2026-08-04
- **Symptom**: Crossover NFT crashed while writing `metrics.jsonl` with `TypeError: Object of type bool_ is not JSON serializable` after `cov_per_sample` diagnostics were added.
- **Root Cause**: `LogFormatter` normalizes top-level numerical values but intentionally leaves ordinary nested dictionaries unchanged, while `Logger._strip_media()` removed media without converting nested NumPy scalars, arrays, or tensors to JSON-native values.
- **Fix**: `logger/abc.py` now recursively converts NumPy scalars and arrays plus scalar/vector tensors at the JSON boundary; `logger/formatting.py` recognizes top-level `np.bool_`; the covariance producer also casts its degeneracy flag to a Python `bool`. Regression tests cover nested `np.bool_`, integer/float NumPy scalars, arrays, and tensors.
- **Lesson**: Algorithm diagnostics should emit Python-native values when practical, but the persistence boundary must still normalize nested scientific-computing types because ordinary diagnostics dictionaries bypass top-level metric formatting.
- **Related Constraint**: N/A

### Globally gathered advantage metrics were rank-reduced again
- **Date**: 2026-08-06
- **Symptom**: The `group_contiguous` advantage logging path performed float64 rank reductions after all ranks already held the same globally gathered reward and advantage arrays, adding redundant collectives to every feedback step.
- **Root Cause**: The log-data builders treated the output of `_gather_for_logging()` as rank-local shards even though both sampler paths make those arrays globally complete before metric construction.
- **Fix**: `advantage/advantage_processor.py` now computes summary and zero-variance metrics locally from the globally complete arrays; SRC diagnostics are appended to the existing float32 gather payload and introduce no additional rank reduction.
- **Lesson**: Mark the communication state of metric arrays at every boundary. Once a gather has replicated a complete payload on every rank, downstream statistics must be local; reducing the replicated payload again only duplicates communication and may introduce unsupported high-precision collectives.
- **Related Constraint**: N/A

### Sample-wise covariance selection accepted a mismatched GDPO policy direction
- **Date**: 2026-08-06
- **Symptom**: `crossover.survivor_score: cov_per_sample` could be configured with `advantage_aggregation: gdpo`, although its frozen per-sample contribution was computed from a weighted-sum scalar reward direction.
- **Root Cause**: The original covariance-selector decoupling rule was applied equally to group-level covariance and sample-wise contribution selection, even though only the latter claims per-sample alignment with the policy scalarization.
- **Fix**: Both crossover training-argument classes and `GeneticAlgorithm` now fail fast unless `cov_per_sample` is paired with `advantage_aggregation: sum`; documentation and SD3.5 LoRA crossover examples expose the restriction. Group-level `survivor_score: covariance` retains its existing independent-proxy behavior under GDPO.
- **Lesson**: A subset-level selection proxy may be intentionally independent from the policy aggregation, but a per-sample score presented as contribution to the policy direction must use the same scalarization or be given a separate method definition and name.
- **Related Constraint**: #6

### NFT SRC validation confused an interpretation boundary with objective validity
- **Date**: 2026-08-06
- **Symptom**: `sample_weighting: src` rejected `trainer_type: nft` with `off_policy: true`, even though the weighted NFT regression objective is well defined for EMA-generated rollout groups.
- **Root Cause**: The configuration guard promoted SRC's fresh-reference raw-reward alignment condition into a runtime requirement. That condition limits the strongest gradient interpretation after the current policy drifts from the EMA old policy; it does not invalidate the complete NFT objective.
- **Fix**: Allow off-policy NFT SRC, keep the undeclared off-policy AWM combination rejected, and log whether NFT uses the current or EMA sampling policy as its reference mode.
- **Lesson**: Theory claim boundaries must be represented as explicit diagnostics and documentation unless violating them makes the implemented objective undefined or incorrect.
- **Related Constraint**: #26

### GA population reward summaries used group counts as sample counts
- **Date**: 2026-08-06
- **Symptom**: `ga/genN/<reward>/pop_mean` and `new_mean` were inflated by `group_size`, while the corresponding standard deviations were never emitted.
- **Root Cause**: The accumulator stored sample-weighted sums but `reduce_stats()` divided population and survivor totals by the number of groups; its output branch also treated every `*_mean` prefix as a mean and discarded the computed variance.
- **Fix**: `trainers/crossover/genetic_algorithm.py` now reduces the total population count, uses sample-count denominators for population, child, and survivor rewards, and writes both mean and standard deviation metrics. Regression tests cover exact known moments.
- **Lesson**: Distributed moment accumulators must reduce explicit counts in the same unit as their sums. Avoid inferring denominators from fixed group sizes or deriving output metric names through conditions that cannot reach every statistic.
- **Related Constraint**: N/A

### Media sidecars rejected non-finite tensor rewards
- **Date**: 2026-08-07
- **Symptom**: NFT and other reward-based trainers crashed on non-main ranks while saving media sidecar JSON with `ValueError: Out of range float values are not JSON compliant: nan`.
- **Root Cause**: The JSON-safe converter unpacked tensors and NumPy arrays with `.item()` or `.tolist()` but did not recursively normalize resulting `nan` and infinite Python floats before `json.dump(..., allow_nan=False)`.
- **Fix**: `logger/formatting.py` recursively normalizes unpacked scientific containers, and `logger/abc.py` applies the same conversion to the complete media entry at the sidecar persistence boundary. Media regression tests cover non-finite tensor rewards and direct `LogImage` metadata.
- **Lesson**: Converting a scientific container to Python-native values does not make its floating-point contents JSON-compliant. Re-run recursive validation after unpacking and enforce it again at strict persistence boundaries.
- **Related Constraint**: N/A

### SRC scalar contrast overemphasized large advantages or lifted near-zero samples
- **Date**: 2026-08-10
- **Symptom**: Raw SRC scores grew without bound with large scalar advantage magnitude, while replacing the magnitude with a hard sign removed the distinction between small nonzero and high-confidence scalar contrasts.
- **Root Cause**: The raw scalar factor was linear and unbounded, whereas the sign-only factor was discontinuous at zero and discarded all confidence in the scalar contrast magnitude.
- **Fix**: `advantage/sample_weighting.py` now supports configurable raw and saturated SRC scores. Saturated mode divides each frozen-group scalar advantage by its absolute value plus the frozen-group RMS and a numerical epsilon; both score variants are logged and covered by formula, limiting-behavior, configuration, and integration tests.
- **Lesson**: A bounded per-sample contrast factor should remain continuous at zero, use a scale frozen from the same reference group, and approach a sign-only limit only when the contrast dominates that scale.
- **Related Constraint**: N/A

### SRC sample mass changed the optimization baseline
- **Date**: 2026-08-21
- **Symptom**: SRC's probability-weighted reward mean and variance changed each sample's advantage, so a sample could cross from a positive to a negative supervision signal even though SRC was intended to reweight sample importance only.
- **Root Cause**: The implementation used the SRC probability distribution both to define the reward baseline and to multiply the optimization objective, conflating reweighted sample mass with the ordinary reward-relative supervision signal.
- **Fix**: `advantage/sample_weighting.py` now keeps uniform prompt-group means, variances, and advantages for optimization; `AdvantageProcessor` applies `K * probability` exactly once to linear advantages or the complete NFT per-sample loss. Weighted-centered advantages remain available as SRC diagnostics, and guidance documents now state the separation explicitly.
- **Lesson**: A sample-weighting method must distinguish the distribution used to rank or weight samples from the baseline used to define their reward direction. If the latter is intentionally changed, expose it as a separate objective mode rather than silently changing per-sample supervision.
- **Related Constraint**: N/A

### Offline reward worker lacked the accelerator barrier contract
- **Date**: 2026-08-27
- **Symptom**: Standalone `tools.reward_evaluation` crashed while constructing UniReward with `AttributeError: '_AcceleratorView' object has no attribute 'wait_for_everyone'`.
- **Root Cause**: The offline reward worker passed a minimal accelerator facade to reward models, but several model constructors unconditionally call `wait_for_everyone()` after loading their weights.
- **Fix**: `tools/reward_evaluation/scoring.py:_AcceleratorView` now exposes a no-op `wait_for_everyone()` method because spawned offline workers are independent processes rather than members of one Accelerate process group. A regression test covers the compatibility contract.
- **Lesson**: An offline model worker should expose every lifecycle method used by registered reward constructors, while collectives must remain explicit no-ops unless the workers share a real distributed process group.
- **Related Constraint**: N/A

### Offline reward workers allowed automatic cross-device model placement
- **Date**: 2026-08-27
- **Symptom**: Multi-device reward workers could load a UniReward model onto an unintended device or shard it across visible devices despite each worker selecting its assigned accelerator.
- **Root Cause**: UniReward passed `device_map="auto"` to third-party model loaders, so the worker's current device did not constrain Hugging Face placement.
- **Fix**: `src/flow_factory/rewards/unireward.py` now constructs an explicit single-device map for CUDA, CPU, and other accelerator devices and uses it for both UniReward backends; regression tests cover the map contract.
- **Lesson**: Setting the current accelerator is not a placement guarantee when a loader performs automatic device mapping. Parallel one-model-per-worker evaluators must pass an explicit full-model device map.
- **Related Constraint**: N/A

### Standalone reward evaluation appeared silent during long scoring
- **Date**: 2026-08-27
- **Symptom**: `tools.reward_evaluation` could run for a long time with low accelerator utilization, no visible reward progress, and no `reward_scores` or final result files until every image and reward worker completed.
- **Root Cause**: The evaluator only printed a final completion line. Image inference and reward scoring had no flushed stage/worker progress, while reward caches were written only after all worker futures returned; long VLM scoring therefore looked stalled and left no durable intermediate result.
- **Fix**: `tools/reward_evaluation/evaluate.py` now emits flushed run/source/image/reward/checkpoint progress. `scoring.py` reports worker lifecycle and approximately ten progress updates per chunk, flushes output, and writes the resumable reward cache after each completed worker. Model-inference progress prints are flushed as well, and the README documents the incremental cache behavior.
- **Lesson**: Long-running evaluators need observable boundaries before and during expensive model calls, plus durable partial artifacts at safe completion boundaries. Low accelerator utilization alone does not distinguish model loading, CPU/PIL preprocessing, sequential autoregressive scoring, and a deadlock.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
