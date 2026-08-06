# Algorithm Guidance

## Table of Contents

- [Overview](#overview)
- [GRPO](#grpo)
   - [Background](#background)
   - [Dynamics Type](#dynamics-type)
   - [Efficiency Strategies](#efficiency-strategies)
     - [Mixing SDE and ODE](#mixing-sde-and-ode)
     - [Decoupled Training and Inference Resolution](#decoupled-training-and-inference-resolution)
   - [Regularization](#regularization)
     - [KL-loss](#kl-loss)
     - [GRPO-Guard](#grpo-guard)

- [DPPO](#dppo)

- [DPO](#dpo)

- [DGPO](#dgpo)

- [DiffusionNFT](#diffusionnft)

- [AWM: Advantage Weighted Matching](#awm-advantage-weighted-matching)

- [CRD: Centered Reward Distillation](#crd-centered-reward-distillation)

- [DiffusionOPD: On-Policy Distillation](#diffusionopd-on-policy-distillation)

- [References](#references)

## Overview

Flow-Factory provides unified implementations of state-of-the-art RL algorithms for flow-matching models. All algorithms share the same model adapter and reward interfaces, enabling direct comparison under controlled conditions.

At a high level, the supported algorithms fall into two paradigms:

- **Coupled paradigm (GRPO and variants)**: Training timesteps are coupled with the SDE-based sampling dynamics, requiring tractable log-probability computation for policy gradient optimization.
- **Decoupled paradigm (DPO, DiffusionNFT, AWM, DGPO, CRD, DiffusionOPD)**: Training timesteps are decoupled from the actual sampling dynamics, making them inherently solver-agnostic — any ODE solver can be used for trajectory generation without modifying the training procedure.

## GRPO

### Background

GRPO has achieved significant success in Flow Matching models. In contrast to the standard deterministic ODE-style update rule:

$$
x_{t+\mathrm{d}t} = x_{t} + v_{\theta}(x_t, t) \mathrm{d}t
$$

References [[1]](#ref1) and [[2]](#ref2) incorporate noise to facilitate RL exploration, proposing the following SDE-based update rule:

$$
x_{t+\mathrm{d}t} = x_{t} + [v_{\theta}(x_t, t) + \frac{\sigma_{t}^{2}}{2t}(x_t + (1-t)v_{\theta}(x_t, t))]\mathrm{d}t + \sigma_{t} \sqrt{\mathrm{d}t} \epsilon
$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma_t$ denotes the noise schedule. This SDE formulation enables the log-probability computation required for policy gradient optimization.

The formulation of $\sigma_t$ differs between methods: it is defined as $\eta\sqrt{\frac{t}{1-t}}$ in Flow-GRPO [[1]](#ref1) and as $\eta$ in DanceGRPO [[2]](#ref2), where $\eta \in [0,1]$ is a hyperparameter controlling the noise level. See the [Dynamics Type](#dynamics-type) section for a complete summary.

This algorithm is implemented as `grpo`. To use this algorithm, set config with:

```yaml
train:
    trainer_type: grpo
```

### Dynamics Type

Flow-Factory implements multiple SDE dynamics through a unified `SDESchedulerMixin` interface. Users can switch between formulations via a single configuration parameter, facilitating systematic comparison of their effects on training stability and sample quality.

| Dynamics   | Noise Schedule $\sigma_t$              | Reference                    |
|------------|----------------------------------------|------------------------------|
| `Flow-SDE` | $\eta\sqrt{t/(1-t)}$                 | Flow-GRPO [[1]](#ref1)       |
| `Dance-SDE`| $\eta$ (constant)                     | DanceGRPO [[2]](#ref2)       |
| `CPS`      | $\sigma_{t-1}\sin(\eta\pi/2)$        | FlowCPS [[9]](#ref9)         |
| `ODE`      | $0$ (deterministic)                   | For NFT [[7]](#ref7) / DGPO [[8]](#ref8) / AWM [[10]](#ref10) |

To switch between these formulations, set:

```yaml
scheduler:
    dynamics_type: 'Flow-SDE' # Options are ['Flow-SDE', 'Dance-SDE', 'CPS', 'ODE'].
```

> **Note**: `ODE` dynamics produce deterministic trajectories and cannot provide log-probability estimates. Therefore, `ODE` can only be used with decoupled algorithms such as `NFT`, `AWM`, and `DGPO`. See the [DiffusionNFT](#diffusionnft), [AWM](#awm-advantage-weighted-matching), and [DGPO](#dgpo) sections.


### Efficiency Strategies


#### Mixing SDE and ODE

Training with the original Flow-GRPO and DanceGRPO methods is computationally expensive, as they require computing log probabilities and optimizing across all denoising steps.

Subsequent works, such as MixGRPO [[3]](#ref3) and TempFlow-GRPO [[4]](#ref4), investigated the effects of mixing ODE and SDE denoising rules. They found that applying SDE updates for only $1\sim 2$ steps—and optimizing only those corresponding steps—is sufficient. This approach significantly reduces the cost of the optimization stage and results in faster performance improvements.

To control this behavior, you can configure `sde_steps` and `num_sde_steps` as follows:

```yaml
scheduler:
    # Candidate steps for SDE noise (early steps typically provide more sample diversity)
    sde_steps: [1, 2, 3] 
    
    # Randomly select `1` step from the specified `sde_steps` list (e.g., step 2) 
    # to use SDE denoising. All other steps will use the standard ODE solver.
    num_sde_steps: 1
```

#### Decoupled Training and Inference Resolution

Flow-GRPO demonstrates that *lower-quality images, generated via fewer denoising steps, are often sufficient for reward computation and GRPO optimization*. PaCo-RL[[6]](#ref6) validates this insight from the perspective of **resolution**.

Research indicates that training on moderately low-resolution images yields sufficient reward signals to guide optimization effectively. Furthermore, *performance gains achieved at lower resolutions successfully transfer to high-resolution outputs*. Given that the computational complexity of modern Diffusion Transformers grows quadratically with image resolution, this decoupling significantly reduces training costs.

You can configure a smaller resolution for the sampling and optimization loop while maintaining the target resolution for inference and evaluation:

```yaml
train:
    resolution: 256  # Reduced resolution (int or [height, width]) for faster RL loops
eval:
    resolution: 1024 # Full resolution for validation and inference
```

### SRC-Reweight sample weighting

`sample_weighting: src` enables prompt-local Sample-wise Reward Concordance
weighting without adding a new trainer or changing rollout support. For each
original on-policy group, the shared `AdvantageProcessor` freezes the unweighted
reward means, computes
`c_ik = w_k (r_ik - mean_k) (R_i - mean_R)`, and uses the weakest active
coordinate `s_i = min_k c_ik`. The dimensionless score
`s_i / (Var(R) + epsilon)` is mapped to a bounded probability

`p_i = (1 - lambda) / K + lambda * softmax(score_i / temperature)`.

The same probability defines the weighted scalar-reward mean and variance. The
processor stores `K * p_i` as `sample_weight`. For GRPO-style linear consumers,
it folds that multiplier into the final advantage. For NFT, it keeps the outer
multiplier separate and applies it to the complete positive/negative
per-sample NFT objective while using a weighted-centered signed optimality
signal for the branch mixture. Independent KL regularizers remain uniformly
aggregated.

```yaml
train:
  trainer_type: grpo  # Also supported: grpo-guard, dppo, nft, awm (on-policy only)
  advantage_aggregation: sum
  stddev_reweighting: false
  sample_weighting: src  # Options: none, src
  src_reweight_interpolation: 0.5  # lambda in [0, 1)
  src_reweight_temperature: 1.0  # > 0
  src_reweight_epsilon: 1.0e-8
  src_reweight_degeneracy_threshold: 1.0e-12
```

SRC requires at least two active rewards with positive, nonnegative fixed
weights for every training source. GRPO-style consumers use prompt-local
weighted centering and scaling regardless of `global_std`. NFT preserves its
declared normalizer contract: `global_std: true` keeps the baseline unweighted
global rollout standard deviation while changing the prompt mean to the SRC-
weighted mean; `global_std: false` uses the corresponding weighted prompt
standard deviation. SRC cannot be combined with `stddev_reweighting`,
off-policy AWM, crossover, DPO, DGPO, or CRD. NFT may use either its current
policy or EMA sampling policy as the rollout/reference distribution.

Logging includes probability and `sample_weight` distributions, ESS and
ESS/K, scalar and weighted variance, uniform-versus-reweighted SRC lower
bounds, degeneracy rate, frozen and weighted-recentered per-reward
contribution/conflict metrics, weighted centering/probability-sum errors, and
`train/src_groups` with per-sample scores, probabilities, multipliers,
advantages, and both contribution vectors. NFT additionally logs its
optimality clipping ratio and positive-branch probability. SRC adds no rank
reduction: its diagnostics ride the existing float32 logging gather and are
summarized from the already-complete payload.

### Regularization

#### KL-Loss

To tame the policy model's behavior and maintain proximity to the original reference model, two types of KL loss are available:

```yaml
train:
    kl_type: 'v-based' # Options: 'x-based', 'v-based'
    kl_beta: 0.04 # KL divergence beta
    ref_param_device: 'same_as_model' # Options: cpu, same_as_model
```

Here, `x-based` calculates the KL loss in the **latent space**,
while v-based calculates it in the **predicted velocity space** (or **noise space**).
The `kl_beta` parameter controls the coefficient of the KL divergence term.

**Memory Considerations**: Since calculating KL loss requires maintaining a copy of the original model, *VRAM usage scales with the number of trainable parameters*. 
- **LoRA Training**: The overhead is minimal and efficient.
- **Full-Parameter Fine-Tuning**: The overhead is significant. You may want to set `ref_param_device` to `cpu` to save memory.
- **No KL-Loss**: Setting `kl_beta` to `0` automatically disables this term and eliminates extra memory usage.

**Pack-composition-dependent adapters (e.g. Bagel)**: GRPO's on-policy step assumes `forward()` is bit-identical between rollout and training. Adapters that batch via sequence packing (NaViT) are not batch-invariant, so the default optimize-time sample shuffle breaks the on-policy `ratio == 1`. Set `train.shuffle_samples: false` (with matched sampling/training `per_device_batch_size`) for these adapters. See `workflow.md` Stage 6 and the train-inference consistency topic doc.


#### GRPO-Guard

The SDE formulation used in Flow-GRPO[[1]](#ref1) and DanceGRPO[[2]](#ref2) inherently results in a *negatively biased ratio distribution* during GRPO optimization. GRPO-Guard [[5]](#ref5) analyzes this phenomenon and proposes a normalization technique to mitigate reward hacking.

This normalization aligns with the time-step-dependent (and noise-level-dependent) loss re-weighting strategy introduced in TempFlow-GRPO[[4]](#ref4). By rebalancing the gradient contributions across different time steps, this strategy stabilizes training and effectively reduces reward hacking.

To enable this reweighting strategy, switch the `trainer_type` to `grpo-guard`:
```yaml
train:
    trainer_type: 'grpo-guard'
scheduler:
    dynamics_type: 'Flow-SDE'
```
> ‼️ **Note**: Currently, `grpo-guard` reweighting is only compatible with `Flow-GRPO` dynamics. Therefore, dynamics_type must be explicitly set to `Flow-SDE`.

#### Crossover GRPO-Guard

`crossover-grpo-guard` intentionally augments GRPO-Guard with off-policy latent
interventions. Within each prompt group, a genetic algorithm selects parents,
crosses or mutates an intermediate latent, denoises the offspring, and keeps a
fixed population of `group_size` survivors. The primary parent supplies the
pre-crossover trajectory prefix; its earlier transitions are deliberately
reward-relabelled with the offspring reward. The crossover boundary stores
old-policy statistics recomputed for the actual intervention latent, so PPO
ratios remain transition-consistent even though the data distribution is
intentionally off-policy.

This trainer requires `Flow-SDE`, `group_contiguous` sampling, and pointwise
reward models. `sampler_type: auto` resolves to `group_contiguous`; incompatible
explicit samplers fail at configuration time.

```yaml
train:
  trainer_type: crossover-grpo-guard
  advantage_aggregation: sum  # Required when survivor_score is cov_per_sample
  global_std: true
  stddev_reweighting: false
  crossover:
    enabled: true
    step_sampling: uniform
    step_range: [0.3, 0.6]
    strategy: block
    augmentation_factor: 1.25
    parent_ratio: 0.25
    mutation_std: 0.05
    survivor_score: cov_per_sample  # Options: advantage, abs_advantage, covariance, cov_per_sample
scheduler:
  dynamics_type: Flow-SDE
```

`survivor_score: advantage` prefers high-fitness candidates when the Pareto set
must be trimmed or filled. `abs_advantage` also preserves strongly negative
samples for contrastive training. `covariance` keeps Pareto non-dominance as a
hard quality gate, then greedily constructs the fixed-size survivor group by
maximizing its weakest active reward contribution. Oversized Pareto fronts are
pruned backward; undersized fronts retain every non-dominated candidate and are
filled forward from the dominated pool. This is a group-level subset score, so
the covariance is recomputed for every hypothetical add or removal.

`cov_per_sample` is the scalar-elitist, sample-wise approximation. It merges
parents and offspring, freezes the merged pool's reward and scalar-reward
means, and computes each candidate's weakest active reward contribution
`min_k w_k (r_ik - mean_k) (R_i - mean_R)` exactly once. The highest scalar
reward candidate is always retained; the remaining `group_size - 1` slots are
filled by descending contribution fitness, then scalar reward, then stable
candidate ID. This path does not apply Pareto filtering or recompute covariance
for hypothetical subsets. It logs both the frozen-reference lower-bound
diagnostics and the true recentered covariance of the selected group.

Its per-group console line reports the scalar elite origin, `frozen_J`, the
separable `lower_bound`, their approximation `gap`, the recentered `true_J`,
scalar variance, and `degenerate_scalar_contrast`. Cross-rank logging packs the
fixed-size metric accumulators into a float32 tensor for reduction and exposes
the following platform metrics per
generation under `ga/genN/cov_per_sample/`: `frozen_score`, `lower_bound`,
`approximation_gap`, `true_score`, `scalar_variance`, `elite_child_rate`, and
`degenerate_scalar_contrast_rate`. The metrics JSONL contains only these aggregate
statistics. Raw selection events from every rank are gathered and stored under
`logs/ga/train_step_NNNNNN.pkl` with `schema_version`, `step`, and a `selections`
list. All floating values in this cross-rank object payload are normalized to
float32 before gathering. Each `(rank, gid, generation)` event records the prompt
and source, ordered reward keys and weights, parent-selection evidence, child
parent IDs, every candidate's rewards and selection score, Pareto membership,
and ordered selected/rejected IDs. Following the selected order from one
generation into the next reconstructs the complete survivor-selection and
child-lineage process.

When `crossover.log_rewards` is enabled, the final post-GA population is also
summarized by original-parent versus crossover-child origin under
`ga/final_population/{parent,child}/<reward>/{mean,std}`. The child branch also
reports `better_than_parent_mean_rate`. These are final-survivor composition
statistics and are distinct from the pre-selection `child_*` and post-selection
`new_*` moments under `ga/genN/<reward>/`. No `crossover/*` metric namespace is
emitted.

Training reward pickles under `logs/rewards/train_step_NNNNNN.pkl` remain the
canonical record of the final population consumed by optimization. They store
prompt-grouped float32 reward arrays and, when available, raw SRC group
diagnostics and per-group non-dominated-set sizes. Those raw fields are removed
from `metrics.jsonl`; only their aggregate metrics remain there and in the
platform backend.

Both covariance strategies operate directly on the raw outputs of the configured
reward models and use their existing source-aware training weights. Those
weights therefore define both the preference vector and the calibration across
reward units. A fixed positive affine rescaling of one reward does not change
the method when its weight is transformed inversely; selector-only bounds would
cancel algebraically and are not required. Do not z-score rewards separately
per prompt before selection, because that changes the intended reward geometry.
For `cov_per_sample`, reward units directly affect candidate ranking, so use
fixed run-wide reward calibration or source-aware weights that compensate for
known scale differences; never estimate normalization statistics separately
for each prompt pool.

The selectors require at least two rewards with positive weights. Group-level
`survivor_score: covariance` continues to use a fixed weighted sum of raw
rewards independently of the policy-side `advantage_aggregation`,
`stddev_reweighting`, and `global_std` settings. In contrast,
`survivor_score: cov_per_sample` requires `advantage_aggregation: sum`, because
its frozen per-sample contribution is defined against that scalar policy
direction; pairing it with GDPO would make selection and policy advantages
optimize different directions. A zero-variance comparison falls back
deterministically to absolute group-normalized weighted-sum reward rather than
the configured policy advantage. The startup log reports both
`advantage_aggregation` and `survivor_selection_aggregation` so this relation is
explicit.

With `advantage_aggregation: sum` and fixed policy weights, either covariance
score directly matches the policy scalarization. For group-level `covariance`
with `gdpo` or dynamic policy reweighting, it remains an independent survivor-
selection proxy and must not be interpreted as the exact weakest contribution
to the final policy gradient.
The genetic algorithm always considers the merged parent-plus-offspring
candidate population and returns exactly `group_size` survivors. Advantage-
based parent/survivor ranking continues to follow the configured
`advantage_aggregation` (`sum` or `gdpo`).

## DPPO

Flow-DPPO (Divergence Proximal Policy Optimization) [[15]](#ref15) is a strict Flow-GRPO variant that keeps GRPO's group advantages and the optional KL-vs-reference penalty, but **replaces the PPO ratio-clip with a divergence proximal constraint**. The argument is that the single-sample probability ratio is a noisy estimate of the true policy divergence, so ratio clipping over-constrains some steps and under-constrains others. Because the per-step policy in a flow model is Gaussian, the KL between the old and new policies is exact and cheap to compute. DPPO uses this in an **asymmetric divergence mask**: it zeroes the gradient for any sample whose per-step KL(current ‖ rollout-old) exceeds `kl_mask_threshold` *and* whose update would push the action further in the wrong direction (`ratio > 1 & adv > 0`, or `ratio < 1 & adv < 0`).

The two KL computations are **decoupled**: `kl_mask_type` selects the space of the trust-region mask's KL(current ‖ old), while `kl_type` selects the space of the optional KL(current ‖ reference) penalty.

```yaml
train:
    trainer_type: 'dppo'
    kl_mask_type: 'x-based'    # Trust-region mask KL(current||old) space: 'x-based' (next_latents_mean) or 'v-based' (noise_pred)
    kl_mask_threshold: 1.0e-6  # Per-step KL trust-region; larger keeps more samples
    kl_type: 'x-based'         # KL(current||reference) penalty space: 'x-based' or 'v-based'
    kl_beta: 1.0e-3            # Optional KL(current||reference) penalty (0 disables)
    kl_guidance_scale: 4.5     # CFG for the KL-vs-reference forward (Null = training guidance_scale)
```

Like GRPO, DPPO is **coupled** and must use SDE dynamics (`Flow-SDE`, `Dance-SDE`, `CPS`). `DPPOTrainingArguments` does not inherit `GRPOTrainingArguments` (no `clip_range`) — its field set is intentionally minimal. When `kl_beta > 0`, the KL-vs-reference term is evaluated at `kl_guidance_scale`; this is reflected in `DPPOTrainingArguments.get_preprocess_guidance_scale()` so negative prompts are encoded at preprocessing whenever `kl_guidance_scale > 1.0`. Example configs: `examples/dppo/lora/{flux2_klein_base,sd3_5}/geneval2_{single,multi}.yaml`.

## DPO

DPO (Direct Preference Optimization) [[11]](#ref11) is a **decoupled** algorithm that optimises a pairwise preference loss on flow-matching velocity targets. Instead of per-sample policy-gradient ratios, it forms chosen/rejected pairs within each group (based on per-sample advantages), then minimises a Bradley-Terry preference loss over the DSM errors of the two policies (current vs. frozen reference). To use this algorithm, set:

```yaml
train:
    trainer_type: 'dpo'
```

### Core Parameters

```yaml
train:
    beta: 2000.0              # DPO temperature; larger ⇒ sharper preference contrast.
    ref_param_device: 'cuda'  # Device to store frozen reference parameters ('cpu' or 'cuda').
```

### Pair Formation & Advantage

DPO forms chosen/rejected pairs at the **start** of `optimize()` after `prepare_feedback()` has stored per-sample advantages. The `advantage_aggregation` controls how multi-reward advantages are combined:

```yaml
train:
    advantage_aggregation: 'gdpo'  # Options: 'sum', 'gdpo'. 'gdpo' normalizes each reward independently.
    global_std: true               # Global std normalization across all samples (vs. per-prompt).
```

### Training Timestep Distribution

```yaml
train:
    num_train_timesteps: 1              # Number of freshly sampled training timesteps per pair.
    weighting_scheme: 'logit_normal'    # Options: 'logit_normal', 'uniform'.
    logit_mean: 0.0                     # Mean for logit-normal sampling.
    logit_std: 1.0                      # Std for logit-normal sampling.
    time_shift: 1.0                     # Shift parameter (1.0 = no shift).
    timestep_range: 0.99               # Float ⇒ (0, x); tuple ⇒ (lo, hi).
```

## DGPO

DGPO (Direct Group Preference Optimization) [[8]](#ref8) is a **decoupled** algorithm that optimises a group-level preference loss on flow-matching targets. In particular, DGPO optimizes group-level preferences directly, extending the Direct Preference Optimization (DPO) framework to handle pairwise groups instead of pairwise samples. In concrete coding practice, DGPO implements a gradient-equivalent loss which aggregates each group's advantage-weighted DSM delta (current vs. reference) through a sigmoid and reweights every sample's DSM loss by the resulting per-group scalar. Training samples use `trajectory_indices=[-1]` and `compute_log_prob=False`; fresh timesteps are drawn from `TimeSampler` at each optimisation step. To use this algorithm, set:

```yaml
train:
    trainer_type: 'dgpo'
```

Because the objective contrasts the current policy against a reference model, DGPO **always requires** a reference model (`requires_ref_model = True`).

### Core Loss Coefficients

```yaml
train:
    dpo_beta: 100.0           # DPO beta scaling for group preference; larger ⇒ sharper sigmoid weighting.
    kl_type: 'v-based'        # DGPO only supports v-based KL (other values are auto-coerced with a warning).
    kl_beta: 0.0              # KL penalty weight. 0 disables the KL term entirely.
    kl_cfg: 1.0               # CFG scale applied to the frozen reference. >1 enables CFG on the KL reference branch.
    guidance_scale: 4.5       # CFG during rollout process.
```

### Guidance on Hyper-parameter tuning

DGPO supports two modes: 1) rollout with CFG, training without CFG; 2) CFG-free in both rollout and training.

For the "rollout with CFG, training without CFG" mode, DGPO can achieve relatively fast training convergence and better OOD performance. As for the key hyperparameters, the reference model is typically frozen without CFG, the dpo_beta is generally set to 10 ~ 100 and clip_range is generally set to 1e-3 ~ 1e-2.

```yaml
# rollout with CFG, training without CFG
train:
    dpo_beta: 100.0           # DPO beta scaling for group preference; larger ⇒ sharper sigmoid weighting.
    kl_type: 'v-based'        # DGPO only supports v-based KL (other values are auto-coerced with a warning).
    kl_beta: 0.001            # KL penalty weight. 0 disables the KL term entirely.
    kl_cfg: 1.0               # CFG scale applied to the frozen reference. >1 enables CFG on the KL reference branch.
    guidance_scale: 4.5       # CFG during rollout process.
    clip_range: 1.0e-3        # PPO clip range (scalar is expanded to (-c, c)).
```

For the "CFG-free" mode, DGPO can achieve significantly faster convergence, but generally at the cost of some OOD performance. In this mode, it is recommended to use a small PPO-style clipping range by default: 1e-5 ~ 1e-4 for stable training. There are two settings for the reference model: one is to use a frozen reference model w/ CFG, in which case dpo_beta is typically set within the range of 10 ~ 100:

```yaml
#  CFG-free in both rollout and training. With frozen reference model.
train:
    dpo_beta: 100.0           # DPO beta scaling for group preference; larger ⇒ sharper sigmoid weighting.
    kl_type: 'v-based'        # DGPO only supports v-based KL (other values are auto-coerced with a warning).
    kl_beta: 0.001            # KL penalty weight. 0 disables the KL term entirely.
    kl_cfg: 4.5               # CFG scale applied to the frozen reference. >1 enables CFG on the KL reference branch.
    guidance_scale: 1.0       # CFG during rollout process.
    clip_range: 1.0e-5        # PPO clip range (scalar is expanded to (-c, c)).
```

Another choice for the reference model in "CFG-free" mode is to use an EMA model as a dynamic reference model, as proposed in TDM-R1 [[12]](#ref12). In this case, dpo_beta is typically set within a larger range of 2000 ~ 5000:

```yaml
#  CFG-free in both rollout and training. With dynamic reference model.
train:
    dpo_beta: 2000.0           # DPO beta scaling for group preference; larger ⇒ sharper sigmoid weighting.
    kl_type: 'v-based'        # DGPO only supports v-based KL (other values are auto-coerced with a warning).
    kl_beta: 0.001            # KL penalty weight. 0 disables the KL term entirely.
    kl_cfg: 1.0               # CFG scale applied to the reference. >1 enables CFG on the KL reference branch.
    guidance_scale: 1.0       # CFG during rollout process.
    clip_range: 1.0e-5        # PPO clip range (scalar is expanded to (-c, c)).
```


### Shared RNG across Groups

Cross-rank-deterministic sampling of both the training timesteps and the per-group noise (seeded from `(seed, epoch, inner_epoch, uid)`). The per-group noise is **timestep-invariant** — all training timesteps within an epoch share the same noise, matching the reference implementation. No `dist.broadcast` / RNG fork is used:

```yaml
train:
    use_shared_noise: true    # Same noise for every sample within a group at each step.
```

### PPO-style Clipping and EMA reference model

A fast-tracking EMA copy of the trainable parameters (`ema_ref`, distinct from the slow sampling EMA) acts as the "old policy" for PPO-style clipping on the DSM / KL losses:

```yaml
train:
    clip_dsm: true            # Clip the DSM loss when the ratio exits clip_range.
    clip_kl: false            # Optionally clip the KL loss using the same ratio mask.
    clip_range: 1.0e-2        # PPO clip range (scalar is expanded to (-c, c)).
    adv_clip_range: 5.0       # Advantage clipping range.
    use_ema_ref: false        # If true, use ema_ref (not the frozen ref) as the DGPO loss reference (TDM-R1 dynamic ref).

    ema_ref_max_decay: 0.3    # Cap of the adaptive decay.
    ema_ref_ramp_rate: 1.0e-3 # Adaptive decay = min(ema_ref_max_decay, ema_ref_ramp_rate * step).
    ema_ref_device: 'cuda'    # Where ema_ref parameters live.
```

`clip_dsm`, `clip_kl`, or `use_ema_ref` being enabled triggers the creation and per-step update of `ema_ref`; otherwise no fast EMA is maintained.

### Sampling Policy Switch

```yaml
train:
    off_policy: false         # If true, use the slow sampling EMA for trajectory generation from step 0.
    switch_ema_ref: 200       # After this many optimizer steps, swap to ema_ref (fast EMA) for sampling.
```

### Training Timestep Distribution

```yaml
train:
    num_train_timesteps: 0    # 0 ⇒ int(num_inference_steps * (timestep_range[1] - timestep_range[0])).
    time_sampling_strategy: 'discrete'  # Options: discrete, discrete_with_init, discrete_wo_init, uniform, logit_normal.
    time_shift: 3.0           # Shift for logit_normal / uniform strategies.
    timestep_range: 0.6       # Float ⇒ (0, x); tuple ⇒ (lo, hi) along the 1000→0 denoise axis.
```

> **Note**: DGPO feeds scheduler-scale timesteps (`[0, 1000]`) into `flow_match_sigma` before constructing `x_t = (1 - σ) x_0 + σ ε`. Training directly on unscaled timesteps would drive reward downward — the σ-scaling is mandatory for correct flow-matching behaviour.

### Group Completeness

DGPO's group-level sigmoid reweighting is only meaningful if every optimizer step sees a **complete group** (all `K = group_size` copies of each prompt). Flow-Factory guarantees this by requiring `GroupDistributedSampler` for DGPO (auto-forced by `Arguments._resolve_sampler_type`).

**How it works**: `GroupDistributedSampler` yields the same prompt-index sequence on every rank; each prompt appears `K / W` times per rank (`W` = `num_replicas`). Since all ranks see the same prompts, local `torch.unique` produces a cross-rank-consistent dense group-id space — no `gather_samples` or cross-rank id coordination is needed. The single `accelerator.reduce` inside `_compute_group_dgpo_loss` sums partial per-rank contributions to recover the full-group sigmoid weight.

**Geometric constraint**: `(num_replicas × per_device_batch_size) % group_size == 0` must hold so that every global micro-batch packs an integer number of complete groups. `Arguments._align_for_group_distributed` auto-adjusts `group_size` (and then `unique_sample_num_per_epoch`) at init time to satisfy this, so no manual tuning is needed.

For a complete runnable setup, see `examples/dgpo/lora/sd3_5/default.yaml`.

## DiffusionNFT

This algorithm is introduced in [[7]](#ref7). Unlike GRPO, which couples sampling dynamics with training timesteps, **DiffusionNFT** decouples them entirely by optimizing a contrastive objective directly on the forward flow-matching process.

Concretely, DiffusionNFT contrasts implicit positive and negative policies ($v_\theta^+$ and $v_\theta^-$), weighted by a normalized reward $r \in [0, 1]$, to identify a policy improvement direction *without* requiring tractable likelihood estimation or SDE-based sampling. This makes the algorithm inherently solver-agnostic.

To use this algorithm, set:

```yaml
train:
    trainer_type: 'nft'
```

With `sample_weighting: src`, NFT keeps the two SRC probabilities distinct.
The weighted-centered scalar signal controls the positive/negative branch
probability, while `sample_weight = K * p_i` multiplies the complete per-sample
NFT regression objective. The multiplier therefore affects both the reward-
driven contrast and the old-policy anchor after the trainable predictor moves.
This mode requires `advantage_aggregation: sum` and fixed nonnegative reward
weights. It does not reweight the independent KL penalty. With
`off_policy: true`, SRC reshapes the empirical EMA-rollout distribution and
the EMA prediction supplies `v_old`; the objective remains well defined, but
the fresh-reference raw-reward alignment interpretation no longer describes
the complete gradient. Startup logging reports the reference mode explicitly.
When `global_std: true`, the normalizer remains the baseline unweighted global
rollout standard deviation; otherwise NFT uses the SRC-weighted prompt
standard deviation. Logging reports both the frozen contribution that produced
the SRC distribution and the contribution after weighted recentering.

Since DiffusionNFT decouples training from sampling dynamics, you can freely choose the sampling solver. Using the `ODE` solver during sampling typically yields higher image quality:

```yaml
train:
  num_train_timesteps: 2 # Number of timesteps to train on. Set `null` to all timesteps.
  time_sampling_strategy: discrete_with_init # Options: uniform, logit_normal, discrete, discrete_with_init, discrete_wo_init
  time_shift: 3.0
  timestep_fraction: 0.3 # Train using only the first 30% of timesteps.

scheduler:
    dynamics_type: 'ODE' # Other options are also available.
```

> **Note**: Since Reinforcement Learning typically requires exploration, it is often beneficial to experiment with SDE-based `dynamics_type` settings as well. Using `CPS`[[9]](#ref9) for NFT sampling is also a good choice.

### Old Policy via EMA

The original DiffusionNFT implementation maintains two separate EMA copies of the model: one for general EMA smoothing and one as the "old policy" used for off-policy sampling. Flow-Factory simplifies this design by retaining only a single EMA copy that serves as the old policy. This reduces memory overhead while preserving the core stabilization mechanism.

When `off_policy` is enabled, the EMA model is used to generate trajectories during sampling, while the current policy is optimized against these trajectories. This off-policy setup stabilizes training by preventing the sampling distribution from shifting too rapidly.

```yaml
train:
  off_policy: true  # Use EMA parameters for off-policy sampling
  ema_decay_schedule: "piecewise_linear"  # Options: constant, power, linear, piecewise_linear, cosine, warmup_cosine
  ema_decay: 0.5        # EMA decay rate (0 to disable)
  ema_update_interval: 1  # EMA update interval (in epochs)
  ema_device: "cuda"      # Device to store EMA model (options: cpu, cuda)
```

> **Tip**: The `piecewise_linear` schedule is recommended for DiffusionNFT. It starts with a lower decay rate to allow faster initial policy divergence and gradually increases the decay to stabilize later training. You can fine-tune this behavior with `flat_steps` and `ramp_rate`.

`crossover-nft` uses the same fixed-population genetic algorithm and
`survivor_score` setting, but keeps NFT's decoupled matching objective. When
covariance selection is enabled, NFT uses the locally linear score
`min_k w_k [Sigma w]_k / Z_S^2`. Under the required group-local scalar
normalization, `Z_S^2` is the hypothetical subset's scalar-reward population
variance and is recomputed for every add or removal. When
`off_policy: true`, both parent and offspring denoising use the EMA sampling
policy. Like the GRPO-Guard variant, it currently requires pointwise rewards
and complete prompt groups on one rank (`group_contiguous`).

## AWM: Advantage Weighted Matching

This algorithm is introduced in [[10]](#ref10). **Advantage Weighted Matching** further aligns RL optimization with the flow-matching pretraining objective by weighting the standard velocity matching loss with per-sample advantages. This formulation incorporates reward-based guidance directly into the velocity matching loss, effectively aligning the optimization target with the original flow-matching objective.

Like DiffusionNFT, AWM decouples training from sampling dynamics and is therefore solver-agnostic. To use this algorithm, set:

```yaml
train:
    trainer_type: 'awm'
```

The relevant sampling and timestep configuration parameters are the same as those described in the [DiffusionNFT](#diffusionnft) section.

### Training Stability

AWM typically converges faster than other algorithms due to its direct advantage weighting on the velocity matching loss. However, this rapid update dynamic also makes it more prone to training instability — the policy can diverge quickly if left unconstrained, leading to reward hacking or training collapse.

To stabilize AWM training, it is strongly recommended to combine **EMA-based KL regularization** with **PPO-style clipping**:

```yaml
train:
  trainer_type: 'awm'
  # EMA KL regularization: penalizes deviation from the EMA-smoothed policy
  ema_kl_beta: 0.1        # Coefficient of KL loss between current policy and EMA policy
  ema_decay: 0.9           # EMA decay rate
  ema_decay_schedule: 'power'  # Options: constant, power, linear, piecewise_linear, cosine, warmup_cosine
  ema_update_interval: 1   # EMA update interval (in epochs)
  ema_device: "cuda"
  # PPO-style clipping: prevents excessively large policy updates
  clip_range: 1.0e-5       # Clipping range for the policy ratio
  adv_clip_range: 5.0      # Advantage clipping range
```

> ‼️ **Important**: Disabling both `ema_kl_beta` and `clip_range` simultaneously is **not recommended** for AWM, as the unconstrained advantage weighting can easily lead to training collapse. In practice, `ema_kl_beta` serves as a soft constraint that keeps the current policy close to a moving average, while `clip_range` provides a hard constraint on per-step policy updates.

### AWM Weighting

AWM computes a per-sample matching loss $\ell = \|v_\theta(x_t, t) - ({\epsilon} - {x}_0)\|^2$ and then applies a weighting function $w(\ell, t)$ before multiplying by the advantage. Different weighting strategies control how the raw matching loss magnitude and timestep position influence the gradient signal:

```yaml
train:
  awm_weighting: 'ghuber'  # Options: Uniform, t, t**2, huber, ghuber
  ghuber_power: 0.25        # Power parameter for generalized Huber weighting (only used with 'ghuber')
```

| Weighting  | Formula $w(\ell, t)$                                                  | Description                                                                                           |
|------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `Uniform`  | $\ell$                                                                | No reweighting. All timesteps contribute equally.                                                     |
| `t`        | $t \cdot \ell$                                                        | Linear timestep weighting. Upweights noisier (larger $t$) timesteps.                                  |
| `t**2`     | $t^2 \cdot \ell$                                                      | Quadratic timestep weighting. More aggressively upweights noisier timesteps.                          |
| `huber`    | $t \cdot (\sqrt{\ell + \varepsilon} - \varepsilon)$                   | Huber-style loss that suppresses large matching errors, weighted by $t$.                              |
| `ghuber`   | $\frac{t}{p} \cdot ((\ell + \varepsilon)^{p} - \varepsilon^{p})$     | Generalized Huber loss with power $p$ (`ghuber_power`). Provides tunable robustness against outliers. |

Here $\varepsilon$ is a small constant for numerical stability and $p$ denotes `ghuber_power` (default `0.25`).

> **Tip**: `ghuber` with a small power (e.g., `0.25`) provides a good balance between robustness and gradient signal strength. `Uniform` is the simplest baseline and works well when reward signals are clean and low-variance.

> **Note**: Like DPO, DGPO, DiffusionNFT, and AWM are foward-diffusion based RL algorithms, which decouples training from sampling dynamics and is solver-agnostic — any ODE/SDE solver can be used for trajectory generation.


## CRD: Centered Reward Distillation

This algorithm is introduced in [[13]](#ref13). **Centered Reward Distillation (CRD)** is a forward-process RL method that matches implicit model rewards (estimated from prediction error in velocity space) with centered external rewards. The key insight is that the unknown prompt-dependent normalizer cancels under *within-prompt centering*, yielding a well-posed reward-matching objective.

CRD maintains two named parameter snapshots alongside the current model:
- **Old model** (`_crd_old`): used to estimate implicit rewards via prediction error difference.
- **Sampling model** (`_crd_sampling`): used for off-policy rollout generation, blended toward the current model over time.

To use this algorithm, set:

```yaml
train:
    trainer_type: 'crd'
```

### Key Hyperparameters

```yaml
train:
  trainer_type: 'crd'

  # CRD loss
  crd_beta: 1.0           # Scaling factor for reward-matching loss
  crd_loss_type: 'mse'    # Options: mse, bce
  use_old_for_loss: true  # Use old model snapshot for implicit reward (recommended)
  adaptive_logp: true     # Adaptive per-sample weighting of implicit reward terms
  weight_temp: -1.0       # Softmax temperature τ for centering (-1 = uniform/τ→∞)

  # Model snapshot decay schedules
  # Format: "start_step-start_value-slope-end_value" or int preset key
  old_model_decay: "0-0.25-0.005-0.999"      # Paper (OCR): min(0.25 + 0.005t, 0.999)
  sampling_model_decay: "75-0.0-0.0075-0.999" # Paper (OCR): delayed start at step 75

  # KL regularization anchored to CFG-guided pretrained reference
  kl_beta: 0.1            # KL coefficient
  kl_cfg: 4.5             # CFG scale for teacher reference model
  reward_adaptive_kl: true  # Scale KL by reward to accelerate early learning
  ref_param_device: 'cuda'

  # Timestep sampling
  timestep_range: 0.99    # Top 99% of denoising steps (original CRD default)
  num_train_timesteps: 20
  time_sampling_strategy: discrete
  time_shift: 3.0

  # Advantage clipping
  adv_clip_range: 5.0
```

### Centering Modes (`weight_temp`)

| `weight_temp` | Mode | Description |
|---|---|---|
| `< 0` | Uniform (τ→∞) | Simple mean centering; recommended default |
| `== 0` | Hard selection | Positive pool (adv > 0) vs negative pool (adv < 0) |
| `> 0` | Softmax temperature | Dual-direction: `softmax(adv/τ)` and `softmax(-adv/τ)` |


## DiffusionOPD: On-Policy Distillation

This algorithm is introduced in [[14]](#ref14). **DiffusionOPD** is a *decoupled-paradigm* multi-task distillation method: instead of jointly optimizing several rewards from scratch, it first trains one task-specialized **teacher** per task (e.g. GenEval, OCR, aesthetics) and then distills their capabilities into a single unified **student** along the student's own rollout trajectories. This reduces reward conflict and catastrophic forgetting relative to multi-reward RL.

Unlike the policy-gradient algorithms above, the loss is a closed-form **per-step KL on the denoising transition** — a pathwise mean-matching objective that covers both stochastic SDE samplers and deterministic ODE samplers:

```
kl_div_j = 0.5 * || mu_S - mu_T ||^2 / denom
```

where `mu_S` / `mu_T` are the student / teacher transition means at the student-visited state `x_j`, and `denom` is the scheduler's transition variance for the active dynamics (centralized in `scheduler.get_kl_divergence_denominator`):

| `dynamics_type` | `denom` | resulting `kl_div_j` |
|---|---|---|
| `ODE` | `1.0` | pure mean matching: `0.5 * ||μ_S − μ_T||²` |
| `Flow-SDE`, `Dance-SDE` | `std_dev_t² · (-dt)` | Gaussian transition KL: `||μ_S − μ_T||² / (2 σ̄²)` |
| `CPS` | `std_dev_t²` | `||μ_S − μ_T||² / (2 std_dev_t²)` |

There is no loss-scaling coefficient (DiffusionOPD has no REINFORCE term). Rewards are used **only** for periodic eval monitoring (`evaluate()`), never in the distillation loss.

### How it works (2-pass per epoch)

Built directly on the multi-dataset infrastructure (`data.datasets`, per-source `source`/`source_id`, `train_dataloaders_by_source`), so each teacher is routed to one or more training datasets:

1. **`sample()`** — the student rolls out on-policy trajectories over the multi-source dataloader (each sample tagged with its `source`), reusing the standard sampling pipeline.
2. **`optimize()` PASS 1** (`no_grad`) — for each teacher (exactly **one** weight swap, via the named-parameter snapshot), forward over its routed samples' stored states `x_j` and cache the teacher means `mu_T` on each sample.
3. **`optimize()` PASS 2** (student params only) — a standard gradient loop forwards the student at the same `x_j`, matching each sample's `mu_S` to its own cached `mu_T` (a micro-batch may mix teachers; the batch-mean is an implicit per-teacher KL averaged over the batch).

Teacher swaps are thus **M-per-epoch** (one per teacher), the gradient loop runs with student params only (no autocast-cache toggling, no DDP bypass), and the loss is a clean student-vs-cached-target MSE.

Which denoising steps are distilled is set by `train.timestep_range` (default `0.99`), the same fraction idiom NFT uses: a float `f` selects the band `[0, f]` of the trajectory's step indices (the first `f`-fraction of denoising steps, skipping the near-clean tail), and a tuple is an explicit `[lo, hi]` band. This reproduces upstream DiffusionOPD's `timestep_fraction` and is **dynamics-agnostic** — it selects by trajectory step index rather than the SDE-only stochastic-step set, so it works identically under ODE and SDE.

### Teacher loading

Teachers are **LoRA-only** (full-parameter teachers are deferred). Each teacher checkpoint is loaded into a named-parameter snapshot and **must share the student's LoRA architecture** (same `target_components` / target modules, compatible rank/alpha), because it is loaded into the student's active adapter slot. Local paths and Hugging Face Hub repo ids are both accepted.

To use this algorithm, set:

```yaml
train:
  trainer_type: 'diffusion-opd'

  teachers:
    - name: "geneval-teacher"                            # unique id (named snapshot + log keys)
      path: "quanhaol/DiffusionOPD/GenEvalTeacher/lora"  # local path or HF spec owner/repo[/subfolder][@rev]
      applicable_datasets: [geneval]                     # distill on geneval rollouts
      # guidance_scale: 4.5                              # (optional) per-teacher CFG override (null = student CFG)
    - name: "ocr-teacher"
      path: "quanhaol/DiffusionOPD/OCRTeacher/lora"
      applicable_datasets: [ocr]

  teacher_param_device: 'cuda'  # teacher snapshot device: 'cuda' (fast swaps) / 'cpu' (low VRAM)
  guidance_scale: 1.0           # student CFG for rollout + forward
  timestep_range: 0.99          # distill the first 99% of denoising steps (upstream timestep_fraction)

scheduler:
  dynamics_type: "ODE"  # mean matching; switch to Flow-SDE + noise_level>0 for SDE distillation
  noise_level: 0.0
```

Each teacher's `applicable_datasets` must reference declared `data.datasets[*].name` entries (validated at config load). The config schema allows several teachers to share a dataset for a future multi-teacher/ensemble trainer, but the current `DiffusionOPDTrainer` requires exactly one teacher per dataset and raises otherwise. See [`examples/opd/lora/sd3_5/`](../examples/opd/lora/sd3_5/) for two complete configs (`DiffusionOPD_aligned.yaml` to reproduce official results).

## References

* <a name="ref1"></a>[1] [**Flow-GRPO:** Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
* <a name="ref2"></a>[2] [**DanceGRPO:** Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
* <a name="ref3"></a>[3] [**MixGRPO:** Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802)
* <a name="ref4"></a>[4] [**TempFlow-GRPO:** When Timing Matters for GRPO in Flow Models](https://arxiv.org/abs/2508.04324)
* <a name="ref5"></a>[5] [**GRPO-Guard:** Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://arxiv.org/abs/2510.22319)
* <a name="ref6"></a>[6] [**PaCo-RL**: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling](https://arxiv.org/abs/2512.04784)
* <a name="ref7"></a>[7] [**DiffusionNFT**: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)
* <a name="ref8"></a>[8] [**DGPO**: Reinforcing Diffusion Models by Direct Group Preference Optimization](https://arxiv.org/abs/2510.08425)
* <a name="ref9"></a>[9] [**<u>C</u>oefficients-<u>P</u>reserving <u>S</u>ampling** for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952)
* <a name="ref10"></a>[10] [**<u>A</u>dvantage <u>W</u>eighted <u>M</u>atching**: Aligning RL with Pretraining in Diffusion Models](https://arxiv.org/abs/2509.25050)
* <a name="ref11"></a>[11] [**Diffusion-DPO**: Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)
* <a name="ref12"></a>[12] [**TDM-R1**: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward](https://arxiv.org/abs/2510.08425)
* <a name="ref13"></a>[13] [**CRD**: Diffusion Reinforcement Learning via Centered Reward Distillation](https://arxiv.org/abs/2603.14128)
* <a name="ref14"></a>[14] [**DiffusionOPD**: A Unified Perspective of On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2605.15055)
* <a name="ref15"></a>[15] [**Flow-DPPO**: Divergence Proximal Policy Optimization for Flow Matching Models](https://arxiv.org/abs/2606.11025) ([Code](https://github.com/Tencent-Hunyuan/UniRL/tree/main/FlowDPPO#readme))
