# MLP Architecture & Loss Functions

Documentation of the FireMLP model, training pipeline, loss functions,
and grid search system.

---

## 1. FireMLP Architecture

**Source:** `models/firemlp.py`

### Overview

Variable-depth MLP that maps 12 aggregate features to a single fire
probability.

```
Input (12 features) -> [Hidden Layers] -> Output (1 logit)
```

Output is a **raw logit**; apply `sigmoid()` externally for probability.

### Construction

Each hidden layer consists of:

```
Linear(in, out) -> ReLU -> Dropout(p)   (dropout omitted if p=0)
```

The final layer is a plain `Linear(last_hidden, 1)` with no activation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_features` | int | 12 | Input dimension |
| `hidden_layers` | list[int] | [64, 32] | Hidden layer sizes |
| `dropout` | float | 0.0 | Dropout probability (0 = disabled) |

### Example Architectures

```
Default:       12 -> 64 -> 32 -> 1           (2,945 params)
Grid search:   12 -> 64 -> 64 -> 64 -> 32 -> 1
               12 -> 128 -> 64 -> 32 -> 1
               12 -> 128 -> 128 -> 128 -> 64 -> 1
```

### Input Features

Defined in `lib/inference.py` as `FEATURE_NAMES`:

| # | Name | Physical Meaning |
|---|------|-----------------|
| 1 | T4_max | Peak 3.9 &mu;m brightness temperature (fire intensity spike) |
| 2 | T4_mean | Mean thermal state (normalizes peak) |
| 3 | T11_mean | Background 11.3 &mu;m temperature (stable reference) |
| 4 | dT_max | Max T4&minus;T11 difference (core fire signature) |
| 5 | SWIR_max | Peak 2.2 &mu;m radiance (fire emits strongly in SWIR) |
| 6 | SWIR_mean | Mean SWIR radiance (normalizes peak) |
| 7 | Red_mean | Mean red radiance (~0 at night, implicit day/night) |
| 8 | NIR_mean | Mean NIR radiance (~0 at night, implicit day/night) |
| 9 | NDVI_min | Lowest vegetation index (burn scar indicator) |
| 10 | NDVI_mean | Mean vegetation index |
| 11 | NDVI_drop | First NDVI &minus; min NDVI (temporal vegetation loss) |
| 12 | obs_count | Number of observations (reliability indicator) |

---

## 2. Loss Functions

**Source:** `lib/losses.py`

All custom losses share a unified forward signature:

```python
loss = criterion(logits, y, w, gt_mask=None)
```

where `logits` are raw model outputs, `y` is the target, `w` is per-sample
weights, and `gt_mask` identifies ground-truth (pre-burn) samples.

### 2.1 Weighted Binary Cross-Entropy (BCE)

The default loss. Uses PyTorch's `BCEWithLogitsLoss` with per-sample
weighting:

```
loss = mean(BCE(logits, y) * w)
```

- Numerically stable (operates on logits, not probabilities).
- Fast convergence.
- Fine-grained control via importance weights.

### 2.2 SoftErrorRateLoss

Differentiable approximation of the error rate metric
`(FN + FP) / P` where P is total actual fire pixels.

```
p = sigmoid(logits)
soft_FP = sum(w * p * (1 - y))
soft_FN = sum(w * (1 - p) * y)
loss = (soft_FN + soft_FP) / P_total
```

| Parameter | Description |
|-----------|-------------|
| `P_total` | Total fire pixels in full training set (pre-computed constant) |
| `gt_fp_penalty` | Extra penalty for FP on ground-truth pixels (default 0) |

When `gt_fp_penalty > 0` and `gt_mask` is provided:

```
gt_FP = sum(p * (1 - y) * gt_mask)
loss += gt_fp_penalty * gt_FP / P_total
```

**Use case:** Direct optimization of the evaluation metric. Uses uniform
weights by default (importance weights are irrelevant since the loss
directly counts errors).

### 2.3 TverskyLoss

Generalized Dice loss with tunable FP/FN asymmetry:

```
p = sigmoid(logits)
TP = sum(w * p * y)
FP = sum(w * p * (1 - y))
FN = sum(w * (1 - p) * y)
Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
loss = 1 - Tversky
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.3 | FP penalty weight |
| `beta` | 0.7 | FN penalty weight |
| `smooth` | 1.0 | Prevents division by zero |

- `alpha < beta` (default): recall-biased (penalizes missed fires more).
- `alpha = beta = 0.5`: reduces to standard Dice loss.
- `alpha > beta`: precision-biased (penalizes false alarms more).

### 2.4 FocalErrorRateLoss

Error-rate loss with focal modulation that upweights hard examples:

```
p = sigmoid(logits)
pt = y * p + (1 - y) * (1 - p)          # correct-class probability
modulator = (1 - pt) ^ gamma            # high for misclassified samples
soft_FP = sum(modulator * w * p * (1 - y))
soft_FN = sum(modulator * w * (1 - p) * y)
loss = (soft_FN + soft_FP) / P_total
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` | 2.0 | Focal strength (0 = SoftErrorRateLoss) |

**Use case:** When difficult boundary cases need more training attention.

### 2.5 CombinedLoss

Weighted blend of BCE and soft error-rate:

```
loss = lambda * weighted_BCE + (1 - lambda) * SoftErrorRate
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | 0.5 | BCE weight (1&minus;&lambda; for error-rate) |

**Use case:** Balances convergence stability (BCE) with metric alignment
(error-rate).

### Loss Comparison Summary

| Loss | Optimizes | Weights | Best For |
|------|-----------|---------|----------|
| Weighted BCE | Log-likelihood | Importance &times; inv-freq | Stable default |
| SoftErrorRate | (FN+FP)/P directly | Uniform (default) | Metric-aligned training |
| Tversky | Soft Dice with asymmetry | Importance &times; inv-freq | Recall-biased detection |
| FocalErrorRate | Hard-example error rate | Importance &times; inv-freq | Difficult boundaries |
| Combined | BCE + error rate | Importance &times; inv-freq | Hybrid stability |

---

## 3. Pixel Weight Computation

**Source:** `lib/losses.py` &rarr; `compute_pixel_weights()`

### Categories

| Category | Condition | Default Importance |
|----------|-----------|-------------------|
| GT (ground truth) | `flight == '24-801-03'` | 10.0 |
| Fire | `y == 1 AND NOT GT` | 5.0 |
| Other | `y == 0 AND NOT GT` | 1.0 |

### Formula

```
w_i = importance_category * (N_total / N_category)
w_normalized = w / mean(w)
```

The inverse-frequency term `(N_total / N_category)` corrects for class
imbalance; the importance multiplier encodes domain knowledge. Weights are
normalized to mean=1 so that gradient scale remains stable regardless of
configuration.

---

## 4. Training Loop

**Source:** `train_mlp.py` &rarr; `train_model()`

### Optimizer & Scheduler

- **Optimizer:** Adam with configurable learning rate (default 1e-3).
- **Scheduler:** CosineAnnealingLR with `T_max = n_epochs` (smooth LR
  decay from initial value to near zero).
- **Gradient clipping:** Max norm 1.0 (prevents exploding gradients).

### Per-Epoch Steps

1. **Forward pass** on each mini-batch (default batch size: 4096).
2. **Loss computation** with per-sample weights and optional GT mask.
3. **Backward pass** and gradient clipping.
4. **Adam step**.
5. **Accumulate metrics:** count train FP, FN, TP at threshold 0.5.
6. **Validation** (optional): compute test FP, FN, TP without gradients.
7. **Scheduler step** (update learning rate).
8. **Checkpoint** every `save_every` epochs (default: 25).

### Loss History

Stored as `(n_epochs, 7)` array with columns:

```
[loss, train_FP, train_FN, train_TP, test_FP, test_FN, test_TP]
```

### Resume Support

Checkpoints store optimizer and scheduler state dicts. Training can be
resumed from any saved epoch:

```python
train_model(..., resume_from='checkpoint/fire_detector_partial.pt')
```

### Return Value

```python
{
    'model': FireMLP,               # trained model (CPU)
    'loss_history': ndarray,        # (n_epochs, 7)
    'epochs_completed': int,
    'optimizer_state': dict,
    'scheduler_state': dict,
}
```

---

## 5. Grid Search

**Source:** `configs/grid_search.yaml` + `train_mlp.py`

### Configuration

```yaml
epochs: 120
batch_size: 4096
save_every: 25
metric: error_rate          # minimize (FN+FP)/P

search_space:
  loss: [bce, error-rate]
  layers:
    - [64, 64, 64, 32]
    - [128, 64, 32]
    - [128, 128, 128, 64]
  learning_rate: [0.01, 0.001, 0.0001]
  importance_weights:       # BCE only; error-rate uses uniform
    - {gt: 10, fire: 5, other: 1}
    - {gt: 15, fire: 3, other: 1}
    - {gt: 5, fire: 10, other: 1}
```

### Grid Size

- **BCE:** 3 architectures &times; 3 LRs &times; 3 weight configs = 27 runs
- **Error-rate:** 3 architectures &times; 3 LRs &times; 1 (uniform weights) = 9 runs
- **Total:** 36 combinations

### Best Model Selection

```
1. Filter: flight-03 (pre-burn) FP <= 70
2. Among eligible: pick lowest (FP + FN) on test set
3. Fallback: if no eligible model, pick lowest flight-03 FP
```

### Crash Recovery

- Results saved to JSON after each run.
- On restart, completed runs are skipped; partial runs are resumed.
- Checkpoints saved every 25 epochs for mid-run recovery.

### Output Artifacts

| File | Contents |
|------|----------|
| `checkpoint/fire_detector_mlp_best.pt` | Best model checkpoint |
| `results/grid_search_mlp_results.json` | All run metrics |
| `results/grid_search_mlp_summary.csv` | Summary table |
| `configs/best_model_mlp.yaml` | Best hyperparameters |

---

## 6. Inference

**Source:** `lib/inference.py`

### Batch Inference

```python
from lib.inference import load_model, predict

model, scaler = load_model('checkpoint/fire_detector_mlp_best.pt')
preds, probs = predict(model, scaler, X_raw)
# preds: bool array, probs: float array [0, 1]
```

Pipeline: NaN &rarr; 0 &rarr; StandardScaler &rarr; model &rarr; sigmoid &rarr; threshold.

### Real-Time Inference

```python
from lib.inference import load_fire_model

detector = load_fire_model()  # auto-discovers best checkpoint
fire_mask = detector.predict_from_gs(gs)      # bool grid
prob_grid = detector.predict_proba_from_gs(gs) # float grid
```

Uses `compute_aggregate_features(gs)` from `lib/fire.py` to extract
12 features directly from grid-state accumulators (no DataFrame needed).

### Checkpoint Discovery

`_find_checkpoint()` searches in order:

1. `checkpoint/fire_detector_mlp_best.pt` (grid search winner)
2. `checkpoint/fire_detector_best.pt` (legacy)
3. `checkpoint/fire_detector_bce.pt` (legacy)
4. `checkpoint/fire_detector_error-rate.pt` (legacy)
5. `checkpoint/fire_detector.pt` (fallback)

### Checkpoint Contents

```python
{
    'model_state': state_dict,
    'n_features': 12,
    'hidden_layers': [64, 32],
    'mean': ndarray,            # scaler means (non-thermal only)
    'std': ndarray,             # scaler stds (non-thermal only)
    'scaler': StandardScaler,   # fitted on non-thermal features
    'T_ignition': 573.15,       # thermal normalization divisor [K]
    'normalization': 'hybrid',  # thermal/T_ign + scaler on non-thermal
    'threshold': 0.5,
    'feature_names': [...],
    'loss_fn': 'bce',
    'loss_history': [...],
    'epoch_metrics': {...},
    'optimizer_state': dict,    # for training resume
    'scheduler_state': dict,
}
```

---

## 7. Evaluation Metrics

**Source:** `lib/training.py` &rarr; `compute_error_rate()`

Primary metric:

```
Error Rate = (FN + FP) / P     where P = TP + FN (actual fire pixels)
```

Standard classification metrics are also computed:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

All metrics reported as **absolute counts** (TP, FP, FN) per user
preference, not percentages.
