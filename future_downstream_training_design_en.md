# Future Design Proposal: Downstream Task Training on Frozen V-JEPA Representations

## 1. Purpose

This document describes a future extension of the current V-JEPA-based video risk analysis prototype.

The current MVP uses V-JEPA as a frozen video representation model. Input videos are split into short clips, each clip is encoded into a learned embedding, and representation shifts between clips are used for simple change detection and rule-based explanation.

Once labeled training data becomes available, the system can be extended from heuristic change detection to supervised downstream task learning.

The core design principle is:

```text
Keep V-JEPA frozen.
Use V-JEPA only as a video feature extractor.
Train lightweight downstream probes on top of the extracted embeddings.
```

This enables domain-specific risk classification, event classification, and near-future risk prediction without fine-tuning the large video backbone.

---

## 2. Current MVP Architecture

The current pipeline can be summarized as follows:

```text
Input video
  ↓
Frame extraction
  ↓
Clip segmentation
  ↓
Frozen V-JEPA encoder
  ↓
Clip-level embeddings
  ↓
Representation shift detection
  ↓
Rule-based risk scoring
  ↓
Template-based explanation
```

The current system can detect that "something changed" in the video, but it does not yet learn domain-specific semantic labels such as:

- near collision
- worker entering a dangerous area
- forklift approaching a person
- object falling
- abnormal crowd movement
- traffic bottleneck
- unsafe infrastructure condition

Those meanings require downstream supervision.

---

## 3. Future Architecture with Labeled Data

Once labeled data is available, the system should evolve into the following structure:

```text
Input video
  ↓
Frame extraction
  ↓
Clip segmentation
  ↓
Frozen V-JEPA encoder
  ↓
Clip-level embeddings
  ├─ Representation shift detector
  ├─ Risk classification probe
  ├─ Event-type classification probe
  └─ Temporal risk prediction probe
  ↓
Risk / event fusion
  ↓
Domain-aware explanation generator
  ↓
Frontend / report / audit log
```

The important separation of responsibility is:

```text
V-JEPA:
  Learns and provides general-purpose video representations.

Downstream probes:
  Learn task-specific mappings from embeddings to business/domain labels.

Rule and explanation layer:
  Converts model outputs into operational descriptions.
```

---

## 4. Downstream Tasks

### 4.1 Clip-Level Risk Classification

Objective:

```text
clip embedding → low / medium / high risk
```

Example labels:

```text
low
medium
high
```

This is the simplest supervised extension and should be implemented first.

Recommended model:

- Logistic Regression
- Linear SVM
- Random Forest
- LightGBM
- Small MLP if enough data is available

Initial recommendation:

```text
StandardScaler + LogisticRegression(class_weight="balanced")
```

Reason:

- easy to train
- easy to inspect
- robust with small datasets
- suitable as a first linear probe

---

### 4.2 Event-Type Classification

Objective:

```text
clip embedding → event type
```

Example labels:

```text
normal
sudden_motion
approach
crossing
occlusion
object_fall
near_collision
congestion
unknown
```

This probe improves explanation quality.

Without an event-type probe, explanations remain abstract:

```text
Large representation shift was detected.
```

With an event-type probe, explanations become more useful:

```text
This segment resembles a near-collision pattern.
```

---

### 4.3 Temporal Risk Prediction

Objective:

```text
sequence of clip embeddings → near-future risk score
```

Instead of classifying one clip independently, this task uses neighboring clips:

```text
[clip_t-2, clip_t-1, clip_t, clip_t+1, clip_t+2]
```

Possible outputs:

```text
risk_score_next_2s
risk_score_next_5s
probability_of_high_risk
```

This is closer to real operational risk prediction because risk is often defined by temporal progression, not by a single static frame.

Recommended first implementation:

```text
Concatenate neighboring embeddings → Logistic Regression / MLP
```

Later implementation:

```text
Embedding sequence → GRU / Transformer / temporal attention probe
```

---

### 4.4 Anomaly Detection on Normal Embeddings

Objective:

```text
clip embedding → anomaly score
```

This is useful when only normal data is available.

Possible methods:

- One-Class SVM
- Isolation Forest
- Gaussian Mixture Model
- Mahalanobis distance
- kNN distance from normal embedding bank

This differs from the current heuristic change detection.

Current MVP:

```text
Large difference between adjacent clips → possible change
```

Future anomaly detection:

```text
Distance from learned normal distribution → possible abnormal state
```

---

## 5. Training Data Design

### 5.1 Annotation Unit

The recommended annotation unit is a short video interval:

```text
video_path, start_sec, end_sec, risk_label, event_type
```

Example:

```csv
video_path,start_sec,end_sec,risk_label,event_type
data/train/warehouse_001.mp4,0,2,low,normal
data/train/warehouse_001.mp4,2,4,medium,sudden_motion
data/train/warehouse_001.mp4,4,6,high,near_collision
```

This is easier than frame-level annotation and sufficient for the first version of downstream probes.

---

### 5.2 Label Design

Recommended initial risk labels:

```text
low
medium
high
```

Recommended initial event labels:

```text
normal
sudden_motion
approach
crossing
occlusion
object_fall
near_collision
unknown
```

The label set should be small at first. Too many labels will make the initial dataset sparse and unstable.

---

### 5.3 Human-in-the-Loop Data Creation

The current heuristic system can reduce annotation cost.

Recommended workflow:

```text
1. Run the current V-JEPA change detector on raw videos.
2. Extract high-change candidate intervals.
3. Human reviewer labels only the candidate intervals.
4. Train downstream probes.
5. Use probe predictions to propose new labels.
6. Human reviewer corrects predictions.
7. Add corrected samples to the training set.
8. Retrain probes periodically.
```

This turns the current MVP into a data collection and bootstrapping tool.

---

## 6. Feature Dataset Generation

A future dataset generation script should create the following files:

```text
data/probe_features/
  X.npy
  y_risk.npy
  y_event.npy
  meta.jsonl
```

Where:

```text
X.npy:
  shape = [num_samples, feature_dim]

y_risk.npy:
  clip-level or interval-level risk labels

y_event.npy:
  event-type labels

meta.jsonl:
  video path, time interval, warnings, source metadata
```

The feature extraction pipeline should use the existing V-JEPA adapter:

```text
frames_rgb
  ↓
model.encode_clips(frames_rgb, clip_size)
  ↓
clip embeddings
  ↓
average or temporal pooling within annotated interval
  ↓
training sample
```

---

## 7. Probe Training Pipeline

Recommended initial training pipeline:

```text
X.npy
  ↓
train / validation split
  ↓
StandardScaler
  ↓
LogisticRegression / LinearSVM / RandomForest
  ↓
evaluation
  ↓
save probe model
```

Example saved artifacts:

```text
probes/
  risk_probe.joblib
  event_probe.joblib
  temporal_risk_probe.joblib
```

Minimum evaluation metrics:

```text
accuracy
macro F1
class-wise precision / recall
confusion matrix
calibration curve if probability is used
```

For imbalanced data, macro F1 and class-wise recall are more important than raw accuracy.

---

## 8. Inference Pipeline with Probes

At inference time, the analyzer should combine heuristic and supervised outputs:

```text
Input video
  ↓
V-JEPA embeddings
  ↓
heuristic change score
  ↓
risk_probe prediction
  ↓
event_probe prediction
  ↓
score fusion
  ↓
explanation generation
```

Example fusion:

```text
final_risk_score =
  0.4 × representation_change_score
+ 0.6 × supervised_high_risk_probability
```

The weights should be configurable.

During the early stage, heuristic scores should not be removed. They act as a fallback when the probe is uncertain or out-of-distribution.

---

## 9. Explanation Design

The explanation layer should use both model outputs and domain rules.

Example input to the explanation layer:

```json
{
  "clip_index": 12,
  "time_range": [24.0, 26.0],
  "change_score": 0.78,
  "risk_label": "high",
  "risk_confidence": 0.83,
  "event_type": "near_collision",
  "event_confidence": 0.74,
  "domain_mode": "warehouse"
}
```

Example explanation:

```text
This segment is classified as high risk. The event-type probe suggests a near-collision pattern, and the representation shift score is also high. In a warehouse context, this indicates that the operator should review movement paths, separation rules, or blind-spot conditions around this interval.
```

The explanation should distinguish between:

```text
model prediction
heuristic signal
domain-specific interpretation
recommended operational action
```

This avoids overstating what V-JEPA itself understands.

---

## 10. Recommended Implementation Modules

Suggested future files:

```text
backend/
  probe_dataset.py
    Build feature datasets from labeled video intervals.

  probe_train.py
    Train risk, event, and temporal probes.

  probe_model.py
    Define reusable sklearn or PyTorch probe classes.

  probe_inference.py
    Load trained probes and run prediction on clip embeddings.

  probe_registry.py
    Manage available probes and metadata.

  explanation_builder.py
    Convert model outputs into domain-aware text.
```

The existing files should remain responsible for their current roles:

```text
model_loader.py:
  frozen V-JEPA loading and encoding

analyzer.py:
  orchestration of video analysis

schemas.py:
  API response models

main.py:
  FastAPI endpoints
```

---

## 11. Deployment Considerations

### 11.1 Offline-First Model Loading

The frozen V-JEPA model and trained probes should be mounted locally:

```text
/models/
  vjepa2_1_vitb_dist_vitG_384.pt
  vjepa2_repo/
    hubconf.py

/probes/
  risk_probe.joblib
  event_probe.joblib
```

The application should work without internet access.

---

### 11.2 Versioning

Each trained probe should have metadata:

```json
{
  "probe_name": "risk_probe",
  "version": "0.1.0",
  "trained_at": "2026-05-12",
  "vjepa_backend": "vjepa2_1_vit_base_384",
  "feature_dim": 768,
  "label_set": ["low", "medium", "high"],
  "training_samples": 1200,
  "macro_f1": 0.71
}
```

This metadata should be included in audit logs and analysis outputs.

---

### 11.3 Failure Modes

The system should explicitly handle:

```text
V-JEPA model unavailable
probe file unavailable
feature dimension mismatch
unknown label
low confidence prediction
out-of-distribution embedding
```

When a probe fails, the system should fall back to the current heuristic change detection.

---

## 12. Roadmap

### Phase 1: Data Preparation

- Define label taxonomy.
- Create annotation manifest format.
- Use current MVP to extract candidate intervals.
- Label a small dataset.

### Phase 2: First Probe

- Train clip-level risk classifier.
- Use Logistic Regression as the first linear probe.
- Add prediction output to API response.

### Phase 3: Event Probe

- Train event-type classifier.
- Add event-based explanation templates.
- Improve frontend display.

### Phase 4: Temporal Probe

- Add neighboring clip embeddings.
- Train near-future risk classifier.
- Introduce temporal context into risk scoring.

### Phase 5: Productionization

- Add probe registry.
- Add model/probe versioning.
- Add monitoring and active learning loop.
- Add audit logs for prediction and explanation.

---

## 13. Summary

The future supervised design should not fine-tune V-JEPA first.

The recommended path is:

```text
Frozen V-JEPA encoder
  ↓
Clip embeddings
  ↓
Lightweight downstream probes
  ↓
Risk / event prediction
  ↓
Domain-aware explanation
```

This design is practical, modular, and suitable for limited labeled data.

It also preserves the current MVP as a useful bootstrapping system: the current change detector can identify candidate segments, while the future supervised probes can gradually learn domain-specific meanings from labeled examples.
