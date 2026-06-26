# Proposal: Feature/Model Monitoring + Drift-Triggered Retraining for yolov8-face

Extends the YOLOv8 face-detection + CLIP similarity-search tutorial with the FSTORE-2048
embedding-monitoring capability and an automated retraining loop. All changes stay inside the
`yolov8-face` repo.

## Current tutorial (baseline)

- **Feature groups**
  - `wider_face_files` v1 — offline, PK `file_path`, event_time `file_timestamp`. Columns:
    `file_path`, `file_timestamp`, `file_size_mb`, `num_bboxes`, `bboxes` (JSON), `label`, `ingested_at`.
  - `image_embeddings` v1 — online, PK `file_path`, `EmbeddingIndex` on `embedding`
    (512-dim CLIP ViT-B/32, L2-normalized, `model=openaiclip_vit_base_patch32`).
    No event_time, no statistics config.
- **Models**: `facerecognition` (YOLOv8n detector, registered by `train.py`, no feature_view),
  `openaiclip_vit_base_patch32` (CLIP embedding model).
- **Jobs** (`run-job.py`): `create_fgs` (runs `create_fgs.py`, env `pandas-training-pipeline`) and
  `train_yolo` (runs `train.py`, env `yolov8`, 1 GPU). `train.py` ends by re-registering
  `facerecognition` — so it already produces a new model version on each run.
- **Deployment**: `similarimages` (KServe) — embeds the request image with CLIP and runs
  `image_embeddings.find_neighbors(embedding, k=3)`.

## Goals

1. **Feature monitoring** on the embedding feature (`image_embeddings.embedding`) and on scalar
   features (`wider_face_files.num_bboxes`, `file_size_mb`).
2. **Model monitoring** that watches the embedding for drift and **triggers retraining** of the
   `facerecognition` detector via the existing `train_yolo` job after N consecutive shifts.
3. **Retraining job**: make `train.py` / `train_yolo` usable as the FSTORE-2048
   `model_retraining_job`, with minimal change since the job already exists.

## Rationale: what drift means here, and what gets retrained

The `embedding` column holds CLIP image embeddings of the ingested images. Drift in their
distribution (centroid shift, norm-distribution change) is a proxy for **the production image
population moving away from what the detector was trained on** (new scenes, lighting, demographics,
camera sources). That is the early-warning signal to **re-fine-tune the YOLOv8 `facerecognition`
detector** on freshly collected/annotated data — which is exactly what the `train_yolo` job does.

Note the embedding model itself (CLIP) is fixed and not retrained here; the embedding is used as a
cheap, model-agnostic drift sensor over the image stream. (An alternative target — rebuilding the
similarity index by re-embedding — is called out under Open decisions.)

## No schema change needed for time windows (commit-time travel)

Feature-group monitoring resolves detection/reference windows via **time-travel on commit time**,
not the user `event_time`:

```python
# MonitoringWindowConfigEngine._fetch_feature_group_data (no model filter)
pre_df.as_of(exclude_until=start_time, wallclock_time=end_time).read()
```

`as_of(...)` reads HUDI/Delta commit timestamps — when rows were committed to the FG. Every
cached/offline FG (including `image_embeddings`, which keeps an offline HUDI table even though its
online store is the vector DB) has this automatically, so rolling-time windows work with no schema
change. A user `event_time` is used for point-in-time joins and training-data generation, not for
monitoring-window selection. Commit-time windowing is also ideal for the demo: a freshly inserted
batch of drifted embeddings lands in the recent detection window by its commit time.

---

## Part A — Feature monitoring (drift visibility, no model needed)

FG-level `create_feature_monitoring(...)`. Embedding features support `centroid_distance` (scalar,
L2 between detection/reference centroids) and distribution metrics over the embedding **norm**
(PSI/KL/JS/Hellinger; PSI default threshold 0.2). Scalar features use mean/PSI as usual.

```python
emb_fg = fs.get_feature_group("image_embeddings", version=1)

# Embedding centroid drift: recent embeddings vs a trailing baseline window
emb_fg.create_feature_monitoring(
    name="embedding_centroid_drift",
    feature_name="embedding",
).with_detection_window(time_offset="1d") \
 .with_reference_window(time_offset="30d", window_length="7d") \
 .compare_on(metric="centroid_distance", threshold=0.1).save()

# Embedding norm-distribution drift (PSI over per-row L2 norm)
emb_fg.create_feature_monitoring(
    name="embedding_norm_psi",
    feature_name="embedding",
).with_detection_window(time_offset="1d") \
 .with_reference_window(time_offset="30d", window_length="7d") \
 .compare_on_distribution(metric="PSI", threshold=0.2).save()

# Scalar example: faces-per-image count drift on the offline FG
files_fg = fs.get_feature_group("wider_face_files", version=1)
files_fg.create_feature_monitoring(
    name="num_bboxes_drift",
    feature_name="num_bboxes",
).with_detection_window(time_offset="1d") \
 .with_reference_window(time_offset="30d", window_length="7d") \
 .compare_on_distribution(metric="PSI", threshold=0.2).save()
```

These run on a schedule (default daily cron) and surface shifts in the Hopsworks UI. No model or
feature view required.

---

## Part B — Model monitoring + drift-triggered retraining

`create_model_monitoring(...)` is **inference-log monitoring**, not batch-FG monitoring. Its
docstring: "Enable feature monitoring on the inference logs of a specific deployed model. Targets
the logging feature group (`{fv_name}_{version}_log`) and filters by `model_name`/`model_version`."
It raises if the feature view does not have logging enabled. So the detection window reads what the
**deployed model logged at inference time** via `fv.log(...)`, filtered to that model.

Key consequence for this tutorial: the **served** model (the `similarimages` CLIP deployment) and
the model we want to **retrain** (the YOLO `facerecognition` detector) are different. The retrain
trigger does not require them to be the same — `model_retraining_job` is any job. So the design is:
monitor the **query embeddings logged by the production `similarimages` deployment**, and when they
drift, run the `train_yolo` job.

1. **Create a logging-enabled feature view over the embedding feature** (one-time):
   ```python
   emb_fg = fs.get_feature_group("image_embeddings", version=1)
   image_embeddings_fv = fs.get_or_create_feature_view(
       name="image_embeddings_fv",
       version=1,
       query=emb_fg.select(["embedding"]),
       logging_enabled=True,
   )
   ```

2. **Make the deployed predictor log each query embedding.** In `predict_similar_images.py`
   (the `similarimages` predictor), after computing the query `embedding` and running
   `find_neighbors`, call `feature_view.log(...)` with the embedding and the served model so the
   inference-log FG fills up. This is the same logging pattern used in the fraud_online predictor.

3. **Reference the existing training job as the retrain job** (already registered by `run-job.py`):
   ```python
   job_api = project.get_job_api()
   train_yolo_job = job_api.get_job("train_yolo")
   ```
   Passing the job explicitly means we do not depend on the monitored model having an originating
   job or a stored `program` path.

4. **Create the model-monitoring config** on the logged embedding, tied to the served model, with
   the retrain trigger pointing at `train_yolo`:
   ```python
   image_embeddings_fv.create_model_monitoring(
       name="detector_retrain_on_embedding_drift",
       model_name="openaiclip_vit_base_patch32",  # the deployed model that logs the embeddings
       model_version=1,
       retrain_model_after_num_shifts=3,
       model_retraining_job=train_yolo_job,        # retrains the YOLO detector
       # model_retraining_job_execution_args="retrain",  # optional argv for train.py
   ).with_detection_window(time_offset="1d") \
    .with_reference_training_dataset() \
    .compare_on(feature_name="embedding", metric="centroid_distance", threshold=0.1).save()
   ```
   After 3 consecutive runs detect a centroid shift in the logged query embeddings past threshold,
   Hopsworks runs `train_yolo`, which re-fine-tunes YOLOv8 and registers a new `facerecognition`
   version.

Note `with_reference_training_dataset()` defaults to the monitored model's training-dataset version,
so the served model should have been created with this feature view (or pass a reference window
instead). This is the main extra wiring vs fraud_online, where the served and retrained model were
the same XGBoost model.

If wiring inference logging into the deployment is more than you want for the tutorial, the
fallback is **Part A only** (FG feature monitoring for drift visibility) plus a **scheduled**
`train_yolo` job — but that does not exercise the FSTORE-2048 auto-retrain trigger.

---

## Part C — Retraining job implementation

The `train_yolo` job already runs `train.py` and re-registers `facerecognition`, so the retraining
capability is largely wiring. Proposed concrete changes:

1. **Make `train.py` retrain-aware** (small additions):
   - Accept an optional argv (e.g. `sys.argv[1] == "retrain"`) so the same script serves initial
     training and retraining; log which mode it ran in.
   - Pull the **current** dataset from `wider_face_files` (and any newly ingested images) when
     building `widerface.yaml`, so a retrain picks up fresh data rather than the static snapshot.
   - Keep the final `mr.python.create_model("facerecognition", ...)` so each retrain yields a new
     model version (already present). Optionally attach `feature_view=image_embeddings_fv` for
     provenance.
   - Because it runs in the `yolov8` GPU environment, all YOLO deps are present (no env change).

2. **Ensure the job exists before wiring monitoring**: document running `python run-job.py train`
   (or call the `train(...)` helper) so `train_yolo` is registered, then fetch it with
   `job_api.get_job("train_yolo")` for `model_retraining_job`.

3. **No change needed to `run-job.py`'s mechanism** — it already builds a PYTHON job config and
   creates the job. Optionally add a tiny helper to print the job so the monitoring notebook can
   reference it.

---

## Where the new code goes

- **New notebook `6-monitoring-and-retraining.ipynb`** — Parts A + B: create the logging-enabled
  embeddings FV, fetch the `train_yolo` job, create the feature-monitoring and model-monitoring
  configs, and a drift-simulation cell (below).
- **Edit the `similarimages` predictor (`predict_similar_images.py` in `predictor.ipynb`)** — log
  each query embedding via `feature_view.log(...)` so model monitoring has inference data (Part B).
- **Edit `train.py`** — retrain-aware argv + pull current data (Part C).
- **Optional `README.md`** — document the monitoring/retraining step and the new notebook.

## Drift simulation (for the demo)

To exercise the trigger without waiting for real drift, add a cell that appends a batch of
embeddings drawn from a shifted distribution (e.g. embed a different image set, or perturb existing
vectors) into `image_embeddings` with recent `embedded_at`, so the detection-window centroid moves
past threshold for the required consecutive runs.

## Open decisions (please confirm before implementation)

1. **Retrain target**: re-fine-tune the YOLOv8 `facerecognition` detector via `train_yolo`
   (recommended, uses the existing job), OR instead rebuild the similarity index by re-embedding
   images (different job, would need to be created). The embedding-drift→detector-retrain story is
   the cleaner fit for the FSTORE-2048 capability.
2. **Inference logging for Part B**: wire `feature_view.log(...)` into the `similarimages`
   deployment so model monitoring has inference data (recommended, exercises the auto-retrain
   trigger) vs Part A only + a scheduled `train_yolo` job (simpler, no trigger).
3. **Which model_name to monitor**: the deployed `openaiclip_vit_base_patch32` (the model that
   actually logs the embeddings) is the technically correct choice; confirm vs monitoring under a
   different registered name.
4. **Thresholds / cadence**: `centroid_distance` threshold (0.1 placeholder), PSI threshold (0.2),
   `retrain_model_after_num_shifts` (3), and the monitoring cron.

## Risks / notes

- This is the FSTORE-2048 embedding-profiler + `centroid_distance`/norm-PSI path; the cluster run is
  the first real integration test (same caveat as the fraud_online extension).
- `facerecognition` v1 must exist in the registry before `create_model_monitoring` is called.
- The `train_yolo` job needs a GPU node available when the trigger fires.
- CLIP embeddings are L2-normalized, so centroid_distance lives on the unit sphere; calibrate the
  threshold against an observed no-drift baseline rather than guessing.
