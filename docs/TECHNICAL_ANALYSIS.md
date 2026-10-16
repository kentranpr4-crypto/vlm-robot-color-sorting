# Technical Analysis

This document maps the actual implementation in this repository to four
robotics-evaluation pillars, and is explicit about what is implemented vs.
what would require additional work to claim.

## Pillar 1 — Learning (Pretrained CLIP image-text matching)

**Implemented.** `panda_vision_simulation.py::_initialize_clip` loads a
pretrained `openai/clip-vit-base-patch32` model via Hugging Face
`transformers` (`CLIPModel`, `CLIPProcessor`). No fine-tuning is performed —
this is zero-shot image-text matching using off-the-shelf CLIP weights.

- `compute_object_similarity` / `_compute_clip_similarity` feed a batch of
  object image crops and a single text prompt into CLIP, take
  `logits_per_text`, and apply `softmax` to get a normalized similarity
  distribution over the detected objects.
- `select_best_object` picks the `argmax` of that distribution.

**Not implemented:** no training loop, no dataset-specific fine-tuning, no
confidence calibration beyond the raw softmax output. The softmax
normalizes scores to sum to 1 across *whatever objects were detected in that
frame* — it is a relative ranking, not an absolute "is this really red cube"
probability. This is why T5 (unavailable object) can still return a
"winning" object with a non-trivial score.

## Pillar 2 — Geometry (3D positions, coordinate frames, inverse kinematics)

**Implemented, with a simplification.**

- 3D object positions come directly from PyBullet's physics state
  (`p.getBasePositionAndOrientation`), not from a perception/pose-estimation
  pipeline.
- `project_3d_to_2d` implements a standard camera projection: world →
  camera-space (view matrix) → clip-space (projection matrix) → perspective
  divide → screen-space pixel coordinates. This is real, textbook
  3D-to-2D projection math, not a stub.
- Inverse kinematics uses PyBullet's built-in solver
  (`p.calculateInverseKinematics`) against the Panda's end-effector link
  (index 11) to convert target Cartesian positions into joint angles for the
  7-DOF arm. `safe_move_to_position` adds a simple waypoint strategy to avoid
  low, table-colliding trajectories — this is a heuristic, not a full motion
  planner (no collision checking against actual geometry, no trajectory
  optimization).

**Caveat:** because 3D object positions are read directly from the physics
engine rather than estimated from the camera image, the "vision" pipeline
does not have to solve 3D localization — only 2D bounding-box placement
(via `get_object_bounding_boxes`, a fixed-radius box around the projected
centroid) or, more precisely, segmentation-mask-derived boxes (see Pillar 4).

## Pillar 4 — Representation (object crops, bounding boxes, poses, joint states)

**Implemented, two tiers of fidelity:**

1. **Approximate (`get_object_bounding_boxes`):** fixed-size (50px) boxes
   centered on the projected 3D centroid of each object. Simple, but boxes
   can be loose or misaligned for non-centroid-symmetric cases.
2. **Precise (`get_object_segmentation_masks` +
   `get_precise_bounding_boxes_from_masks` + `crop_image_with_masks`):**
   uses PyBullet's per-pixel segmentation array (returned alongside the RGB
   image by `getCameraImage`) to get an exact binary mask per object ID,
   derive a tight bounding box from the mask's nonzero pixel extents, and
   produce a background-removed (white-background) crop. This is the path
   actually used by `color_sorting_vlm.py`'s interactive demo via
   `capture_and_process_scene`.
- Joint state representation: `set_initial_robot_pose` and `move_to_position`
  work in terms of the Panda's 7 revolute joint positions plus 2 gripper
  finger joints (indices 9/10), controlled via
  `p.setJointMotorControl2(..., p.POSITION_CONTROL, ...)`.

**Not implemented:** no learned object detector/segmenter — segmentation
comes for free from the simulator's ground-truth render, which would not be
available on a real robot without a real perception stack (e.g. SAM,
Mask R-CNN, or a depth-based segmenter).

## Pillar 7 — Evidence (real test results and evaluation metrics)

**Implemented as tooling, not as pre-filled results.** `evaluation/evaluate.py`
computes five metrics — Object Selection Accuracy, Pick Success Rate, Place
Success Rate, End-to-End Task Success Rate, and Average Execution Time —
from a CSV of trial records (`evaluation/trials_template.csv`). Each metric
reports its own denominator, and missing/unknown fields are never counted
as a success (see `tests/test_evaluation.py` for unit tests enforcing this
on synthetic fixtures).

**Important distinction:** this repository ships the *measurement
instrument*, not fabricated results. `evaluation/trials_template.csv` is
intentionally empty of outcomes — real numbers only exist once someone runs
the test scenarios in `docs/TEST_PLAN.md` against the live simulator and
records the outcomes. Any metrics reported in a README, demo script, or
presentation must come from an actual filled-in CSV produced during a real
run, not from this template or from invented numbers.

## Explicitly unverified / rhetorical-risk claims to avoid

- Do **not** describe the PyBullet simulation as a "synchronized digital
  twin" — there is no real-world robot or sensor feed being mirrored; it is
  a standalone physics simulation.
- Do **not** describe the CLIP-selection + inverse-kinematics pipeline as an
  "end-to-end VLA (Vision-Language-Action) model" — there is no single
  learned model mapping vision+language directly to actions. The pipeline
  is a composition of a pretrained vision-language *matching* model (CLIP)
  with a classical, non-learned IK solver and hand-coded motion primitives.
  This is a legitimate and useful VLM-guided robotics pipeline, but it is
  architecturally different from a VLA model trained end-to-end on
  action trajectories.
