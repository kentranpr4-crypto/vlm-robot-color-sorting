# Architecture

This diagram reflects the actual pipeline implemented in
`panda_vision_simulation.py` as driven by `color_sorting_vlm.py`'s
interactive text-prompt mode (`run_interactive_text_prompt_demo` →
`capture_and_process_scene` → `compute_object_similarity` →
`select_best_object` → `pick_object` → `place_object_in_sorting_zone` /
`place_object_in_specific_zone`).

```mermaid
flowchart TD
    A[Language Command<br/><i>text prompt, e.g. 'red cube'</i>] --> B[Simulated Camera<br/><i>capture_camera_image_with_segmentation</i>]
    B --> C[Object Crop / Segmentation<br/><i>get_object_segmentation_masks →<br/>get_precise_bounding_boxes_from_masks →<br/>crop_image_with_masks</i>]
    C --> D[CLIP Vision-Language Matching<br/><i>compute_object_similarity<br/>(zero-shot, pretrained CLIP)</i>]
    D --> E[Selected Object<br/><i>select_best_object (argmax)</i>]
    E --> F[Inverse Kinematics<br/><i>p.calculateInverseKinematics<br/>end-effector link 11</i>]
    F --> G[Franka Panda Controller<br/><i>p.setJointMotorControl2<br/>POSITION_CONTROL, 7 joints + gripper</i>]
    G --> H[Pick-and-Place Execution<br/><i>pick_object → place_object_in_sorting_zone<br/>/ place_object_in_specific_zone / throw_object_away</i>]
    H --> I[Verification<br/><i>operator / evaluation/evaluate.py<br/>against trials CSV</i>]
```

## Notes on fidelity to source code

- **Simulated Camera**: `capture_camera_image_with_segmentation` returns RGB,
  depth, and a per-pixel segmentation array from PyBullet's `getCameraImage`
  — segmentation is ground-truth from the simulator, not a learned detector.
- **Object Crop / Segmentation**: the interactive demo path uses the
  mask-based precise cropping (`capture_and_process_scene`), not the looser
  fixed-radius `get_object_bounding_boxes` path (which still exists and is
  used by `run_vision_guided_pick_and_place`, a separate non-interactive
  entry point not wired into the color-sorting CLI menu).
- **Inverse Kinematics**: PyBullet's built-in IK solver against a single
  target pose; `safe_move_to_position` inserts an intermediate waypoint when
  needed but is not shown as a separate box since it's a variant of the same
  IK + joint-control step, not a distinct pipeline stage.
- **Verification**: there is no automatic in-sim success detector (e.g. no
  code checks "is the object actually inside the zone bounds"); verification
  is a human (or test harness) observation recorded into the CSV consumed by
  `evaluation/evaluate.py`.
