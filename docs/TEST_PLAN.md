# Test Plan — VLM-Guided Pick-and-Place

This test plan validates the CLIP-guided object selection and pick-and-place
behavior implemented in `panda_vision_simulation.py` and exercised through
`color_sorting_vlm.py` (interactive text-prompt mode, option `2`).

All test cases use objects and zones that actually exist in the simulator
(see `panda_vision_simulation.py::create_multiple_objects` and
`::sorting_zones`):

- Objects: `red_cube`, `red_sphere`, `green_sphere`, `yellow_cylinder`, `blue_cube`
- Zones: `red_zone`, `blue_zone`, `green_zone`, `yellow_zone`

There is no purple object and no purple zone in the simulator, which is why
T5 below is a true "unavailable object" negative test rather than a renamed
existing object.

## How to run each test

1. Start the interactive demo: `python3 color_sorting_vlm.py`
2. Choose mode `2` (Interactive Text Prompts).
3. Enter the input prompt for the test case below.
4. Observe CLIP similarity scores printed to the console and the object the
   robot picks.
5. When prompted for a placement zone, choose the zone matching the object's
   color (1=red, 2=blue, 3=green, 4=yellow, 5=throw away).
6. Record the outcome in `evaluation/trials_template.csv` (or a copy of it).

---

### T1 — Pick red cube

- **Input prompt:** `"red cube"`
- **Expected object:** `red_cube`
- **Expected placement zone:** `red_zone`
- **Expected behavior:** CLIP similarity score for `red_cube` is the highest
  among all detected objects; robot picks it up and (after zone choice `1`)
  places it in the red zone without collision.
- **Verification method:** Read the printed similarity scores table and the
  `Selected: ...` line; visually confirm the object lands inside the red
  zone borders via the camera image / GUI.
- **Pass/fail criteria:** PASS if `selected_object == red_cube` AND the
  gripper successfully grasps and releases it inside `red_zone`. FAIL if a
  different object is selected, the grasp fails (object not lifted / falls
  during transport), or it lands outside the zone borders.

### T2 — Pick blue cube

- **Input prompt:** `"blue cube"`
- **Expected object:** `blue_cube`
- **Expected placement zone:** `blue_zone`
- **Expected behavior:** Same as T1 but for the blue cube and blue zone.
- **Verification method:** Same as T1.
- **Pass/fail criteria:** Same as T1, substituting `blue_cube` / `blue_zone`.

### T3 — Pick green sphere

- **Input prompt:** `"green sphere"`
- **Expected object:** `green_sphere`
- **Expected placement zone:** `green_zone`
- **Expected behavior:** Same pattern; also validates that the shape
  descriptor ("sphere") does not get confused with `red_sphere` by CLIP.
- **Verification method:** Confirm `green_sphere` (not `red_sphere`) has the
  highest similarity score before pick.
- **Pass/fail criteria:** PASS only if `green_sphere` specifically is
  selected (selecting `red_sphere` is a FAIL, since it tests color
  disambiguation).

### T4 — Pick yellow cylinder

- **Input prompt:** `"yellow cylinder"`
- **Expected object:** `yellow_cylinder`
- **Expected placement zone:** `yellow_zone`
- **Expected behavior:** Same pattern as above.
- **Verification method:** Same as T1.
- **Pass/fail criteria:** Same as T1, substituting `yellow_cylinder` /
  `yellow_zone`.

### T5 — Request unavailable purple cube

- **Input prompt:** `"purple cube"`
- **Expected object:** none (no purple object exists in the scene)
- **Expected placement zone:** none (no pick/place should occur)
- **Expected behavior:** CLIP will still return a highest-scoring object
  among the 5 available objects (it has no "none of these" option), so the
  system does not refuse automatically. The expected *correct* behavior for
  this test is for the operator to recognize via the printed similarity
  scores that no object is a confident color match (all scores should be
  low/similar relative to T1-T4) and decline to proceed by choosing `n` at
  the "What to do with ... ? (y/n)" confirmation prompt.
- **Verification method:** Inspect the printed similarity score distribution
  — a correct run shows no object with a dominant score the way T1-T4 do.
  Record whichever object CLIP ranks first as `selected_object` in the CSV
  for reference, but mark `selection_correct=False` unless the operator
  correctly withheld the pick (i.e., no object was actually picked).
- **Pass/fail criteria:** PASS if no pick/place is executed (operator
  answers `n`, or scores are logged showing clear ambiguity). FAIL if the
  system proceeds to pick and place some object as if it confidently
  matched "purple cube" — this documents a known limitation (see
  `docs/TECHNICAL_ANALYSIS.md`), not a bug to silently work around.

---

## Notes on reproducibility

- CLIP similarity scores are deterministic for a fixed scene and model
  weights (no sampling/temperature involved), so repeated runs of the same
  prompt against an unchanged scene should produce the same ranking.
  Minor floating point differences may occur across hardware (CPU vs GPU).
- Object initial positions are hardcoded in `create_multiple_objects`, so the
  scene layout is identical across runs unless objects have shifted from a
  previous physics step in the same session.
