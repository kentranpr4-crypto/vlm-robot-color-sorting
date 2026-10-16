# PyBullet Robotic Arm Simulation with Vision-Language Models

A Python simulation that combines PyBullet robotic arm control with zero-shot image-text matching capabilities using the CLIP vision-language model.

> **Attribution:** This project is based on the original work by
> [Nabil-Miri](https://github.com/Nabil-Miri/vlm-robot-color-sorting), created
> as an exploration into applying vision-language models in robotics. This
> fork ([tuanfptu/vlm-robot-color-sorting](https://github.com/tuanfptu/vlm-robot-color-sorting))
> adds a reproducible evaluation harness, test plan, and documentation on top
> of the original simulation and VLM demo code for a hackathon submission.

## Technology Stack

- **Simulation:** [PyBullet](https://pybullet.org/) (physics engine, Franka Panda URDF)
- **Vision-Language Model:** [CLIP](https://github.com/openai/CLIP) (`openai/clip-vit-base-patch32`) via Hugging Face `transformers`
- **Core libraries:** PyTorch, NumPy, Pillow, Matplotlib
- **Robot:** Franka Emika Panda (7-DOF arm + 2-finger gripper), simulated via `franka_panda/panda.urdf`

<p align="center">
  <img src="https://github.com/user-attachments/assets/8edd7f96-cdcd-4495-b8e8-9ef2164e3d8a" alt="vlm (online-video-cutter" width="1000"/>
</p>

## Overview

This project demonstrates:
- **Physics-based robotic arm simulation** using PyBullet
- **Vision-guided object selection** using CLIP vision-language model
- **Intelligent pick and place operations** based on text prompts

The robot captures images from its camera, analyzes them using CLIP, selects objects based on text descriptions, and performs pick-and-place operations, including color sorting and interactive placement.


## Implementation Videos
1) VLM interaction:


https://github.com/user-attachments/assets/17893bf4-5b80-47f8-9a8a-8a106cedf039



2) Automatic Sorting:


https://github.com/user-attachments/assets/93f907f5-6c96-4eb3-babd-b46292f15680


## Project Structure

```
├── panda_vision_simulation.py             # Vision-guided robot simulation Class
├── color_sorting_vlm.py                   # Color sorting and interactive demo with VLM
├── simple_pick_place_demo.py              # Simple pick and place demo
├── requirements.txt                       # Python dependencies
├── evaluation/
│   ├── evaluate.py                        # Computes metrics from a trial-results CSV
│   └── trials_template.csv                # Blank CSV schema for recording trial outcomes
├── docs/
│   ├── ARCHITECTURE.md                    # Pipeline diagram (Mermaid), matched to source code
│   ├── TEST_PLAN.md                       # T1-T5 test scenarios with pass/fail criteria
│   ├── TECHNICAL_ANALYSIS.md              # Learning / Geometry / Representation / Evidence breakdown
│   └── DEMO.md                            # 90-second demo script and troubleshooting
├── tests/
│   └── test_evaluation.py                 # Unit tests for evaluation/evaluate.py (synthetic fixtures)
└── README.md                              # This documentation
```

## Features

### Vision-Guided Robot Simulation
- **Camera image capture** from PyBullet simulation
- **Object detection** using 3D-to-2D projection and segmentation masks
- **Vision-language matching** with **CLIP** for object selection
- **Text-prompted pick and place** ("pick up the red cube")
- **Multiple objects** (red cube, red sphere, blue cube, green sphere, yellow cylinder)
- **Color sorting zones** with physical borders to prevent objects from falling
- **Interactive zone selection and throw action** for object placement

## Requirements

### Basic Requirements (for robot simulation)
- Python 3.7+
- PyBullet
- NumPy

### Full Requirements (for vision integration)
- All basic requirements plus:
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- Matplotlib
- Requests

## Installation
1. Clone or download this project:
   ```bash
   git clone https://github.com/Nabil-Miri/vlm-robot-color-sorting.git
   ```

2. (Recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Color Sorting and Interactive Demo

To run the color sorting and interactive demo:
```bash
python3 color_sorting_vlm.py
```

When you start the demo, you will be prompted to choose a mode:

1. **Automatic Color Sorting**: The robot will automatically sort all objects into their matching colored zones.
2. **Interactive Text Prompt**: You can enter text prompts to select which object the robot should pick up, and choose where to place it (including a "throw away" option).


Follow the on-screen instructions to interact with the robot and sorting zones.

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline
diagram (Language Command → Simulated Camera → Segmentation → CLIP Matching
→ Selected Object → Inverse Kinematics → Panda Controller → Pick-and-Place →
Verification), with notes on exactly which functions implement each stage.

## Evaluation

Reproducible evaluation lives in `evaluation/`:

```bash
# Fill in a copy of the template with real trial outcomes while running
# color_sorting_vlm.py, then score it:
cp evaluation/trials_template.csv evaluation/my_trials.csv
# ... edit my_trials.csv with real results ...
python3 evaluation/evaluate.py evaluation/my_trials.csv
```

This computes five metrics, each reported with its own denominator so a
missing field is never silently counted as a success or failure:
1. **Object Selection Accuracy** — fraction of trials where the CLIP-selected object matched the expected object.
2. **Pick Success Rate** — fraction of trials where the pick action succeeded.
3. **Place Success Rate** — fraction of trials where the place action succeeded.
4. **End-to-End Task Success Rate** — fraction of trials where selection, pick, and place *all* succeeded.
5. **Average Execution Time** — mean trial duration, in seconds.

Run the unit tests (synthetic fixtures only, never presented as real
experimental results):
```bash
python3 -m unittest tests/test_evaluation.py -v
```

See [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) for the five test scenarios
(T1-T5) used to generate real trial data, and
[`docs/TECHNICAL_ANALYSIS.md`](docs/TECHNICAL_ANALYSIS.md) for what is and
isn't actually implemented (e.g. zero-shot CLIP matching + classical IK is
**not** an end-to-end VLA model; the PyBullet scene is a standalone physics
simulation, **not** a synchronized digital twin of a real robot).

For a 90-second presentation script and troubleshooting tips, see
[`docs/DEMO.md`](docs/DEMO.md).

---





## Technical Details

### CLIP Vision-Language Matching

The robot uses CLIP to match images of objects to your text prompt:

```python
# For each object crop, CLIP returns a similarity score with the text prompt
similarity_scores = model.compute_object_similarity(crops, text_prompt)
selected_object, best_score = model.select_best_object(similarity_scores)
# Robot picks and places the selected object
```

CLIP compares each cropped object image to your prompt (like "red cube") and returns a score for each. The robot picks the object with the highest score and moves it as you choose.

### Inverse Kinematics

The robot uses PyBullet's built-in inverse kinematics solver to calculate joint angles needed to reach target positions:

```python
joint_positions = p.calculateInverseKinematics(
    self.robot_id,
    endEffectorLinkIndex=11,  # Panda end-effector link
    targetPosition=target_position,
    targetOrientation=target_orientation
)
```

### Robot Control

Joint control uses position control mode:
- **Position Control**: Joints move to target positions
- **Gripper Control**: Two-finger gripper with synchronized motion

## Limitations

- **No real-world robot connection.** This is a standalone PyBullet physics
  simulation; it is not a digital twin synchronized to any physical robot or
  physical sensor feed.
- **Not an end-to-end VLA model.** The pipeline composes a pretrained
  vision-language *matching* model (CLIP, zero-shot, no fine-tuning) with a
  classical, non-learned inverse-kinematics solver and hand-coded motion
  primitives — it does not learn vision→action mappings end-to-end.
- **Ground-truth segmentation.** Object crops come from PyBullet's exact
  per-pixel segmentation output, not a learned detector — a real-world
  deployment would need an actual perception stack (e.g. SAM, Mask R-CNN,
  or depth-based segmentation).
- **No automatic success verification.** Nothing in the simulator code
  checks whether a placed object actually landed inside a zone's bounds;
  success/failure must be observed and recorded manually (or by an external
  harness) into the evaluation CSV.
- **Unavailable-object requests are not refused automatically.** CLIP always
  returns a highest-scoring object among whatever is in the scene, even for
  prompts naming objects that don't exist (e.g. "purple cube") — see T5 in
  `docs/TEST_PLAN.md`.
- **Fixed scene layout.** Object positions, zone positions, and prompts are
  tuned to the specific 5 objects and 4 zones hardcoded in
  `panda_vision_simulation.py`; it does not generalize to arbitrary object
  sets without code changes.

## Team Contributions

This hackathon fork was built by a 3-person team in a 3-hour session:

- **Robotic simulation** — PyBullet physics setup, Panda arm control,
  pick-and-place motion primitives.
- **VLM application** — CLIP integration, object selection logic,
  interactive color-sorting demo.
- **Evaluation & documentation** (this branch, `feature/evaluation`) —
  reproducible evaluation harness (`evaluation/`), test plan
  (`docs/TEST_PLAN.md`), technical analysis (`docs/TECHNICAL_ANALYSIS.md`),
  architecture diagram (`docs/ARCHITECTURE.md`), demo script (`docs/DEMO.md`),
  and this README update.

## Future Enhancements

Possible extensions to this project:
- Integrate advanced object segmentation for more accurate identification.
- Detect and localize area zones visually, not just by preset coordinates.
- Enable multi-object reasoning for commands involving relationships (e.g., “Stack all blue cubes, then put the red sphere on top”).

## Resources

- [PyBullet Documentation](https://pybullet.org/)
- [Panda Robot Specifications](https://www.franka.de/technology)
- [Robotics and Physics Simulation Tutorial](https://pybullet.org/wordpress/)

## Contributing

Feel free to fork, modify, and submit PRs! Suggestions and improvements are welcome.
