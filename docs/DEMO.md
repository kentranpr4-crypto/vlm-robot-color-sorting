# 90-Second Demo Script

Target entry point: `python3 color_sorting_vlm.py`, mode `2` (Interactive
Text Prompts).

## Timeline

### 0–15s: Introduce robot and environment
- Show the PyBullet GUI window with the Franka Panda arm, table, 5 objects
  (`red_cube`, `red_sphere`, `green_sphere`, `yellow_cylinder`, `blue_cube`),
  and the 4 colored sorting zones already visible on screen.
- Say: "This is a Franka Panda arm in a PyBullet physics simulation. It uses
  a pretrained CLIP model to pick objects based on natural-language
  instructions."

### 15–30s: Show user language instruction
- In the terminal, at the `🤖 Enter your prompt:` input, type a prompt such
  as `red cube`.
- Say the prompt out loud as you type it so viewers connect speech → text →
  system input.

### 30–45s: Show CLIP object selection
- Let the console print the segmentation capture steps and the
  "📊 Individual Crop Similarity Scores" table.
- Point at the score bars — highlight that the chosen object has a visibly
  higher score than the others.
- Say: "CLIP compares each object's image crop against the text prompt and
  scores them — the robot picks whichever scores highest."

### 45–75s: Robot picks and places the selected object
- Confirm the pick (`y` at the confirmation prompt).
- Choose the matching zone number (e.g. `1` for red zone) at the
  "Choose destination (1-5):" prompt.
- Let the simulation play out: approach → lower → grasp → lift → move to
  zone → release → retreat → return home.

### 75–90s: Show actual measurements and conclusions
- Switch to a terminal pane and run:
  ```bash
  python3 evaluation/evaluate.py evaluation/trials_template.csv
  ```
  using a copy of the template that you have **actually filled in** with
  real outcomes from trials run before/during rehearsal (never present the
  blank template's N/A output as if it were a real result).
- Say: "Every trial is logged to this CSV and scored automatically — this
  gives us reproducible accuracy and success-rate numbers instead of just a
  vibe that 'it worked'."

## Exact demo commands

```bash
# 1. Launch the interactive demo
python3 color_sorting_vlm.py
# -> choose "2" for Interactive Text Prompts

# 2. At the prompt, type one of:
#    red cube
#    blue cube
#    green sphere
#    yellow cylinder

# 3. Confirm pick: y
# 4. Choose placement zone: 1 (red) / 2 (blue) / 3 (green) / 4 (yellow) / 5 (throw away)

# 5. After the session, show metrics from a CSV you have filled in:
python3 evaluation/evaluate.py path/to/your_filled_trials.csv
```

## Expected input prompts

Use prompts that match the five real objects in the scene:
- `"red cube"` → `red_cube`
- `"blue cube"` → `blue_cube`
- `"green sphere"` → `green_sphere`
- `"yellow cylinder"` → `yellow_cylinder`
- `"red sphere"` or `"round red thing"` → `red_sphere` (good for showing
  disambiguation against `red_cube` / `green_sphere`)

Avoid prompts for objects that don't exist (e.g. "purple cube") during the
main demo — save that for the T5 test case in `docs/TEST_PLAN.md` if you
want to show a known limitation, not as the headline demo moment.

## How to choose placement zone

The zone should match the picked object's color: red objects → zone `1`
(red), blue → `2`, green → `3`, yellow → `4`. Zone `5` ("throw away") is a
dramatic alternate path (`throw_object_away`) — fun for a demo aside, but
don't use it as the primary success path since it doesn't demonstrate
color-matched sorting.

## What to record

For each trial run during rehearsal or the live demo, fill one row of
`evaluation/trials_template.csv` (or a copy) with:
- the prompt typed,
- the object CLIP selected (`selected_object`) vs. what you intended
  (`expected_object`),
- whether the pick and place each visually succeeded,
- the wall-clock duration of the trial,
- any notes (e.g. "gripper slipped on first attempt").

## Troubleshooting

- **CLIP model download hangs / fails:** `transformers` needs to download
  `openai/clip-vit-base-patch32` weights on first run. Pre-download before
  the demo (run the script once offline-adjacent to the venue, if possible)
  to avoid live network dependency.
- **GUI window doesn't appear / PyBullet connect error:** ensure no other
  PyBullet GUI instance is already running (`p.connect(p.GUI)` fails if a
  previous process didn't call `p.disconnect()`); kill stray Python
  processes and retry.
- **Robot arm moves into the table / objects fly off zones:** this is a
  known limitation of the hand-tuned waypoint heights in
  `safe_move_to_position` — if it happens, retry with a cube/sphere close to
  its original spawn position, and avoid dragging out the interactive
  session long enough for objects to settle in colliding poses.
- **CLIP picks the wrong object in rehearsal:** check the printed similarity
  score table — if scores are close, rephrase the prompt to be more
  specific ("the blue cube" rather than just "cube"). This is expected
  model behavior, not a bug to patch live.
- **Console filled with traceback instead of scores:** the script's
  exception handler in the interactive loop prints the full traceback and
  continues the loop — read the last line of the traceback and retype the
  prompt; a single bad prompt does not require restarting the whole demo.
