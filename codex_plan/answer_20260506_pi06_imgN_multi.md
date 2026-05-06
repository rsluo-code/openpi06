PI06 imgN validation multi-episode support

Files changed
- `validation_1_hand_pi06_imgN.py`
- `validation_1_hand_pi06_imgN_multi.py`
- `034validation_1_hand_pi06.sh`
- `035validation_1_hand_pi06_multi.sh`

What changed
- Refactored `validation_1_hand_pi06_imgN.py` into:
  - reusable `Args`
  - reusable `_eval_single_episode(client, args, episode_dir)`
  - cleaner output naming and output directory handling
- Added `validation_1_hand_pi06_imgN_multi.py` following the same pattern as `validation_1_hand_valuenet_multi.py`
  - supports:
    - `episode_dirs`
    - `episode_dirs_file`
    - `episode_glob`
    - `prompt_types`
- Reworked `034validation_1_hand_pi06.sh`
  - single-episode entry
  - configurable episode dir / output dir / prompt type / arm side / naming metadata
- Added `035validation_1_hand_pi06_multi.sh`
  - multi-episode entry
  - configurable episode list / episode file / glob / prompt types

Behavior
- `034` now remains the single-episode entry.
- `035` is the multi-episode batch wrapper, analogous to `025`.
- Output png names are either:
  - `name_save` if explicitly provided
  - or auto-generated from:
    - `model_time`
    - `model_step`
    - `model_dim`
    - `prompt_type`
    - arm side
    - episode directory name

Validation
- `python3 -m py_compile validation_1_hand_pi06_imgN.py validation_1_hand_pi06_imgN_multi.py`
- `bash -n 034validation_1_hand_pi06.sh 035validation_1_hand_pi06_multi.sh`
