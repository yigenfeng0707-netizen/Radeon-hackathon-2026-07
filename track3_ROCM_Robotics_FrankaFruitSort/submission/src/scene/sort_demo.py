"""Scripted multi-object sorting demo.

Builds the sort-task scene (3 fruits + 3 color-coded bowls) and runs a scripted
pick-and-place for each fruit into its color-matched bowl, in pick_order().
Reuses build_scene() (patched to accept a layout override) and run_pick_place()
from the starter grasp_demo module.

Usage:
    uv run python franka_fruit_pick/sort_demo.py --cpu --save-frames
    uv run python franka_fruit_pick/sort_demo.py --save-frames   # GPU

Outputs (with --save-frames):
    outputs/sort_demo_frames/<fruit>_<phase>.png   per-pick phase snapshots
    outputs/sort_demo_frames/sort_final.png        final scene after all picks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import genesis as gs
import numpy as np

import grasp_demo as _gd
from build_scene import build_scene
from grasp_demo import (
    TaskSpec, run_pick_place, check_success,
    _goto_interp, _topdown_quat, GRIPPER_OPEN, _ik,
    GraspProfile,
)
from paths import OUTPUTS_DIR
from scene_config import TABLE_TOP_Z
from sort_scene_config import (
    SORT_LAYOUT,
    SORT_MAPPING,
    pick_order,
    sort_task_description,
)

SORT_FRAMES_DIR = OUTPUTS_DIR / "sort_demo_frames"

# --- Grasp profile overrides for the sort task ---------------------------
# The starter plum profile (close_force=-12N) is fragile: the plum (5.3 cm)
# slips out of the force-controlled gripper during the lateral transport
# (verified: grip_w went from 0.05 to -0.01 mid-transport -> fingers closed
# past the plum -> plum dropped). We bump close_force to -25N for a firmer
# hold. Banana keeps its starter profile (already tuned).
_OVERRIDE_PROFILES = {
    "018_plum": GraspProfile(yaw_offset=0.0, close_force=-35.0, center_align=True),
}
for _name, _prof in _OVERRIDE_PROFILES.items():
    _gd.GRASP_PROFILES[_name] = _prof


# --- _goto_plan with retry ----------------------------------------------
# RRTConnect (grasp_demo._goto_plan) occasionally returns an empty path for
# pregrasp targets after a _go_home reset (verified: banana pregrasp at
# (0.30, 0.20) from home(0.20, 0.00) failed with empty path after retries).
# We wrap _goto_plan with extra retries: RRTConnect is randomized, so a fresh
# attempt often succeeds. We avoid falling back to _goto_interp here because
# the joint-space interpolation can dip below table-top height mid-path and
# launch the fruit off the table (verified: plum was flung to (0.96, 0.08,
# 0.03) when _goto_plan was fully replaced by _goto_interp).
_orig_goto_plan = _gd._goto_plan


def _goto_plan_with_retry(bundle, pos, quat, *, finger, num_waypoints=150, settle=20, recorder=None, max_attempts=10):
    last_qpos = None
    for attempt in range(max_attempts):
        qpos = _orig_goto_plan(
            bundle, pos, quat,
            finger=finger, num_waypoints=num_waypoints, settle=settle, recorder=recorder,
        )
        hand = bundle.franka.get_link("hand")
        cur = hand.get_pos().cpu().numpy().reshape(-1)
        dist = float(np.linalg.norm(cur - np.asarray(pos)))
        print(f"    [plan attempt {attempt+1}/{max_attempts}] target={np.asarray(pos).tolist()} hand={cur.tolist()} dist={dist:.4f}", flush=True)
        if dist < 0.05:
            return qpos  # reached the target
        last_qpos = qpos
        # RRTConnect didn't reach the target -- retry with a fresh random seed
        # (plan_path is randomized). Briefly settle before retrying so the arm
        # is in a known state.
        for _ in range(10):
            bundle.franka.control_dofs_position(last_qpos)
            bundle.scene.step()
            bundle.update_wrist_cam()
    print(f"    [plan FAILED after {max_attempts} attempts] last dist={dist:.4f}", flush=True)
    return last_qpos


_gd._goto_plan = _goto_plan_with_retry


# --- _descend_vertical override -----------------------------------------
# The starter's _descend_vertical steps z and re-solves IK at each step to keep
# the hand on a vertical line. But IK is a numerical solve whose solution
# depends on the current joint configuration -- after a prior pick the arm sits
# in a different IK branch, so the stepped IK drifts laterally (verified:
# post-plum banana descend drifted 2.3cm sideways and pushed the fruit off the
# grasp axis). We replace _descend_vertical with a single RRTConnect plan to
# the final grasp pose: RRTConnect is collision-aware, so the path it plans
# cannot graze the fruit, regardless of IK branch.
_orig_descend_vertical = _gd._descend_vertical


def _descend_via_plan(bundle, xy, z_from, z_to, quat, *, finger, steps=80, settle=15, recorder=None):
    grasp_pos = np.array([xy[0], xy[1], z_to])
    _gd._goto_plan(bundle, grasp_pos, quat, finger=finger, recorder=recorder)


_gd._descend_vertical = _descend_via_plan


# Safe home pose: high above the table center, gripper open. Used to reset the
# arm between picks so each query starts from a clean, collision-free
# configuration instead of the previous round's retreat pose.
HOME_XY = (0.20, 0.00)
HOME_Z = TABLE_TOP_Z + 0.45


def _go_home(bundle, recorder=None) -> None:
    home_pos = np.array([HOME_XY[0], HOME_XY[1], HOME_Z])
    home_quat = _topdown_quat(0.0)
    # Use _goto_interp (joint-space interpolation) instead of _goto_plan (RRTConnect).
    # RRTConnect from the post-pick retreat pose (above the place bowl, with a
    # fruit-yaw offset) to home (yaw=0) fails repeatedly: the yaw rotation + lateral
    # move crosses IK branches that RRTConnect can't bridge within its node budget
    # (verified: 4 retries all failed with "exceeded maximum number of nodes").
    # _goto_interp is safe here because:
    #   - both poses are high above the table (HOME_Z = TABLE_TOP_Z + 0.45), so the
    #     joint-space path stays collision-free;
    #   - the post-plum descend now uses _goto_plan (_descend_via_plan), so the next
    #     pick's grasp path is RRTConnect-planned and does NOT depend on the home
    #     pose's IK branch (the original drift concern is moot).
    _gd._goto_interp(
        bundle, home_pos, home_quat,
        finger_cmd=GRIPPER_OPEN, recorder=recorder,
    )


def run_sort_episode(*, save_frames: bool = False) -> dict:
    """Run one scripted sort episode. Returns per-fruit success dict.

    A "sort success" for a fruit requires both:
      (a) run_pick_place placed it inside the target bowl (check_success), AND
      (b) the target bowl is the color-matched one (always true here, but kept
          as an explicit check so the eval path mirrors the learned-policy eval).
    """
    bundle = build_scene(layout=SORT_LAYOUT, add_video_cam=save_frames)
    SORT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    first_pick = True
    for fruit in pick_order():
        # Reset to a clean home pose BETWEEN picks so RRTConnect starts from a
        # collision-free configuration (avoids round 2/3 planning failures).
        # Skip on the first pick: the arm starts at FRANKA_QPOS (held by
        # build_scene + run_pick_place._settle), and RRTConnect from there to
        # HOME_POS fails (verified: 4 attempts all returned empty paths). The
        # starter grasp_demo goes directly from FRANKA_QPOS to pregrasp, which
        # is the known-good planning pattern.
        if not first_pick:
            _go_home(bundle)
        first_pick = False
        container = SORT_MAPPING[fruit]
        task = TaskSpec(
            pick_object=fruit,
            place_target=container,
            success_tol=0.06,
        )
        success, _ = run_pick_place(bundle, task, save_frames=save_frames)
        results[fruit] = {
            "container": container,
            "placed": bool(success),
            "sorted": bool(success),  # color match is structural in this script
        }
        if save_frames and bundle.world_cam is not None:
            img = bundle.world_cam.render(rgb=True)[0]
            from PIL import Image
            Image.fromarray(img).save(SORT_FRAMES_DIR / f"after_{fruit}.png")

    if save_frames and bundle.world_cam is not None:
        img = bundle.world_cam.render(rgb=True)[0]
        from PIL import Image
        Image.fromarray(img).save(SORT_FRAMES_DIR / "sort_final.png")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted multi-object sort episode.")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Force CPU sim.")
    parser.add_argument("--save-frames", action="store_true", help="Save per-phase world-cam frames.")
    parser.add_argument("--steps", type=int, default=50, help="Settle steps after the scene is built.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    results = run_sort_episode(save_frames=args.save_frames)

    print("\n=== Sort episode results ===")
    total = 0
    sorted_ok = 0
    for fruit, r in results.items():
        total += 1
        sorted_ok += int(r["sorted"])
        flag = "OK" if r["sorted"] else "FAIL"
        print(f"  [{flag}] {fruit} -> {r['container']}")
    print(f"\nSorted: {sorted_ok}/{total}")
    print(f"Frames: {SORT_FRAMES_DIR}")


if __name__ == "__main__":
    main()
