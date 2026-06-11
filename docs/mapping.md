# Practice-day mapping (no map ahead of time)

If you don't have the track's map but get practice time on it, build the map
during practice with SLAM, then everything downstream is one command.

## On the car
1. **Install SLAM** (once):  `sudo apt install ros-$ROS_DISTRO-slam-toolbox`
2. **Start mapping + live raceline** while you drive:
   ```
   ros2 launch f1tenth_gym_ros slam_mapping_launch.py \
        drive:=true learn:=true base_frame:=base_link
   ```
   - `drive:=true` runs `mapping_driver.py` — a slow (~1.5 m/s) disparity-extender
     follow-the-gap driver that traverses the track safely for a clean map.
     Or omit it and hand-teleop the laps (safest).
   - `learn:=true` runs `track_learner.py` — it watches the live `/map` and
     re-runs the optimizer every few seconds, so the racing line **sharpens lap
     after lap**. Output: `racelines/learned_raceline.csv` (+ `_overlay.png`),
     and the best feasible line so far is promoted to `best_raceline.csv`.
   - `base_frame:=base_link` is the **only** sim→hardware difference in this
     launch (the gym calls it `ego_racecar/base_link`). It flows to both
     slam_toolbox and track_learner so they stay on the same frame.
   - slam_toolbox fuses `/scan` + VESC odom into `/map`; watch it fill in (RViz or
     the dashboard). A couple of clean laps is enough.
3. **Finish** — re-save the map and regenerate the final line in one step:
   ```
   tools/finish_mapping.sh <track_name> <seed_x> <seed_y>
   ```
   → `maps/<track_name>.{yaml,pgm}` + `racelines/best_raceline.csv`
   (seed = any point on the start straight, in map metres. With `learn:=true`
   you may already have a good `best_raceline.csv`; this just locks in the map.)
4. **Race** — switch to localization against the saved map (the F1TENTH stack's
   particle filter / the `map_server` + `lifecycle_manager_localization` already
   in the launch) and run the follower. On hardware the odom topic differs from
   the sim, so point the agent at it:
   ```
   ros2 run f1tenth_gym_ros racing_agent --ros-args -p odom_topic:=/odom
   ```
   (`mapping_driver` and the follower use `/scan` + `/drive`, which match the
   real f1tenth_system, so only `odom_topic` needs overriding.)

So your practice time is ~½ hour mapping, the rest racing and tuning.

## Notes
- This is the **real-car** path — the sim always has a map (it renders lidar from
  one), so SLAM isn't needed there.
- `config/slam_mapping.yaml` is tuned for a small, fast platform (0.05 m grid,
  loop closure on). Bump `resolution` coarser if CPU is tight.
- Tune the line afterward in the dashboard (margin / apex-bias) without re-mapping.
