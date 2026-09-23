# Distilling a goal-conditioned driving policy for the car

Goal: a policy small enough for the Jetson that follows a language goal ("turn left, then
go straight to the end and stop"), trained from the simulator's goal-conditioned expert,
with Qwen-Drive-1.0 (the Sept 2026 driving VLM) as the teacher for scene/goal understanding.

```
sim (tools/sim_generate.py)  ->  LeRobot dataset  ->  ml/extract_teacher.py (Qwen-Drive, GPU)
                                                  ->  ml/train_student.py  (BC + distillation, GPU)
                                                  ->  runs/student/student.onnx  ->  the car
```

Why split it this way: the expert's actions are exact for this 1/10 embodiment, while
Qwen-Drive's planner is calibrated to full-size road driving. So the student imitates the
expert's (steer, speed) and, through a feature-matching loss, the teacher's visual
representation. If the teacher cannot be loaded (custom architecture), the student trains
BC-only and everything still works.

## On the TJ cluster (Slurm)

```
git clone https://github.com/AtlasAutoware/atlasautoware.git ~/atlasautoware
cd ~/atlasautoware && bash ml/slurm/pipeline.sh      # submits env -> data -> teacher -> student
squeue -u $USER ; tail -f slurm_logs/*.out
```

Jobs (all on the gpu partition: its nodes are python 3.12; the compute nodes are 3.14 with no CUDA torch wheels; unicron = 6x Quadro RTX 8000 48 GB): `01_env` (venv on NFS), `02_data` (regenerates the sim dataset, ~1 ep/s,
600 episodes default), `03_teacher` (gpu partition, 2x Turing, downloads 13.8 GB once),
`04_student` (1 GPU, ~40 epochs, exports ONNX).

## Student

Inputs: front frame 96x128, lidar BEV 96x96, state (vx, wz, gyro), hashed bag-of-words
instruction. ~1 M parameters. Outputs (steer rad, speed m/s). Validation reports MAE on
held-out episodes. `student.onnx` inputs: front[1,3,96,128], bev[1,1,96,96], state[1,5],
ids[1,24] (from `text_ids()` in train_student.py).
