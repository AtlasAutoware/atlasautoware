---
title: 'AtlasAutoware: An integrated autonomous racing stack for F1TENTH-class vehicles with benchmark-gated component adoption'
tags:
  - Python
  - robotics
  - autonomous racing
  - F1TENTH
  - model predictive control
  - state estimation
authors:
  - name: Eshan Iyer
    orcid: 0009-0009-5178-9113
    affiliation: 1
affiliations:
  - name: Thomas Jefferson High School for Science and Technology, Alexandria, VA, United States
    index: 1
date: 12 June 2026
bibliography: paper.bib
---

# Summary

AtlasAutoware is an integrated software stack for 1/10-scale autonomous
racing (the F1TENTH/RoboRacer vehicle class [@okelly2020f1tenth]). It
provides, in a single pip-installable ROS 2 package, the components a
competitive race car needs: a linear-time-varying model-predictive raceline
tracker with a persistent warm-started quadratic-program formulation built
on OSQP [@stellato2020osqp], a tire-aware Model- and Acceleration-based
Pursuit fallback controller [@becker2023map], minimum-curvature raceline
refinement and friction-limited velocity profiling following
@heilmeier2020mincurv, slip-rejecting velocity estimation from a
three-state IMU/wheel-odometry EKF, lidar motion de-skew,
actuation-latency compensation, IMU-based traction governance,
speed-adaptive emergency braking, and a Frenet-frame overtaking planner
re-implemented from the ForzaETH race stack [@baumann2024forzaeth].

The same control, planning, and safety nodes drive both the F1TENTH gym
simulator and a physical car built from commodity hardware (NVIDIA Jetson,
OAK-D Pro camera, RPLidar, VESC motor controller). A hardware abstraction
layer probes and selects the available actuation backend (I2C PWM bridge
or direct VESC UART) at startup, and neural opponent detection selects
among three interchangeable inference paths (TensorRT FP16, OpenCV CUDA
DNN, or fully on-camera inference on the OAK-D's Myriad X VPU) with a CPU
fallback, so no GPU is required anywhere in the stack.

# Statement of need

The published state of the art for this vehicle class is distributed
across separate repositories with firmware-pinned drivers and GPU-assumed
perception pipelines, while most fielded student and hobbyist cars still
run geometric followers with hand-tuned speed tables. AtlasAutoware closes
that gap for research and education: every algorithm runs unmodified in
simulation and on hardware, hardware specifics are isolated in
configuration, and the real-time path is pure Python/numpy on CPU
(the 76-variable QP solves in well under a millisecond, about 3% of the
50 Hz control budget).

A distinguishing feature for research use is the shared deterministic
closed-loop benchmark harness that ships with the package: a
kinematic-bicycle plant, a competition raceline, and benchmark scripts
(`tools/benchmark_*.py`) against which every candidate component was
evaluated and adopted only on measured gains. The harness doubles as the
regression-test plant for the 52 hardware-free unit and closed-loop tests,
so reported component gains (for example, a 9% lap-time reduction from
raceline refinement and recovery of tracking performance under 100–140 ms
of injected actuator delay) are regenerable with single commands. This
makes the stack useful both as a competitive baseline and as an
instrumented testbed for evaluating new racing components in isolation.

A companion preprint describing the methods and component-wise evaluation
in detail is available on arXiv.

# Acknowledgements

The codebase and this paper were developed with substantial assistance
from Claude (Anthropic), an AI system, under the direction and review of
the author, who takes full responsibility for the content. The stack
builds on the F1TENTH gym simulator and re-implements methods from the
ForzaETH, MAP, and TUM minimum-curvature publications cited above.

# References
