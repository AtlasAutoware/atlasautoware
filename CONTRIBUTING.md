# Contributing to AtlasAutoware

Thanks for your interest in the project!

## Reporting issues / getting support
Open a GitHub issue at https://github.com/AtlasAutoware/atlasautoware/issues with:
- what you ran (command, config, branch/commit),
- what you expected vs. what happened,
- relevant logs or tracebacks.

Questions about setup, calibration, or racing configuration are welcome as issues too — tag them `question`.

## Contributing code
1. Fork the repo and create a feature branch.
2. Install dev deps: `pip3 install numpy scipy "osqp<1" pytest matplotlib`.
3. Make your change. New components should be evaluated against the shared closed-loop benchmark harness (`tests/closed_loop.py`, `tools/benchmark_*.py`) — adoption of racing techniques is gated on measured closed-loop gains, and negative results are welcome in PR descriptions.
4. Run the test suite: `python3 -m pytest tests/ -q` (all tests are hardware-free).
5. Open a pull request describing the change and benchmark results where applicable.

## Code style
Pure-Python/numpy, no GPU dependence in the control path; hardware-specific behavior belongs behind the existing abstraction layers (drive node backends, perception `backend` parameter) and in configuration, not in algorithm code.

## Conduct
Be respectful and constructive. This project is maintained by a student; patience with response times is appreciated.
