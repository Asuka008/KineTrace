# KineTrace / VLM

This repository belongs to the VLM project. Keep the following workflow across future sessions.

- Develop on the local Mac. Run only small offline CPU checks, such as unit tests, CLI checks and fixtures.
- Use GitHub to synchronize source and configuration. The user pulls the code and runs full tests, training, model inference and benchmarks on the school server.
- Target one visible A100 slice on a 40 GB setup. Read the memory actually visible to PyTorch inside the allocation, and record it with each run. Size jobs from that value.
- Keep videos, model weights, indexes, experiment output and cluster-specific credentials outside Git. Expose data paths, account, partition and GPU resource through command arguments or the submission environment.
- On handoff, distinguish locally verified behavior, implemented server commands and measured server results. Keep the README commands in sync with the CLI.

- Create and update school-server Conda environments under `~/conda-envs/`, alongside the project checkout. Use `~/conda-envs/kinetrace` for this project. Track dependency specifications in Git and keep the installed environment on the server.
- After completing the requested changes and passing the relevant local checks, automatically commit only the task-related files and push to `origin/main`. Exclude `conda-envs/`, data, model weights and secrets from commits. If the remote has advanced, inspect and integrate both sides' changes before pushing. Never force-push. Report the pushed commit's short hash or the specific blocker.
