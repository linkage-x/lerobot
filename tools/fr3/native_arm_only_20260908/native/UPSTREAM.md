# Native source provenance

This directory contains a local replay-development snapshot based on
[panda-py by Jean Elsner](https://github.com/JeanElsner/panda-py), licensed under
Apache-2.0. See the repository's root `LICENSE` and the upstream
[license](https://github.com/JeanElsner/panda-py/blob/main/LICENSE).

The snapshot was copied from the isolated Thor replay bundle on 2026-09-09.
It includes local FR3 limit handling, guarded controller lifecycle and diagnostic
controller additions. It must not be represented as an unmodified upstream
panda-py release. The current native-arm player adds a shared-ownership Python
binding for the existing JointTrajectory controller without changing that
controller's control law.

No compiled libraries or credentials are included. Consult
`docs/replay_handoff/README.md` in the repository root for runtime dependencies
and the scope of verification. Building this code does not authorize robot
motion or establish hardware safety validation.
