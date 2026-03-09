# Policy Runtime Substrate

## Purpose

This note captures the current abstraction for building a reusable policy
execution substrate in this repository.

The goal is not to build a one-off ACT runner. The goal is to define a stable
architecture that can support multiple policies under `src/lerobot/policies`
through shared runtime contracts.

## Core Idea

The system should be understood as a `Policy Runtime Substrate` composed of:

- a canonical robot runtime layer
- a policy adapter layer
- a policy runtime layer
- a training and dataset layer

The policy itself is only one component inside this structure.

## Four Layers

### 1. Robot Runtime Layer

This layer owns the real robot control loop and should not know policy-specific
details.

Responsibilities:

- read robot observations
- send robot actions
- maintain canonical robot observation and action contracts

Example canonical contracts:

- observation:
  - `ee.x`
  - `ee.y`
  - `ee.z`
  - `ee.wx`
  - `ee.wy`
  - `ee.wz`
  - `gripper.pos`
  - camera frames
- action:
  - `enabled`
  - `target_x`
  - `target_y`
  - `target_z`
  - `target_wx`
  - `target_wy`
  - `target_wz`
  - `gripper`

### 2. Policy Adapter Layer

This layer is the only legal coupling point between robot runtime and policy
runtime.

Responsibilities:

- convert canonical robot observation into policy-specific input
- convert policy-specific output into canonical robot action

The robot runtime should not know:

- whether a checkpoint expects rotvec or quaternion
- how state vectors are packed
- how image keys are renamed
- how action vectors are unpacked

That belongs to adapters.

### 3. Policy Runtime Layer

This layer owns policy lifecycle and inference semantics.

Responsibilities:

- resolve checkpoint directory
- load config
- load processors
- instantiate policy
- preprocess inputs
- run inference
- postprocess outputs
- manage action chunks or stepwise inference

This layer should not know real robot details.

### 4. Training and Dataset Layer

This layer defines training-time schema and feature alignment.

Responsibilities:

- define dataset feature names and shapes
- provide normalization statistics
- constrain train and inference compatibility

This layer should not directly control hardware execution.

## Four Protocols

### Protocol 1: Canonical Robot Protocol

```python
class CanonicalRobot:
    def get_observation(self) -> RobotObservation: ...
    def send_action(self, action: RobotAction) -> None: ...
```

### Protocol 2: Observation Adapter Protocol

```python
class ObservationAdapter:
    def to_policy_input(self, obs: RobotObservation) -> dict: ...
```

### Protocol 3: Action Adapter Protocol

```python
class ActionAdapter:
    def to_robot_action(self, policy_action) -> RobotAction: ...
```

### Protocol 4: Policy Runtime Protocol

```python
class PolicyRuntime:
    def reset(self) -> None: ...
    def infer_chunk(self, policy_input: dict): ...
    def infer_step(self, policy_input: dict): ...
```

Any new policy can be integrated cleanly if it can be bound to these four
protocols.

## Policy Capability Abstraction

Policies should not be grouped only by algorithm name. They should also be
grouped by runtime capability.

### Type A: Vector Policies

Policies whose inputs and outputs fit a standard tensor pipeline.

Examples:

- `act`
- `diffusion`
- `vqbet`
- `tdmpc`

These are the easiest to support in a shared runtime.

### Type B: Chunk-Native Policies

Policies that naturally emit action chunks.

Examples:

- `act`
- `diffusion`
- `smolvla`
- `pi0`
- `pi05`

These need queue management but still fit a unified runtime well.

### Type C: Multimodal Policies

Policies that depend on more than simple state and image tensors.

Examples:

- `groot`
- `wall_x`
- `xvla`
- `smolvla`

These often require custom processors, tokenizers, VLM components, or richer
input packing.

### Type D: Non-BC Runtime Policies

Policies or models that do not naturally belong to a standard imitation rollout
runner.

Examples:

- `sac`
- `reward_classifier`
- `sarm`

These may reuse factory and config infrastructure but should not automatically
be treated as drop-in replacements for behavior-cloning inference.

## Capability Matrix

When integrating a policy into the shared runtime, evaluate it along four
dimensions.

### Input Form

- state
- images
- text
- tokens
- history
- extra metadata

### Output Form

- joint
- ee
- delta joint
- delta ee
- chunk
- single-step

### Inference Semantics

- stateless
- internal queue
- external queue
- recurrent memory
- rtc-aware

### Runtime Dependencies

- plain torch only
- tokenizer
- VLM processor
- remote model assets
- custom postprocessor

These dimensions are more stable than algorithm family names.

## Recommended Shared Objects

### `PolicySpec`

Describes what a checkpoint expects.

```python
@dataclass
class PolicySpec:
    policy_type: str
    input_kind: str
    output_kind: str
    orientation_format: str
    uses_images: bool
    uses_text: bool
    chunked: bool
```

### `PolicyCapability`

Describes runtime requirements.

```python
@dataclass
class PolicyCapability:
    supports_predict_chunk: bool
    supports_select_action: bool
    requires_tokenizer: bool
    requires_custom_processor: bool
    requires_history: bool
```

### `RuntimeBinding`

Binds the robot side and policy side.

```python
@dataclass
class RuntimeBinding:
    robot: object
    obs_adapter: object
    act_adapter: object
    policy_runtime: object
```

## What This Architecture Enables

If these boundaries are respected:

- swapping robots should not require changing policy runtime
- swapping policies should not require changing robot runtime
- swapping dataset schema should primarily affect adapters and training config

This is the actual source of extensibility.

## What This Architecture Is Not

This is not:

- an ACT-only inference script
- a guarantee that every policy in `src/lerobot/policies` is automatically
  plug-and-play
- a replacement for all policy-specific runtime needs

It is a substrate that makes multi-policy support structured and predictable.

## Final Statement

The long-term architecture should treat the system as three standard assets:

- `Robot Asset`
- `Policy Asset`
- `Dataset Asset`

Adapters are the only valid place where those assets meet.

As long as that rule is preserved, the repository can evolve toward a reusable
multi-policy training and inference framework without collapsing into
algorithm-specific execution code.
