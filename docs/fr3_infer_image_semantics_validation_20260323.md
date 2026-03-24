# FR3 Inference Image Semantics Validation (2026-03-23)

## Scope

This note records the final conclusion for the FR3 real-robot inference image observation path and the `left/right` camera semantics validation done on 2026-03-23.

## Final Conclusion

The current inference image mapping is considered correct:

- `left -> observation.images.left`
- `right -> observation.images.right`

Manual visual verification confirmed that the `normal` mapping matches the training-time left/right semantics, while the swapped mapping does not.

## What Was Verified

The validation script used was:

- `tools/fr3/fr3_validate_infer_image_semantics.py`

It was used to export:

- live `left/right` images
- dataset episode-start reference images
- `normal vs swapped` similarity summaries
- preview summary JSON

The machine-computed similarity comparison also favored the `normal` mapping on the sampled nearby episode starts.

## Important Clarification About Color Channels

A temporary bug existed in the validation script output path:

- live PNG export used `cv2.imwrite` directly on RGB arrays
- grayscale similarity conversion also used the wrong OpenCV color conversion constant

That bug affected the saved PNG appearance and slightly affected the similarity script's grayscale metric, but it did **not** mean that the actual inference input to the policy was wrong.

The inference path itself remained correct because:

- OpenCV camera config defaults to RGB output
- the camera backend converts OpenCV BGR frames to RGB before returning them
- `FrankaResearch3.get_observation()` forwards those camera frames without swapping channels again
- policy preprocessing only normalizes and permutes HWC to CHW, without RGB/BGR swapping

The validation script has now been fixed so future exported PNGs are consistent with RGB input.

## Current Status

Closed for now:

- `left/right` image semantics for inference are considered correct
- the earlier live PNG color issue was isolated to the validation script export path
- inference image tensors themselves are not considered affected by that export bug

Still open:

- tactile preview remains blocked by the known DAS tactile decoder mismatch for the current `448-byte` hardware payload

## Practical Outcome

For current FR3 real-robot inference work:

- keep the existing `left/right` camera mapping
- do not swap the two cameras in the runtime
- treat the image-observation path as validated
- continue debugging preview/runtime failures under the tactile track, not the image track
