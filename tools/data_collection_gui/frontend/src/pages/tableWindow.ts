import type { TableWindow } from "../types";

/** The base-frame rectangle a plot is showing, derived from the plot's own axes.
 *
 * Both maps place the camera backdrop by handing this window to the gateway, which returns the
 * still re-projected to cover exactly it. That makes alignment a property of one number: if the
 * window does not describe the same rectangle the points are drawn in, the picture is offset
 * from them everywhere, which is the failure this replaced. So the window is derived from the
 * plot's own inverse mapping rather than recomputed from centres and spans -- there is then no
 * second expression of the axes that can drift from the first.
 */
export type PlotRect = { left: number; top: number; right: number; bottom: number };

export function windowForPlot(
  frame: { toWorldX: (screenX: number) => number; toWorldY: (screenY: number) => number },
  rect: PlotRect
): TableWindow {
  return {
    minX: frame.toWorldX(rect.left),
    maxX: frame.toWorldX(rect.right),
    // Screen y grows downward and base y grows upward, so the bottom edge of the plot is the
    // minimum. Swapping these does not fail loudly: it flips the backdrop and leaves every
    // point where it was.
    minY: frame.toWorldY(rect.bottom),
    maxY: frame.toWorldY(rect.top)
  };
}
