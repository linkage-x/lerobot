import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import {
  CROP_HANDLES,
  cropCursor,
  cropFromCorners,
  cropHitTest,
  cropStyleBox,
  isFullFrame,
  moveCrop,
  resizeCrop,
  type CropDragMode,
  type CropHandle,
  type CropPoint,
  type CropRect
} from "./cropGeometry";

/** Grip radius in CSS pixels, converted to source pixels against the rendered size. */
const GRIP_CSS_PX = 10;

type Drag = { mode: CropDragMode; origin: CropPoint; startRect: CropRect };

function pct(value: number, total: number): string {
  return `${total > 0 ? (value / total) * 100 : 0}%`;
}

/**
 * Draw the crop on the frame it will be applied to.
 *
 * The numbers alone could not answer the only question being asked -- "does this box hold the
 * workspace?" -- so the box is drawn over a real frame of the recording and dragged there.
 * Coordinates stay in source pixels throughout; the overlay is positioned in percentages so it
 * tracks the image at whatever size the panel renders it.
 */
export function CameraCropPicker({
  frameUrl,
  frameW,
  frameH,
  rect,
  disabled,
  onChange
}: {
  frameUrl: string;
  frameW: number;
  frameH: number;
  rect: CropRect;
  disabled: boolean;
  onChange: (rect: CropRect) => void;
}) {
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [hover, setHover] = useState<CropDragMode>("new");
  // The frame that is on screen, which lags `frameUrl` by one decode. Swapping <img src>
  // directly blanks the tile between frames, and a blank tile under a box being dragged reads
  // as the picker having lost the recording.
  const [shownUrl, setShownUrl] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    if (!frameUrl) {
      setShownUrl("");
      setStatus("idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      setShownUrl(frameUrl);
      setStatus("ready");
    };
    image.onerror = () => {
      if (!cancelled) setStatus("error");
    };
    image.src = frameUrl;
    return () => {
      cancelled = true;
      image.onload = null;
      image.onerror = null;
    };
  }, [frameUrl]);

  const toSourcePoint = useCallback(
    (clientX: number, clientY: number): CropPoint => {
      const box = surfaceRef.current?.getBoundingClientRect();
      if (!box || box.width <= 0 || box.height <= 0) return { x: 0, y: 0 };
      return {
        x: ((clientX - box.left) / box.width) * frameW,
        y: ((clientY - box.top) / box.height) * frameH
      };
    },
    [frameW, frameH]
  );

  const gripTolerance = useCallback((): number => {
    const box = surfaceRef.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return GRIP_CSS_PX;
    return (GRIP_CSS_PX * frameW) / box.width;
  }, [frameW]);

  const hitTest = useCallback(
    (point: CropPoint): CropDragMode => {
      const mode = cropHitTest(rect, point, gripTolerance());
      // A full-frame box has no inside worth grabbing: it cannot move and it cannot grow, so a
      // press in the middle of an untouched frame would do nothing at all. Draw instead -- that
      // is the whole first gesture on a camera nobody has cropped yet.
      return mode === "move" && isFullFrame(rect, frameW, frameH) ? "new" : mode;
    },
    [rect, frameW, frameH, gripTolerance]
  );

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled || frameW <= 0 || frameH <= 0) return;
    const point = toSourcePoint(event.clientX, event.clientY);
    const mode = hitTest(point);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    setDrag({ mode, origin: point, startRect: rect });
    setHover(mode);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    const point = toSourcePoint(event.clientX, event.clientY);
    if (!drag) {
      setHover(hitTest(point));
      return;
    }
    if (drag.mode === "new") {
      onChange(cropFromCorners(drag.origin, point, frameW, frameH));
      return;
    }
    if (drag.mode === "move") {
      onChange(moveCrop(drag.startRect, point.x - drag.origin.x, point.y - drag.origin.y, frameW, frameH));
      return;
    }
    onChange(resizeCrop(drag.startRect, drag.mode as CropHandle, point, frameW, frameH));
  };

  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDrag(null);
  };

  const box = cropStyleBox(rect, frameW, frameH);
  const right = pct(rect.x + rect.w, frameW);
  const bottom = pct(rect.y + rect.h, frameH);

  return (
    <div className="crop-picker">
      <div
        className={`crop-surface${disabled ? " disabled" : ""}`}
        ref={surfaceRef}
        style={{
          aspectRatio: frameW > 0 && frameH > 0 ? `${frameW} / ${frameH}` : undefined,
          cursor: disabled ? "default" : cropCursor(drag?.mode ?? hover)
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {shownUrl ? <img className="crop-frame" src={shownUrl} alt="" draggable={false} /> : null}
        {/* Four shades rather than one masked overlay: the cut-out has to be pixel-exact
            against the box, and percentage rects are exact at any rendered size. */}
        <div className="crop-shade" style={{ left: 0, top: 0, width: "100%", height: box.top }} />
        <div className="crop-shade" style={{ left: 0, top: bottom, width: "100%", bottom: 0 }} />
        <div className="crop-shade" style={{ left: 0, top: box.top, width: box.left, height: box.height }} />
        <div className="crop-shade" style={{ left: right, top: box.top, right: 0, height: box.height }} />
        <div className="crop-box" style={box}>
          <span className="crop-box-label">
            {rect.w}&times;{rect.h}
          </span>
          {CROP_HANDLES.map((handle) => (
            <span className={`crop-handle ${handle}`} key={handle} />
          ))}
        </div>
        {!frameUrl ? (
          <div className="crop-frame-note">No recording to preview. The box below still applies to every build.</div>
        ) : null}
        {frameUrl && status === "loading" && !shownUrl ? (
          <div className="crop-frame-note">Decoding frame…</div>
        ) : null}
        {status === "error" ? (
          <div className="crop-frame-note error">
            No frame for this camera. Pick another episode, or check the recording has video.
          </div>
        ) : null}
      </div>
      <p className="crop-hint">
        {disabled
          ? "Locked while a build is running."
          : "Drag on the frame to draw a box, inside it to move, on a handle to resize."}
        {status === "loading" && shownUrl ? " · decoding…" : ""}
      </p>
    </div>
  );
}

/**
 * One crop coordinate, committed on blur or Enter.
 *
 * Not a controlled number input: clearing the field to retype a value would parse as NaN and
 * snap the box back mid-edit. The typed text is local until it is committed, and what is
 * committed is whatever the caller's `normalizeCrop` makes of it.
 */
export function CropNumberField({
  label,
  value,
  disabled,
  onCommit
}: {
  label: string;
  value: number;
  disabled: boolean;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  const commit = () => {
    const parsed = Number(draft);
    setDraft(null);
    if (draft === null || draft.trim() === "" || !Number.isFinite(parsed)) return;
    onCommit(Math.round(parsed));
  };

  return (
    <label>
      <span>{label}</span>
      <input
        type="number"
        step={2}
        value={draft ?? String(value)}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
          if (event.key === "Escape") setDraft(null);
        }}
      />
    </label>
  );
}
