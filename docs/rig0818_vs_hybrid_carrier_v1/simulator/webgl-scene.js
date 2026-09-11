/* Dependency-free WebGL view of the real-input scene.
 *
 * Two projections share one set of buffers: an orbit perspective camera, and
 * a production camera's OpenCV equidistant fisheye model evaluated per vertex,
 * so the "camera view" puts the rig where that camera's image actually has it.
 * Straight triangle edges are not re-curved; this is a view, not the renderer. */

const VERTEX = `
precision highp float;
attribute vec3 aPos; attribute vec3 aNormal; attribute vec3 aColor;
uniform mat4 uModel; uniform mat4 uView; uniform mat4 uProj;
uniform float uFisheye; uniform vec4 uK; uniform vec4 uD; uniform vec2 uImage; uniform float uFlat;
varying vec3 vColor;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vec3 n = (uModel * vec4(aNormal, 0.0)).xyz;
  float len = length(n);
  float shade = (uFlat > 0.5 || len < 1e-6) ? 1.0 : 0.56 + 0.44 * abs(dot(n / len, normalize(vec3(0.35, -0.45, 0.82))));
  vColor = aColor * shade;
  vec4 c = uView * world;
  if (uFisheye < 0.5) { gl_Position = uProj * c; return; }
  float r = length(c.xy);
  float theta = atan(r, c.z);
  float t2 = theta * theta;
  float td = theta * (1.0 + uD.x * t2 + uD.y * t2 * t2 + uD.z * t2 * t2 * t2 + uD.w * t2 * t2 * t2 * t2);
  float s = r > 1e-9 ? td / r : 1.0;
  float u = uK.x * c.x * s + uK.z;
  float v = uK.y * c.y * s + uK.w;
  float z = clamp(length(c.xyz) / 6.0, 0.0, 1.0) * 2.0 - 1.0;
  gl_Position = theta > 1.45 ? vec4(0.0, 0.0, 2.0, 1.0) : vec4(u / uImage.x * 2.0 - 1.0, 1.0 - v / uImage.y * 2.0, z, 1.0);
}`;

const FRAGMENT = `
precision mediump float;
varying vec3 vColor; uniform float uAlpha;
void main() { gl_FragColor = vec4(vColor, uAlpha); }`;

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const unit = a => { const n = Math.hypot(...a) || 1; return scale(a, 1 / n); };

export function hexRgb(hex) {
  const h = hex.replace('#', ''), n = parseInt(h.length === 3 ? h.split('').map(x => x + x).join('') : h, 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/* 4x4 matrices: plain arrays of rows in the report, Float32Array column-major for GL. */
export function rowsToColumn(rows) {
  const m = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[c * 4 + r] = rows[r][c];
  return m;
}
export function multiplyRows(A, B) {
  return A.map((row, i) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
}
export function invertRigidRows(T) {
  const R = [[T[0][0], T[1][0], T[2][0]], [T[0][1], T[1][1], T[2][1]], [T[0][2], T[1][2], T[2][2]]];
  const t = [T[0][3], T[1][3], T[2][3]];
  const ti = R.map(row => -dot(row, t));
  return [[...R[0], ti[0]], [...R[1], ti[1]], [...R[2], ti[2]], [0, 0, 0, 1]];
}
export function pose7Rows([x, y, z, qx, qy, qz, qw]) {
  const n = Math.hypot(qx, qy, qz, qw) || 1; qx /= n; qy /= n; qz /= n; qw /= n;
  return [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), x],
    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), y],
    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), z],
    [0, 0, 0, 1]];
}
export function applyRows(T, p) {
  return [T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3], T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3], T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3]];
}
function perspectiveRows(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return [[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]];
}
function lookAtRows(eye, centre, up) {
  const z = unit(sub(eye, centre)), x = unit(cross(up, z)), y = cross(z, x);
  return [[...x, -dot(x, eye)], [...y, -dot(y, eye)], [...z, -dot(z, eye)], [0, 0, 0, 1]];
}

export function projectFisheyeCamera(camera, T_cam_world, point) {
  const c = applyRows(T_cam_world, point);
  if (c[2] <= 0.02) return null;
  const r = Math.hypot(c[0], c[1]), theta = Math.atan2(r, c[2]), t2 = theta * theta, [k1, k2, k3, k4] = camera.D;
  const td = theta * (1 + k1 * t2 + k2 * t2 * t2 + k3 * t2 ** 3 + k4 * t2 ** 4), s = r > 1e-9 ? td / r : 1;
  return {u: camera.K[0][0] * c[0] * s + camera.K[0][2], v: camera.K[1][1] * c[1] * s + camera.K[1][2], depth: Math.hypot(...c)};
}

class Builder {
  constructor() { this.p = []; this.n = []; this.c = []; }
  tri(a, b, c, color) {
    const n = unit(cross(sub(b, a), sub(c, a)));
    for (const v of [a, b, c]) { this.p.push(...v); this.n.push(...n); this.c.push(...color); }
  }
  poly(points, color, offset = 0) {
    const n = unit(cross(sub(points[1], points[0]), sub(points[2], points[0])));
    const pts = offset ? points.map(p => add(p, scale(n, offset))) : points;
    for (let i = 1; i + 1 < pts.length; i++) this.tri(pts[0], pts[i], pts[i + 1], color);
  }
  line(a, b, color) { for (const v of [a, b]) { this.p.push(...v); this.n.push(0, 0, 0); this.c.push(...color); } }
}

function icosphere(builder, centre, radius, color) {
  const t = (1 + Math.sqrt(5)) / 2;
  let v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(unit);
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const next = [];
  for (const [a, b, c] of f) {
    const ab = v.push(unit(scale(add(v[a], v[b]), .5))) - 1, bc = v.push(unit(scale(add(v[b], v[c]), .5))) - 1, ca = v.push(unit(scale(add(v[c], v[a]), .5))) - 1;
    next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
  }
  f = next;
  for (const [a, b, c] of f) builder.tri(add(centre, scale(v[a], radius)), add(centre, scale(v[b], radius)), add(centre, scale(v[c], radius)), color);
}

function squarePrism(builder, a, b, half, color) {
  const axis = unit(sub(b, a)), helper = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const u = unit(cross(axis, helper)), v = cross(axis, u);
  const ring = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([su, sv]) => add(scale(u, half * su), scale(v, half * sv)));
  for (let k = 0; k < 4; k++) {
    const p0 = add(a, ring[k]), p1 = add(a, ring[(k + 1) % 4]), q0 = add(b, ring[k]), q1 = add(b, ring[(k + 1) % 4]);
    builder.tri(p0, p1, q1, color); builder.tri(p0, q1, q0, color);
  }
}

/* Target meshes in their own CAD frame (metres), from simulator-core target objects. */
export function targetMesh(target) {
  const b = new Builder(), black = [0.08, 0.09, 0.09], white = [0.95, 0.94, 0.9];
  if (target.family === 'rig') {
    for (const anchor of target.anchors) {
      const centre = scale(anchor.points.reduce((s, p) => add(s, p), [0, 0, 0]), .25), edge = Math.hypot(...sub(anchor.points[1], anchor.points[0]));
      const pad = anchor.points.map(p => add(centre, scale(sub(p, centre), 0.0696 / edge)));
      squarePrism(b, [0, 0, 0], sub(centre, scale(anchor.normal, 0.004)), 0.005, black);
      b.poly(pad.slice().reverse(), black, 0);
      b.poly(pad, white, 0.0006 * -1);
      b.poly(anchor.points, black, -0.0012);
    }
  } else {
    const body = [0.77, 0.78, 0.8];
    for (const tri of target.triangles) b.tri(tri[0], tri[1], tri[2], body);
    for (const poly of target.displayPolygons) {
      if (poly.anchor) {
        b.poly(poly.points, white, 0.0004);
        b.poly(poly.markerPoints.slice().reverse(), black, 0.0008);
      } else {
        b.poly(poly.points, hexRgb(poly.color), 0.0003);
      }
    }
  }
  return b;
}

export class SceneView {
  constructor(canvas, overlay) {
    this.canvas = canvas;
    this.overlay = overlay;
    const gl = canvas.getContext('webgl', {antialias: true, preserveDrawingBuffer: true}) || canvas.getContext('experimental-webgl');
    if (!gl) throw new Error('WebGL unavailable in this browser');
    this.gl = gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX)); gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    this.program = program;
    this.loc = Object.fromEntries(['aPos', 'aNormal', 'aColor'].map(n => [n, gl.getAttribLocation(program, n)]));
    this.uni = Object.fromEntries(['uModel', 'uView', 'uProj', 'uFisheye', 'uK', 'uD', 'uImage', 'uFlat', 'uAlpha'].map(n => [n, gl.getUniformLocation(program, n)]));
    this.meshes = {};
    this.dynamic = null;
  }

  upload(builder, mode) {
    const gl = this.gl, mesh = {count: builder.p.length / 3, mode, buffers: {}};
    for (const [name, data] of [['aPos', builder.p], ['aNormal', builder.n], ['aColor', builder.c]]) {
      const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      mesh.buffers[name] = buffer;
    }
    return mesh;
  }

  release(mesh) { if (mesh) for (const buffer of Object.values(mesh.buffers)) this.gl.deleteBuffer(buffer); }

  setStatic({cameras, gripper, targets, mounts, tcpInBox}) {
    this.cameras = cameras.map(c => ({...c, T_cam_world: invertRigidRows(c.T_world_camera)}));
    this.mounts = mounts;
    this.tcpInBox = tcpInBox;
    const grey = [0.36, 0.38, 0.41], housing = [0.25, 0.27, 0.3], skin = [0.8, 0.62, 0.5];
    const g = new Builder();
    for (const tri of gripper.mechanism.triangles) g.tri(tri[0], tri[1], tri[2], grey);
    for (const tri of gripper.housing.triangles) g.tri(tri[0], tri[1], tri[2], housing);
    const hand = new Builder();
    for (const sphere of gripper.hand.spheres) icosphere(hand, sphere.c, sphere.r, skin);
    this.meshes.gripper = this.upload(g, this.gl.TRIANGLES);
    this.meshes.hand = this.upload(hand, this.gl.TRIANGLES);
    this.meshes.R0 = this.upload(targetMesh(targets.R0), this.gl.TRIANGLES);
    this.meshes.H = this.upload(targetMesh(targets.H1), this.gl.TRIANGLES);
    const lines = new Builder(), frame = [0.72, 0.84, 0.79];
    for (const camera of this.cameras) {
      const T = camera.T_world_camera, o = [T[0][3], T[1][3], T[2][3]];
      const corners = [[-0.9, -0.5, 0.55], [0.9, -0.5, 0.55], [0.9, 0.5, 0.55], [-0.9, 0.5, 0.55]].map(p => applyRows(T, scale(p, 0.14)));
      corners.forEach((c, i) => { lines.line(o, c, frame); lines.line(c, corners[(i + 1) % 4], frame); });
    }
    for (let x = -0.2; x <= 1.4001; x += 0.1) lines.line([x, -0.8, 0], [x, 0.8, 0], [0.2, 0.28, 0.25]);
    for (let y = -0.8; y <= 0.8001; y += 0.1) lines.line([-0.2, y, 0], [1.4, y, 0], [0.2, 0.28, 0.25]);
    this.meshes.frames = this.upload(lines, this.gl.LINES);
  }

  setReplay(replay, colourOf) {
    this.replay = replay;
    this.rowOf = new Map(replay.hand.map((hand, i) => [`${hand}:${replay.index[i]}`, i]));
    this.boxes = replay.box_pose7.map(pose7Rows);
    this.tcp = this.boxes.map(T => applyRows(T, this.tcpInBox));
    this.recolour(colourOf);
  }

  recolour(colourOf) {
    const lines = new Builder(), r = this.replay;
    for (let i = 1; i < r.hand.length; i++) {
      if (r.hand[i] !== r.hand[i - 1] || r.episode[i] !== r.episode[i - 1]) continue;
      lines.line(this.tcp[i - 1], this.tcp[i], colourOf(i));
    }
    this.release(this.meshes.trajectory);
    this.meshes.trajectory = this.upload(lines, this.gl.LINES);
  }

  draw(mesh, model, flat = false, alpha = 1) {
    const gl = this.gl;
    for (const name of ['aPos', 'aNormal', 'aColor']) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buffers[name]); gl.enableVertexAttribArray(this.loc[name]);
      gl.vertexAttribPointer(this.loc[name], 3, gl.FLOAT, false, 0, 0);
    }
    gl.uniformMatrix4fv(this.uni.uModel, false, rowsToColumn(model));
    gl.uniform1f(this.uni.uFlat, flat ? 1 : 0);
    gl.uniform1f(this.uni.uAlpha, alpha);
    gl.drawArrays(mesh.mode, 0, mesh.count);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(16, Math.round(rect.width * dpr)), h = Math.max(16, Math.round(rect.height * dpr));
    for (const c of [this.canvas, this.overlay]) if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return {w, h, dpr, cssW: rect.width, cssH: rect.height};
  }

  render(state) {
    const gl = this.gl, size = this.resize(), identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    const row = state.row, hand = this.replay.hand[row], index = this.replay.index[row], other = this.rowOf.get(`${hand === 'left' ? 'right' : 'left'}:${index}`);
    const target = state.arm === 'R0' ? 'R0' : 'H';
    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.07, 0.13, 0.11, 1);
    let viewport = [0, 0, size.w, size.h], camera = null;
    if (state.mode === 'camera') {
      camera = this.cameras[state.cameraIndex] || this.cameras[0];
      const aspect = camera.width / camera.height, w = Math.min(size.w, size.h * aspect), h = w / aspect;
      viewport = [Math.round((size.w - w) / 2), Math.round((size.h - h) / 2), Math.round(w), Math.round(h)];
    }
    gl.viewport(0, 0, size.w, size.h);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.viewport(...viewport);
    let view, project;
    if (camera) {
      view = camera.T_cam_world;
      gl.uniform1f(this.uni.uFisheye, 1);
      gl.uniform4f(this.uni.uK, camera.K[0][0], camera.K[1][1], camera.K[0][2], camera.K[1][2]);
      gl.uniform4f(this.uni.uD, ...camera.D);
      gl.uniform2f(this.uni.uImage, camera.width, camera.height);
      gl.uniformMatrix4fv(this.uni.uProj, false, rowsToColumn(identity));
      project = p => { const uv = projectFisheyeCamera(camera, view, p); return uv && {x: (viewport[0] + uv.u / camera.width * viewport[2]) / size.dpr, y: (viewport[1] + uv.v / camera.height * viewport[3]) / size.dpr}; };
    } else {
      const centre = state.orbit.centre, {azimuth: a, elevation: e, distance: d} = state.orbit;
      const eye = add(centre, [d * Math.cos(e) * Math.cos(a), d * Math.cos(e) * Math.sin(a), d * Math.sin(e)]);
      view = lookAtRows(eye, centre, [0, 0, 1]);
      const proj = perspectiveRows(0.8, size.w / size.h, 0.02, 20);
      gl.uniform1f(this.uni.uFisheye, 0);
      gl.uniformMatrix4fv(this.uni.uProj, false, rowsToColumn(proj));
      const PV = multiplyRows(proj, view);
      project = p => { const q = [...applyRows(PV, p), PV[3][0] * p[0] + PV[3][1] * p[1] + PV[3][2] * p[2] + PV[3][3]]; return q[3] <= 0 ? null : {x: (q[0] / q[3] + 1) / 2 * size.cssW, y: (1 - q[1] / q[3]) / 2 * size.cssH}; };
    }
    gl.uniformMatrix4fv(this.uni.uView, false, rowsToColumn(view));
    const own = this.boxes[row], mount = this.mounts[target];
    if (state.mode !== 'camera') this.draw(this.meshes.frames, identity, true, 1);
    this.draw(this.meshes.trajectory, identity, true, 1);
    for (const T of [own, other === undefined ? null : this.boxes[other]]) {
      if (!T) continue;
      this.draw(this.meshes.gripper, T); this.draw(this.meshes.hand, T);
    }
    const T_world_rig = multiplyRows(own, mount);
    this.draw(this.meshes[target], T_world_rig);
    // Rays from every camera that decoded at least one of this target's markers.
    const decoded = this.replay.decoded[target][row], centroid = applyRows(T_world_rig, state.targetCentroid[target]);
    const rays = new Builder();
    this.cameras.forEach((cam, k) => {
      if (!decoded[k]) return;
      const o = [cam.T_world_camera[0][3], cam.T_world_camera[1][3], cam.T_world_camera[2][3]];
      rays.line(o, centroid, decoded[k] >= 2 ? [0.45, 0.93, 0.7] : [0.95, 0.75, 0.3]);
    });
    if (state.mode !== 'camera' && rays.p.length) {
      const mesh = this.upload(rays, gl.LINES); this.draw(mesh, identity, true, 1); this.release(mesh);
    }
    const ctx = this.overlay.getContext('2d');
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
    ctx.clearRect(0, 0, size.cssW, size.cssH);
    ctx.font = '11px system-ui, sans-serif';
    if (camera) {
      ctx.strokeStyle = 'rgba(214,236,226,.45)';
      ctx.strokeRect(viewport[0] / size.dpr + .5, viewport[1] / size.dpr + .5, viewport[2] / size.dpr - 1, viewport[3] / size.dpr - 1);
      ctx.fillStyle = 'rgba(230,244,238,.9)';
      ctx.fillText(`${camera.name} · ${camera.width}×${camera.height} fisheye · 解码 ${decoded[state.cameraIndex] || 0} 个 marker`, viewport[0] / size.dpr + 10, viewport[1] / size.dpr + 18);
    } else {
      this.cameras.forEach((cam, k) => {
        const p = project([cam.T_world_camera[0][3], cam.T_world_camera[1][3], cam.T_world_camera[2][3]]);
        if (!p) return;
        ctx.fillStyle = decoded[k] ? 'rgba(160,240,200,.95)' : 'rgba(200,214,208,.55)';
        ctx.fillText(`${cam.name}${decoded[k] ? ` · ${decoded[k]}` : ''}`, p.x + 6, p.y - 6);
      });
    }
    return {project};
  }
}
