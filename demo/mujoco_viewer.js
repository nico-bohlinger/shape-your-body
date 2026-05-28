/**
 * Universal MuJoCo Viewer Module
 *
 * Loads any MuJoCo XML scene, auto-discovers all referenced files
 * (includes, meshes, textures), and renders it interactively with
 * Three.js. Supports drag-perturbation and orbit controls.
 *
 * Usage:
 *   import { MuJoCoViewer } from './mujoco_viewer.js';
 *   const viewer = new MuJoCoViewer(containerEl, {
 *     sceneXML: 'xmls/scene_handover.xml',  // relative to the HTML page
 *     wasmURL:  '../dist/mujoco_wasm.js',
 *   });
 *   await viewer.init();
 *
 * Dependencies: three.js + OrbitControls loaded via importmap before this.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Helpers ───────────────────────────────────────────────

/** MuJoCo Z-up → Three.js Y-up */
function getPosition(buf, i, target) {
  return target.set(buf[i*3], buf[i*3+2], -buf[i*3+1]);
}
function getQuaternion(buf, i, target) {
  return target.set(-buf[i*4+1], -buf[i*4+3], buf[i*4+2], -buf[i*4+0]);
}
function toMujocoPos(v) { return v.set(v.x, -v.z, v.y); }

const BINARY_EXTS = new Set([
  '.stl', '.obj', '.msh', '.png', '.jpg', '.jpeg', '.bin', '.glb', '.mtl',
]);
function isBinaryFile(path) {
  const ext = '.' + path.split('.').pop().toLowerCase();
  return BINARY_EXTS.has(ext);
}

// ─── XML File Collector ────────────────────────────────────

/**
 * Parse a MuJoCo XML, recursively follow <include> tags,
 * and collect all referenced mesh/texture files.
 *
 * Returns { files: Set<string> } — paths relative to baseURL.
 */
async function collectFiles(xmlRelPath, baseURL, visited = new Set()) {
  if (visited.has(xmlRelPath)) return new Set();
  visited.add(xmlRelPath);

  const files = new Set();
  files.add(xmlRelPath);

  const xmlURL = baseURL + xmlRelPath;
  const resp = await fetch(xmlURL);
  if (!resp.ok) { console.warn(`collectFiles: failed to fetch ${xmlURL}`); return files; }
  const text = await resp.text();

  // Directory this XML lives in (for resolving relative paths)
  const xmlDir = xmlRelPath.includes('/') ? xmlRelPath.substring(0, xmlRelPath.lastIndexOf('/') + 1) : '';

  // Parse compiler directives (meshdir, texturedir, assetdir).
  // assetdir is MuJoCo shorthand: sets both meshdir and texturedir when those
  // aren't given explicitly.
  let meshdir = '', texturedir = '';
  const compilerMatch = text.match(/<compiler[^>]*>/i);
  if (compilerMatch) {
    const ad = compilerMatch[0].match(/assetdir\s*=\s*["']([^"']+)["']/);
    const md = compilerMatch[0].match(/meshdir\s*=\s*["']([^"']+)["']/);
    const td = compilerMatch[0].match(/texturedir\s*=\s*["']([^"']+)["']/);
    if (ad) {
      const a = ad[1].replace(/\/$/, '') + '/';
      meshdir = a; texturedir = a;
    }
    if (md) meshdir = md[1].replace(/\/$/, '') + '/';
    if (td) texturedir = td[1].replace(/\/$/, '') + '/';
  }

  // Collect <include file="..."/>
  const includeRe = /<include\s+file\s*=\s*["']([^"']+)["']\s*\/?>/gi;
  let m;
  while ((m = includeRe.exec(text)) !== null) {
    const inclPath = xmlDir + m[1];
    const sub = await collectFiles(inclPath, baseURL, visited);
    for (const f of sub) files.add(f);
  }

  // Collect <mesh file="..."/>
  const meshRe = /<mesh\b[^>]*\bfile\s*=\s*["']([^"']+)["']/gi;
  while ((m = meshRe.exec(text)) !== null) {
    files.add(xmlDir + meshdir + m[1]);
  }

  // Collect <texture ... file="..."/> (skip builtins)
  const texRe = /<texture\b[^>]*\bfile\s*=\s*["']([^"']+)["']/gi;
  while ((m = texRe.exec(text)) !== null) {
    files.add(xmlDir + texturedir + m[1]);
  }

  // Collect <hfield ... file="..."/>
  const hfRe = /<hfield\b[^>]*\bfile\s*=\s*["']([^"']+)["']/gi;
  while ((m = hfRe.exec(text)) !== null) {
    files.add(xmlDir + meshdir + m[1]);
  }

  return files;
}

// ─── Viewer Class ──────────────────────────────────────────

export class MuJoCoViewer {
  /**
   * @param {HTMLElement} container  – the DOM element to render into
   * @param {Object}      opts
   * @param {string}      opts.sceneXML   – path to scene XML (relative to page)
   * @param {string}      opts.wasmURL    – path to mujoco_wasm.js  (default '../dist/mujoco_wasm.js')
   * @param {number[]}    [opts.cameraPos]   – [x,y,z] initial camera position
   * @param {number[]}    [opts.cameraTarget]– [x,y,z] orbit target
   * @param {string}      [opts.background]  – CSS hex color (default '#f0f2f5')
   * @param {Function}    [opts.onStatus]    – called with status text during loading
   * @param {Function}    [opts.onReady]     – called with (mjModel, mjData, mujoco) once loaded
   * @param {boolean}     [opts.enableDrag]  – enable drag perturbation (default true)
   * @param {boolean}     [opts.paused]      – start paused (default false)
   */
  constructor(container, opts = {}) {
    this.container = container;
    this.sceneXML  = opts.sceneXML  || 'scene.xml';
    this.wasmURL   = opts.wasmURL   || '../dist/mujoco_wasm.js';
    this.bgColor   = opts.background || '#f0f2f5';
    this.onStatus  = opts.onStatus  || (() => {});
    this.onReady   = opts.onReady   || (() => {});
    this.enableDrag = opts.enableDrag !== false;
    this.paused    = opts.paused || false;
    this._frozen   = false;

    this.initialCameraPos    = opts.cameraPos    || null;
    this.initialCameraTarget = opts.cameraTarget || null;

    // MuJoCo
    this.mujoco  = null;
    this.mjModel = null;
    this.mjData  = null;

    // Three.js
    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.controls = null;

    // Body tracking
    this.mujocoRoot = null;
    this.bodies     = {};
    this.bodyMeshes = [];

    // Textures
    this.textureLoader = new THREE.TextureLoader();
    this.textures = {};

    // Drag state
    this.dragging       = false;
    this.dragBodyId     = -1;
    this.dragPointLocal = new THREE.Vector3();
    this.dragPointWorld = new THREE.Vector3();
    this.dragCursorWorld= new THREE.Vector3();
    this.dragPlane      = new THREE.Plane();
    this.dragArrow      = null;
    this.raycaster      = new THREE.Raycaster();
    this.pointer        = new THREE.Vector2();

    // Timing
    this._lastTime = 0;
    this._disposed = false;

    // Trajectory playback
    this._traj           = null;   // { qpos, qvel, ctrl, mocap, metadata }
    this._trajStep       = 0;
    this._trajPlaying    = false;
    this._trajLoop       = true;
    this._trajSpeed      = 1.0;    // 1× = real-time
    this._trajAccum      = 0;      // accumulated ms for sub-stepping
    this._trajCallbacks  = { onStep: null, onDone: null, onPhysicsStep: null };
    this._trajQdes       = null;  // Float64Array[armCount] — latest retargeted targets

    // Trajectory mode & PD control
    this._trajMode         = 'forward';  // 'forward' (mj_forward) or 'step' (mj_step + PD)
    this._trajScalarKp     = 200;
    this._trajScalarKd     = 10;
    this._trajArmCount     = 7;
    this._trajGainBase     = [1, 1, 1, 1, 0.5, 0.5, 0.5];
    this._trajTorqueLimits = [87, 87, 87, 87, 12, 12, 12];
    this._trajCtrlDecimation = 1;  // ZOH: hold retargeted q_des for N traj steps
    this._trajSubStep      = 0;
    this._trajNoise        = 0;    // Noise magnitude for q_des perturbation
  }

  // ── Public API ──────────────────────────────────────────

  async init() {
    await this._loadWASM();
    await this._loadModelFiles();
    this._initModel();
    await this._loadTextures();
    this._buildScene();
    this._bindEvents();

    this.onReady(this.mjModel, this.mjData, this.mujoco);

    this._lastTime = performance.now();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  /** Completely stop the render loop (no rAF scheduling). */
  freeze() {
    this._frozen = true;
  }

  /** Resume the render loop after a freeze. */
  unfreeze() {
    if (this._frozen) {
      this._frozen = false;
      this._lastTime = performance.now();
      requestAnimationFrame(this._animate);
    }
  }

  dispose() {
    this._disposed = true;
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.domElement.remove();
    }
  }

  resetData() {
    if (!this.mjModel || !this.mjData) return;
    this.mujoco.mj_resetData(this.mjModel, this.mjData);
    this.mujoco.mj_forward(this.mjModel, this.mjData);
  }

  // ── Trajectory Playback API ─────────────────────────────

  /**
   * Load a trajectory dataset (JSON) for playback.
   *
   * Expected JSON format:
   *   { metadata: { num_steps, decimation, shapes: {...} },
   *     qpos: [[...], ...],   // [num_steps][nq]
   *     qvel: [[...], ...],   // [num_steps][nv]   (optional)
   *     ctrl: [[...], ...],   // [num_steps][nu]   (optional)
   *     mocap: [[[pos,quat], ...], ...]  // [num_steps][nmocap][7] (optional)
   *   }
   *
   * @param {string|Object} src – URL to JSON file, or pre-parsed object
   * @returns {Promise<Object>} loaded trajectory metadata
   */
  async loadTrajectory(src) {
    let data;
    if (typeof src === 'string') {
      const resp = await fetch(src);
      if (!resp.ok) throw new Error(`Failed to fetch trajectory: ${resp.status}`);
      data = await resp.json();
    } else {
      data = src;
    }

    this._traj = {
      qpos:  data.qpos  || null,
      qvel:  data.qvel  || null,
      ctrl:  data.ctrl  || null,
      mocap: data.mocap  || null,
      meta:  data.metadata || {},
    };
    this._trajStep    = 0;
    this._trajPlaying = false;
    this._trajAccum   = 0;
    this._trajSubStep = 0;

    // Apply first frame immediately
    this._applyTrajStep(0);

    return this._traj.meta;
  }

  /** Number of steps in the loaded trajectory (0 if none). */
  get trajLength() {
    return this._traj?.qpos?.length ?? 0;
  }

  /** Current playback step index. */
  get trajStep() { return this._trajStep; }

  /** Whether trajectory is currently playing. */
  get trajPlaying() { return this._trajPlaying; }

  /** Playback speed multiplier (default 1.0). */
  get trajSpeed() { return this._trajSpeed; }
  set trajSpeed(v) { this._trajSpeed = Math.max(0.1, Math.min(v, 10)); }

  /** Whether playback loops at the end. */
  get trajLoop() { return this._trajLoop; }
  set trajLoop(v) { this._trajLoop = !!v; }

  /** Trajectory mode: 'forward' (kinematic) or 'step' (physics + PD). */
  get trajMode() { return this._trajMode; }
  set trajMode(v) {
    if (v !== 'forward' && v !== 'step') return;
    const changed = this._trajMode !== v;
    this._trajMode = v;
    if (changed && this._traj) {
      this._trajSubStep = 0;
      this._trajAccum   = 0;
      this._applyTrajStep(this._trajStep);
    }
  }

  /** Scalar Kp gain for PD control in step mode. */
  get trajScalarKp() { return this._trajScalarKp; }
  set trajScalarKp(v) { this._trajScalarKp = Math.max(1, v); }

  /** Scalar Kd gain for PD control in step mode. */
  get trajScalarKd() { return this._trajScalarKd; }
  set trajScalarKd(v) { this._trajScalarKd = Math.max(0.01, v); }

  /** Configure PD controller parameters for step mode. */
  setTrajPDConfig({ armCount, gainBase, torqueLimits } = {}) {
    if (armCount !== undefined) this._trajArmCount = armCount;
    if (gainBase) this._trajGainBase = gainBase;
    if (torqueLimits) this._trajTorqueLimits = torqueLimits;
  }

  /** Control decimation: hold retargeted q_des for N trajectory steps (ZOH). */
  get trajCtrlDecimation() { return this._trajCtrlDecimation; }
  set trajCtrlDecimation(v) { this._trajCtrlDecimation = Math.max(1, Math.round(v)); }

  /** Noise magnitude for q_des perturbation in step mode. */
  get trajNoise() { return this._trajNoise; }
  set trajNoise(v) { this._trajNoise = Math.max(0, v); }

  /** Start / resume playback. */
  trajPlay() {
    if (!this._traj) return;
    this._trajPlaying = true;
    this._trajAccum   = 0;
  }

  /** Pause playback. */
  trajPause() {
    this._trajPlaying = false;
  }

  /** Toggle play/pause. Returns new playing state. */
  trajToggle() {
    this._trajPlaying ? this.trajPause() : this.trajPlay();
    return this._trajPlaying;
  }

  /**
   * Seek to a specific step.
   * @param {number} step – 0-based step index
   */
  trajSeek(step) {
    if (!this._traj) return;
    this._trajStep    = Math.max(0, Math.min(step, this.trajLength - 1));
    this._trajAccum   = 0;
    this._trajSubStep = 0;
    this._applyTrajStep(this._trajStep);
  }

  /** Set callback fired every step. Signature: fn(stepIndex, numSteps). */
  onTrajStep(fn)  { this._trajCallbacks.onStep = fn; }

  /** Set callback fired when playback finishes (non-looping). */
  onTrajDone(fn)  { this._trajCallbacks.onDone = fn; }

  /** Set callback fired each physics sub-step with current qpos & qdes. */
  onPhysicsStep(fn) { this._trajCallbacks.onPhysicsStep = fn; }

  /** Apply trajectory data at given step to mjData and run mj_forward. */
  _applyTrajStep(step) {
    if (!this._traj || !this.mjData) return;
    const t = this._traj;

    // qpos
    if (t.qpos && t.qpos[step]) {
      const q = t.qpos[step];
      for (let i = 0; i < q.length && i < this.mjData.qpos.length; i++) {
        this.mjData.qpos[i] = q[i];
      }
    }

    // qvel
    if (t.qvel && t.qvel[step]) {
      const v = t.qvel[step];
      for (let i = 0; i < v.length && i < this.mjData.qvel.length; i++) {
        this.mjData.qvel[i] = v[i];
      }
    }

    // ctrl
    if (t.ctrl && t.ctrl[step]) {
      const c = t.ctrl[step];
      for (let i = 0; i < c.length && i < this.mjData.ctrl.length; i++) {
        this.mjData.ctrl[i] = c[i];
      }
    }

    // mocap bodies: each entry is [nmocap][7] where 7 = pos(3) + quat(4)
    if (t.mocap && t.mocap[step]) {
      const mocapArr = t.mocap[step];
      for (let m = 0; m < mocapArr.length; m++) {
        const vals = mocapArr[m]; // [px, py, pz, qw, qx, qy, qz]
        if (this.mjData.mocap_pos) {
          this.mjData.mocap_pos[m * 3 + 0] = vals[0];
          this.mjData.mocap_pos[m * 3 + 1] = vals[1];
          this.mjData.mocap_pos[m * 3 + 2] = vals[2];
        }
        if (this.mjData.mocap_quat) {
          this.mjData.mocap_quat[m * 4 + 0] = vals[3];
          this.mjData.mocap_quat[m * 4 + 1] = vals[4];
          this.mjData.mocap_quat[m * 4 + 2] = vals[5];
          this.mjData.mocap_quat[m * 4 + 3] = vals[6];
        }
      }
    }

    // Forward kinematics (compute body positions/orientations from qpos)
    this.mujoco.mj_forward(this.mjModel, this.mjData);
  }

  /** Advance trajectory by elapsed ms (called from _animate). */
  _stepTrajectory(dtMs) {
    if (!this._traj || !this._trajPlaying) return;

    if (this._trajMode === 'step') {
      this._stepTrajectoryPhysics(dtMs);
      return;
    }

    // ── Forward mode (kinematic replay via mj_forward) ──
    const timestepMs = this.mjModel.opt.timestep * 1000;
    const decimation = this._traj.meta.decimation || 1;
    const frameMs    = timestepMs * decimation;
    const scaledMs   = dtMs * this._trajSpeed;

    this._trajAccum += scaledMs;

    while (this._trajAccum >= frameMs) {
      this._trajAccum -= frameMs;
      this._trajStep++;

      if (this._trajStep >= this.trajLength) {
        if (this._trajLoop) {
          this._trajStep = 0;
        } else {
          this._trajStep = this.trajLength - 1;
          this._trajPlaying = false;
          this._trajAccum = 0;
          if (this._trajCallbacks.onDone) this._trajCallbacks.onDone();
          return;
        }
      }

      this._applyTrajStep(this._trajStep);
      if (this._trajCallbacks.onStep) {
        this._trajCallbacks.onStep(this._trajStep, this.trajLength);
      }
    }
  }

  /**
   * Physics-based trajectory stepping (mj_step + PD control).
   *
   * Computes retargeted position targets from recorded torques:
   *   q_des = rec_qpos + (rec_tau + kd · rec_qvel) / kp
   *
   * Then applies PD control:
   *   ctrl = kp · (q_des − qpos) − kd · qvel
   */
  _stepTrajectoryPhysics(dtMs) {
    const timestepMs = this.mjModel.opt.timestep * 1000;
    const decimation = this._traj.meta.decimation || 1;
    const scaledMs   = dtMs * this._trajSpeed;
    this._trajAccum += scaledMs;

    const t = this._traj;
    let simSteps = 0;

    while (this._trajAccum >= timestepMs && simSteps < 30) {
      this._trajAccum -= timestepMs;
      simSteps++;

      const step = this._trajStep;

      // ── Set mocap bodies from trajectory ──
      if (t.mocap && t.mocap[step]) {
        const mocapArr = t.mocap[step];
        for (let m = 0; m < mocapArr.length; m++) {
          const vals = mocapArr[m];
          if (this.mjData.mocap_pos) {
            this.mjData.mocap_pos[m * 3]     = vals[0];
            this.mjData.mocap_pos[m * 3 + 1] = vals[1];
            this.mjData.mocap_pos[m * 3 + 2] = vals[2];
          }
          if (this.mjData.mocap_quat) {
            this.mjData.mocap_quat[m * 4]     = vals[3];
            this.mjData.mocap_quat[m * 4 + 1] = vals[4];
            this.mjData.mocap_quat[m * 4 + 2] = vals[5];
            this.mjData.mocap_quat[m * 4 + 3] = vals[6];
          }
        }
      }

      // ── Retargeted PD control for arm joints ──
      // Zero-order hold: snap to every Nth trajectory step
      const ctrlDec  = this._trajCtrlDecimation;
      const holdStep = Math.floor(step / ctrlDec) * ctrlDec;
      if (t.qpos && t.qpos[holdStep] && t.ctrl && t.ctrl[holdStep]) {
        const rec_q   = t.qpos[holdStep];
        const rec_v   = t.qvel ? t.qvel[holdStep] : null;
        const rec_tau = t.ctrl[holdStep];
        const nArm    = Math.min(this._trajArmCount, rec_tau.length);

        for (let i = 0; i < nArm; i++) {
          const kp = this._trajScalarKp * (this._trajGainBase[i] ?? 1);
          const kd = this._trajScalarKd * (this._trajGainBase[i] ?? 1);

          // Retarget: q_des = rec_qpos + (rec_tau + kd · rec_qvel) / kp
          const rv    = rec_v ? rec_v[i] : 0;
          let q_des = rec_q[i] + (rec_tau[i] + kd * rv) / kp;

          // Add noise perturbation if enabled
          if (this._trajNoise > 0) {
            q_des += (Math.random() - 0.5) * 2 * this._trajNoise;
          }

          // Store for response plot
          if (!this._trajQdes || this._trajQdes.length !== nArm) {
            this._trajQdes = new Float64Array(nArm);
          }
          this._trajQdes[i] = q_des;

          // PD: τ = kp · (q_des − q) − kd · q̇
          let torque = kp * (q_des - this.mjData.qpos[i]) - kd * this.mjData.qvel[i];

          // Clamp to actuator torque limits
          const limit = this._trajTorqueLimits[i] ?? 87;
          torque = Math.max(-limit, Math.min(limit, torque));

          this.mjData.ctrl[i] = torque;
        }

        // Non-arm actuators: pass recorded ctrl through directly
        for (let i = nArm; i < rec_tau.length && i < this.mjData.ctrl.length; i++) {
          this.mjData.ctrl[i] = rec_tau[i];
        }

        // Fire physics step callback with current qpos and qdes
        if (this._trajCallbacks.onPhysicsStep && this._trajQdes) {
          this._trajCallbacks.onPhysicsStep(this.mjData.qpos, this._trajQdes, nArm);
        }
      }

      // Drag perturbation (if active)
      if (this.dragging) {
        this.syncBodies();
        this._updateDragAnchor();
        this._applyDragForce();
      } else {
        for (let i = 0; i < this.mjData.qfrc_applied.length; i++) {
          this.mjData.qfrc_applied[i] = 0;
        }
      }

      // Physics step
      this.mujoco.mj_step(this.mjModel, this.mjData);

      // Advance trajectory step counter (respecting decimation)
      this._trajSubStep++;
      if (this._trajSubStep >= decimation) {
        this._trajSubStep = 0;
        this._trajStep++;

        if (this._trajStep >= this.trajLength) {
          if (this._trajLoop) {
            this._trajStep = 0;
            this._applyTrajStep(0);  // reset sim state for seamless loop
          } else {
            this._trajStep = this.trajLength - 1;
            this._trajPlaying = false;
            this._trajAccum = 0;
            if (this._trajCallbacks.onDone) this._trajCallbacks.onDone();
            return;
          }
        }

        if (this._trajCallbacks.onStep) {
          this._trajCallbacks.onStep(this._trajStep, this.trajLength);
        }
      }
    }
  }

  // ── WASM Loading ────────────────────────────────────────

  async _loadWASM() {
    this.onStatus('Loading MuJoCo WASM…');
    const mod = await import(this.wasmURL);
    this.mujoco = await mod.default();
    this.mujoco.FS.mkdir('/working');
    this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, '/working');
  }

  // ── File Discovery + Download ───────────────────────────

  async _loadModelFiles() {
    this.onStatus('Scanning scene files…');

    // Determine base URL: the directory the sceneXML lives in
    const sceneDir = this.sceneXML.includes('/')
      ? this.sceneXML.substring(0, this.sceneXML.lastIndexOf('/') + 1)
      : '';
    const sceneFile = this.sceneXML.includes('/')
      ? this.sceneXML.substring(this.sceneXML.lastIndexOf('/') + 1)
      : this.sceneXML;

    // Collect all files by parsing XML includes recursively
    const files = await collectFiles(sceneFile, sceneDir);
    const fileList = Array.from(files);

    this.onStatus(`Downloading ${fileList.length} files…`);

    // Create directories in MEMFS
    const dirs = new Set();
    fileList.forEach(f => {
      const parts = f.split('/');
      let cur = '/working';
      for (let i = 0; i < parts.length - 1; i++) {
        cur += '/' + parts[i];
        dirs.add(cur);
      }
    });
    for (const d of [...dirs].sort()) {
      if (!this.mujoco.FS.analyzePath(d).exists) {
        this.mujoco.FS.mkdir(d);
      }
    }

    // Download all files
    let loaded = 0;
    const total = fileList.length;
    const self = this;

    async function fetchOne(relPath) {
      const url = sceneDir + relPath;
      const resp = await fetch(url);
      if (!resp.ok) { console.warn(`Failed to fetch ${url}: ${resp.status}`); return; }
      if (isBinaryFile(relPath)) {
        const buf = await resp.arrayBuffer();
        self.mujoco.FS.writeFile('/working/' + relPath, new Uint8Array(buf));
      } else {
        const text = await resp.text();
        self.mujoco.FS.writeFile('/working/' + relPath, text);
      }
      loaded++;
      if (loaded % 5 === 0 || loaded === total) {
        self.onStatus(`Downloading… (${loaded}/${total})`);
      }
    }

    // Parallel download in batches of 20
    for (let i = 0; i < fileList.length; i += 20) {
      await Promise.all(fileList.slice(i, i + 20).map(fetchOne));
    }
  }

  // ── Init MuJoCo Model ──────────────────────────────────

  _initModel() {
    this.onStatus('Compiling model…');
    const sceneFile = this.sceneXML.includes('/')
      ? this.sceneXML.substring(this.sceneXML.lastIndexOf('/') + 1)
      : this.sceneXML;
    this.mjModel = this.mujoco.MjModel.loadFromXML('/working/' + sceneFile);
    this.mjData  = new this.mujoco.MjData(this.mjModel);
    this.mujoco.mj_forward(this.mjModel, this.mjData);
  }

  // ── Load Textures ──────────────────────────────────────

  async _loadTextures() {
    this.onStatus('Loading textures…');
    
    const sceneDir = this.sceneXML.includes('/')
      ? this.sceneXML.substring(0, this.sceneXML.lastIndexOf('/') + 1)
      : '';

    // Parse XML files from MEMFS to build material→texture→file mappings
    const texNameToFile = new Map();   // texture name → resolved file path in MEMFS
    const matNameToTexName = new Map(); // material name → texture name

    const walkDir = (dir) => {
      try {
        const entries = this.mujoco.FS.readdir(dir);
        for (const entry of entries) {
          if (entry === '.' || entry === '..') continue;
          const path = dir + '/' + entry;
          const stat = this.mujoco.FS.stat(path);
          if (this.mujoco.FS.isDir(stat.mode)) {
            walkDir(path);
          }
        }
      } catch (e) { /* ignore */ }
    };

    // Read and parse all XML files for texture/material definitions
    const parseXMLForTextures = (dir) => {
      try {
        const entries = this.mujoco.FS.readdir(dir);
        for (const entry of entries) {
          if (entry === '.' || entry === '..') continue;
          const path = dir + '/' + entry;
          const stat = this.mujoco.FS.stat(path);
          if (this.mujoco.FS.isDir(stat.mode)) {
            parseXMLForTextures(path);
          } else if (entry.endsWith('.xml')) {
            try {
              const xmlText = new TextDecoder().decode(this.mujoco.FS.readFile(path));
              const xmlDir = path.substring(0, path.lastIndexOf('/') + 1);

              // Parse compiler directives for texturedir/assetdir.
              // assetdir is MuJoCo shorthand: sets texturedir when texturedir isn't given.
              let texturedir = '';
              const compilerMatch = xmlText.match(/<compiler[^>]*>/i);
              if (compilerMatch) {
                const ad = compilerMatch[0].match(/assetdir\s*=\s*["']([^"']+)["']/);
                const td = compilerMatch[0].match(/texturedir\s*=\s*["']([^"']+)["']/);
                if (ad) texturedir = ad[1].replace(/\/$/, '') + '/';
                if (td) texturedir = td[1].replace(/\/$/, '') + '/';
              }

              // Parse <texture ... name="..." file="..." />
              const texRe = /<texture\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*\bfile\s*=\s*["']([^"']+)["'][^>]*/gi;
              const texReAlt = /<texture\b[^>]*\bfile\s*=\s*["']([^"']+)["'][^>]*\bname\s*=\s*["']([^"']+)["'][^>]*/gi;
              let m;
              while ((m = texRe.exec(xmlText)) !== null) {
                const resolvedPath = xmlDir + texturedir + m[2];
                texNameToFile.set(m[1], resolvedPath);
              }
              while ((m = texReAlt.exec(xmlText)) !== null) {
                const resolvedPath = xmlDir + texturedir + m[1];
                texNameToFile.set(m[2], resolvedPath);
              }

              // Parse <material ... name="..." texture="..." />
              const matRe = /<material\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*\btexture\s*=\s*["']([^"']+)["'][^>]*/gi;
              const matReAlt = /<material\b[^>]*\btexture\s*=\s*["']([^"']+)["'][^>]*\bname\s*=\s*["']([^"']+)["'][^>]*/gi;
              while ((m = matRe.exec(xmlText)) !== null) {
                matNameToTexName.set(m[1], m[2]);
              }
              while ((m = matReAlt.exec(xmlText)) !== null) {
                matNameToTexName.set(m[2], m[1]);
              }
            } catch (e) { /* skip unreadable xml */ }
          }
        }
      } catch (e) { /* ignore */ }
    };

    parseXMLForTextures('/working');

    console.log('[MuJoCo Viewer] Material→Texture mappings:', Object.fromEntries(matNameToTexName));
    console.log('[MuJoCo Viewer] Texture→File mappings:', Object.fromEntries(texNameToFile));

    // Resolve the full chain: material name → texture name → MEMFS file path
    // Store in this.matTexFiles for use in _buildBodies
    this._matNameToTexFile = new Map();
    for (const [matName, texName] of matNameToTexName.entries()) {
      const filePath = texNameToFile.get(texName);
      if (filePath) {
        this._matNameToTexFile.set(matName, filePath);
      }
    }
    console.log('[MuJoCo Viewer] Material→File resolved:', Object.fromEntries(this._matNameToTexFile));

    // Collect unique texture files to load
    const filesToLoad = new Set(this._matNameToTexFile.values());
    console.log('[MuJoCo Viewer] Texture files to load:', Array.from(filesToLoad));

    // Load each texture image via Three.js
    const loadPromises = [];
    for (const memfsPath of filesToLoad) {
      // Convert MEMFS path to URL: strip "/working/" prefix, prepend sceneDir
      const relPath = memfsPath.replace(/^\/working\//, '');
      const url = sceneDir + relPath;
      console.log(`[MuJoCo Viewer] Loading texture: ${url}`);

      const promise = new Promise((resolve) => {
        this.textureLoader.load(
          url,
          (texture) => {
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.colorSpace = THREE.SRGBColorSpace;
            this.textures[memfsPath] = texture;
            console.log(`[MuJoCo Viewer] ✓ Loaded texture: ${url}`);
            resolve();
          },
          undefined,
          (err) => {
            console.warn(`[MuJoCo Viewer] ✗ Failed to load texture ${url}:`, err);
            resolve();
          }
        );
      });
      loadPromises.push(promise);
    }

    await Promise.all(loadPromises);
    console.log('[MuJoCo Viewer] Texture loading complete. Loaded:', Object.keys(this.textures).length);
  }

  // ── Three.js Scene ─────────────────────────────────────

  _buildScene() {
    this.onStatus('Building 3D scene…');

    const w = this.container.clientWidth;
    const h = this.container.clientHeight;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();
    const bg = new THREE.Color(this.bgColor);
    this.scene.background = bg;
    this.scene.fog = new THREE.Fog(bg, 4, 10);

    // Camera — auto-frame from model stat if no position given
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    const stat = this.mjModel.stat;
    const center = new THREE.Vector3(stat.center[0], stat.center[2], -stat.center[1]);
    const extent = stat.extent || 1.0;

    if (this.initialCameraPos) {
      this.camera.position.set(...this.initialCameraPos);
    } else {
      this.camera.position.set(
        center.x + extent * 1.2,
        center.y + extent * 0.8,
        center.z + extent * 1.2
      );
    }
    const target = this.initialCameraTarget
      ? new THREE.Vector3(...this.initialCameraTarget)
      : center.clone();

    this.camera.lookAt(target);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.minDistance = extent * 0.2;
    this.controls.maxDistance = extent * 6;
    this.controls.update();

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(2, 4, 3);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    const shadowRange = extent * 2;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far  = extent * 10;
    dirLight.shadow.camera.left   = -shadowRange;
    dirLight.shadow.camera.right  =  shadowRange;
    dirLight.shadow.camera.top    =  shadowRange;
    dirLight.shadow.camera.bottom = -shadowRange;
    this.scene.add(dirLight);
    this.scene.add(new THREE.DirectionalLight(0xffffff, 0.3).translateX(-2).translateY(1).translateZ(-1));

    // Ground
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.ShadowMaterial({ opacity: 0.12 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const gridSize = Math.max(4, Math.ceil(extent * 4));
    const grid = new THREE.GridHelper(gridSize, gridSize, 0xcccccc, 0xe0e0e0);
    grid.position.y = 0.001;
    this.scene.add(grid);

    // Drag arrow
    if (this.enableDrag) {
      const arrowMat = new THREE.MeshPhysicalMaterial({
        color: 0xff4444, roughness: 0.4, transparent: true, opacity: 0.9,
      });
      const shaftGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 12);
      shaftGeo.translate(0, 0.5, 0);
      const shaft = new THREE.Mesh(shaftGeo, arrowMat); shaft.name = 'shaft';
      const headGeo = new THREE.ConeGeometry(0.022, 0.06, 16);
      headGeo.translate(0, 0.03, 0);
      const head = new THREE.Mesh(headGeo, arrowMat); head.name = 'head';
      this.dragArrow = new THREE.Group();
      this.dragArrow.add(shaft);
      this.dragArrow.add(head);
      this.dragArrow.visible = false;
      this.scene.add(this.dragArrow);
    }

    // Build MuJoCo bodies
    this._buildBodies();
  }

  // ── Build Bodies from mjModel ──────────────────────────

  _buildBodies() {
    const meshGeometries = {};

    this.mujocoRoot = new THREE.Group();
    this.mujocoRoot.name = 'MuJoCo Root';
    this.scene.add(this.mujocoRoot);

    for (let g = 0; g < this.mjModel.ngeom; g++) {
      if (!(this.mjModel.geom_group[g] < 3)) continue;

      const b    = this.mjModel.geom_bodyid[g];
      const type = this.mjModel.geom_type[g];
      const size = [
        this.mjModel.geom_size[g*3+0],
        this.mjModel.geom_size[g*3+1],
        this.mjModel.geom_size[g*3+2],
      ];

      if (!(b in this.bodies)) {
        this.bodies[b] = new THREE.Group();
        this.bodies[b].bodyID = b;
        this.bodies[b].has_custom_mesh = false;
      }

      let geometry = new THREE.SphereGeometry(size[0] * 0.5);

      if (type === 0) {
        // Plane — skip (we have our own ground)
        continue;
      } else if (type === 2) {
        geometry = new THREE.SphereGeometry(size[0]);
      } else if (type === 3) {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 20, 20);
      } else if (type === 5) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2);
      } else if (type === 6) {
        geometry = new THREE.BoxGeometry(size[0]*2, size[2]*2, size[1]*2);
      } else if (type === 7) {
        const meshID = this.mjModel.geom_dataid[g];
        if (meshID < 0) continue;
        if (!(meshID in meshGeometries)) {
          geometry = new THREE.BufferGeometry();
          const vertBuf = new Float32Array(this.mjModel.mesh_vert.subarray(
            this.mjModel.mesh_vertadr[meshID]*3,
            (this.mjModel.mesh_vertadr[meshID]+this.mjModel.mesh_vertnum[meshID])*3));
          for (let v = 0; v < vertBuf.length; v += 3) {
            const tmp = vertBuf[v+1]; vertBuf[v+1] = vertBuf[v+2]; vertBuf[v+2] = -tmp;
          }
          let normBuf;
          if (this.mjModel.mesh_normaladr && this.mjModel.mesh_normalnum) {
            normBuf = new Float32Array(this.mjModel.mesh_normal.subarray(
              this.mjModel.mesh_normaladr[meshID]*3,
              (this.mjModel.mesh_normaladr[meshID]+this.mjModel.mesh_normalnum[meshID])*3));
            for (let v = 0; v < normBuf.length; v += 3) {
              const tmp = normBuf[v+1]; normBuf[v+1] = normBuf[v+2]; normBuf[v+2] = -tmp;
            }
          }
          const faceBuf = new Int32Array(this.mjModel.mesh_face.subarray(
            this.mjModel.mesh_faceadr[meshID]*3,
            (this.mjModel.mesh_faceadr[meshID]+this.mjModel.mesh_facenum[meshID])*3));
          geometry.setAttribute('position', new THREE.BufferAttribute(vertBuf, 3));
          if (normBuf && normBuf.length === vertBuf.length) {
            geometry.setAttribute('normal', new THREE.BufferAttribute(normBuf, 3));
          }
          // UV coordinates
          if (this.mjModel.mesh_texcoord && this.mjModel.mesh_texcoordadr) {
            const texcoordAdr = this.mjModel.mesh_texcoordadr[meshID];
            if (texcoordAdr >= 0) {
              // texcoord count matches vertex count in MuJoCo
              const nVerts = this.mjModel.mesh_vertnum[meshID];
              const uvBuf = new Float32Array(this.mjModel.mesh_texcoord.subarray(
                texcoordAdr * 2, (texcoordAdr + nVerts) * 2));
              // Flip V coordinate (MuJoCo uses bottom-left origin, Three.js uses top-left)
              for (let u = 1; u < uvBuf.length; u += 2) {
                uvBuf[u] = 1.0 - uvBuf[u];
              }
              geometry.setAttribute('uv', new THREE.BufferAttribute(uvBuf, 2));
            }
          }
          geometry.setIndex(Array.from(faceBuf));
          geometry.computeVertexNormals();
          meshGeometries[meshID] = geometry;
        } else {
          geometry = meshGeometries[meshID].clone();
        }
        this.bodies[b].has_custom_mesh = true;
      } else {
        continue;
      }

      // Material
      let color;
      let matId = -1;
      
      if (this.mjModel.geom_matid[g] !== -1) {
        matId = this.mjModel.geom_matid[g];
        color = [
          this.mjModel.mat_rgba[matId*4], this.mjModel.mat_rgba[matId*4+1],
          this.mjModel.mat_rgba[matId*4+2], this.mjModel.mat_rgba[matId*4+3],
        ];
      } else {
        color = [
          this.mjModel.geom_rgba[g*4], this.mjModel.geom_rgba[g*4+1],
          this.mjModel.geom_rgba[g*4+2], this.mjModel.geom_rgba[g*4+3],
        ];
      }

      // Get texture if material has one
      let texture = null;
      if (matId >= 0) {
        // Read material name
        const matNameAddr = this.mjModel.name_matadr ? this.mjModel.name_matadr[matId] : -1;
        let matName = '';
        if (matNameAddr >= 0) {
          for (let j = matNameAddr; this.mjModel.names[j] !== 0; j++) {
            matName += String.fromCharCode(this.mjModel.names[j]);
          }
        }

        // Look up via XML-parsed mapping: material name → MEMFS file path → loaded texture
        if (this._matNameToTexFile && matName) {
          const texFile = this._matNameToTexFile.get(matName);
          if (texFile && this.textures[texFile]) {
            texture = this.textures[texFile];
          }
        }
      }

      const materialParams = {
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3],
        roughness: 0.45,
        metalness: 0.1,
        clearcoat: 0.05,
      };
      
      if (texture) {
        materialParams.map = texture;
        // Use white base color so texture isn't darkened by color multiplication
        materialParams.color = new THREE.Color(1, 1, 1);
      }

      const material = new THREE.MeshPhysicalMaterial(materialParams);

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.bodyID = b;
      this.bodies[b].add(mesh);
      this.bodyMeshes.push(mesh);

      getPosition(this.mjModel.geom_pos, g, mesh.position);
      getQuaternion(this.mjModel.geom_quat, g, mesh.quaternion);
    }

    // Hierarchy
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (!(b in this.bodies)) {
        this.bodies[b] = new THREE.Group();
        this.bodies[b].bodyID = b;
        this.bodies[b].has_custom_mesh = false;
      }
      if (b === 0) this.mujocoRoot.add(this.bodies[b]);
      else         this.bodies[0].add(this.bodies[b]);
    }
    this.syncBodies();
  }

  // ── Sync pose ──────────────────────────────────────────

  syncBodies() {
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (!(b in this.bodies)) continue;
      getPosition(this.mjData.xpos, b, this.bodies[b].position);
      getQuaternion(this.mjData.xquat, b, this.bodies[b].quaternion);
    }
  }

  // ── Drag interaction ───────────────────────────────────

  _onPointerDown(e) {
    if (!this.enableDrag || e.button !== 0) return;
    // In trajectory mode, drag only works in 'step' (physics) mode
    if (this._traj && this._trajMode !== 'step') return;

    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.bodyMeshes, false);

    for (const hit of hits) {
      const bid = hit.object.bodyID;
      if (bid !== undefined && bid > 0) {
        this.dragging   = true;
        this.dragBodyId = bid;
        this.dragPointWorld.copy(hit.point);
        this.dragCursorWorld.copy(hit.point);
        this.dragPointLocal.copy(hit.point);
        this.bodies[bid].worldToLocal(this.dragPointLocal);

        const camDir = new THREE.Vector3();
        this.camera.getWorldDirection(camDir);
        this.dragPlane.setFromNormalAndCoplanarPoint(camDir, hit.point);

        // Block this pointerdown from reaching OrbitControls entirely.
        // OrbitControls never tracks this pointer → no stale-pointer crash
        // on pointerup.
        e.stopImmediatePropagation();
        // Capture the pointer so all events route to the canvas even if
        // the cursor leaves the iframe / canvas bounds.
        if (e.pointerId != null) {
          this.renderer.domElement.setPointerCapture(e.pointerId);
          this._dragPointerId = e.pointerId;
        }
        return;
      }
    }
  }

  _onPointerMove(e) {
    if (!this.dragging) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

    this._updateDragAnchor();
    const camDir = new THREE.Vector3();
    this.camera.getWorldDirection(camDir);
    this.dragPlane.setFromNormalAndCoplanarPoint(camDir, this.dragPointWorld);

    this.raycaster.setFromCamera(this.pointer, this.camera);
    const target = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(this.dragPlane, target)) return;
    this.dragCursorWorld.copy(target);
    this._applyDragForce();
  }

  _onPointerUp(e) {
    if (!this.dragging) return;
    this.dragging = false;
    this.dragBodyId = -1;
    if (this.dragArrow) this.dragArrow.visible = false;
    for (let i = 0; i < this.mjData.qfrc_applied.length; i++) {
      this.mjData.qfrc_applied[i] = 0;
    }
    // Release pointer capture
    if (this._dragPointerId != null) {
      try { this.renderer.domElement.releasePointerCapture(this._dragPointerId); } catch(_) {}
      this._dragPointerId = null;
    }
  }

  _updateDragAnchor() {
    if (!this.dragging) return;
    this.dragPointWorld.copy(this.dragPointLocal);
    this.bodies[this.dragBodyId].localToWorld(this.dragPointWorld);
  }

  _applyDragForce() {
    if (!this.dragging) return;

    const disp = this.dragCursorWorld.clone().sub(this.dragPointWorld);
    const len = disp.length();

    if (this.dragArrow) {
      if (len > 0.001) {
        const dir = disp.clone().normalize();
        this.dragArrow.position.copy(this.dragPointWorld);
        const quat = new THREE.Quaternion();
        quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        this.dragArrow.quaternion.copy(quat);
        const shaft = this.dragArrow.getObjectByName('shaft');
        const head  = this.dragArrow.getObjectByName('head');
        const arrowLen = Math.min(len, 0.5);
        const headLen  = Math.min(arrowLen * 0.35, 0.06);
        const shaftLen = arrowLen - headLen;
        shaft.scale.set(1, Math.max(shaftLen, 0.001), 1);
        head.position.set(0, shaftLen, 0);
        this.dragArrow.visible = true;
      } else {
        this.dragArrow.visible = false;
      }
    }

    for (let i = 0; i < this.mjData.qfrc_applied.length; i++) {
      this.mjData.qfrc_applied[i] = 0;
    }
    const mass = this.mjModel.body_mass[this.dragBodyId];
    const forceVec = toMujocoPos(disp.clone().multiplyScalar(mass * 250));
    const pointVec = toMujocoPos(this.dragPointWorld.clone());
    this.mujoco.mj_applyFT(
      this.mjModel, this.mjData,
      new Float64Array([forceVec.x, forceVec.y, forceVec.z]),
      new Float64Array([0, 0, 0]),
      new Float64Array([pointVec.x, pointVec.y, pointVec.z]),
      this.dragBodyId,
      this.mjData.qfrc_applied
    );
  }

  // ── Events ─────────────────────────────────────────────

  _bindEvents() {
    const el = this.renderer.domElement;
    this._pdown = (e) => this._onPointerDown(e);
    this._pmove = (e) => this._onPointerMove(e);
    this._pup   = (e) => this._onPointerUp(e);
    // Capture phase so our handler fires BEFORE OrbitControls' handler.
    // If we start a drag, stopImmediatePropagation blocks OrbitControls
    // from ever seeing the pointerdown.
    el.addEventListener('pointerdown', this._pdown, true);
    // Listen on window so drag works even when cursor leaves the canvas
    window.addEventListener('pointermove', this._pmove);
    window.addEventListener('pointerup', this._pup);
    window.addEventListener('blur', this._pup);
    window.addEventListener('resize', () => this._onResize());

    // Touch
    el.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._onPointerDown({ button: 0, clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
      }
    }, { passive: true });
    el.addEventListener('touchmove', (e) => {
      if (this.dragging && e.touches.length === 1) {
        e.preventDefault();
        this._onPointerMove({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
      }
    }, { passive: false });
    el.addEventListener('touchend', (e) => this._onPointerUp(e));
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (this.renderer) {
      this.renderer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  // ── Animation loop ─────────────────────────────────────

  /**
   * Override this to run custom control (PD, RL, etc.) each sub-step.
   * Called BETWEEN mj_step1 and mj_step2.
   * @param {Object} mjModel
   * @param {Object} mjData
   */
  onControlStep(mjModel, mjData) {
    // no-op by default; override in subclass or set viewer.onControlStep = fn
  }

  _animate(time) {
    if (this._disposed || this._frozen) return;
    requestAnimationFrame(this._animate);

    if (this.paused) {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this._lastTime = time;
      return;
    }

    const dt = Math.min(time - this._lastTime, 50);
    this._lastTime = time;

    // ── Trajectory playback mode ──────────────────────────
    if (this._traj) {
      this._stepTrajectory(dt);
      this.syncBodies();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // ── Normal physics mode ───────────────────────────────
    const timestepMs = this.mjModel.opt.timestep * 1000;
    const nsteps = Math.floor(dt / timestepMs);

    for (let i = 0; i < Math.min(nsteps, 30); i++) {
      if (this.dragging) {
        this.syncBodies();
        this._updateDragAnchor();
        this._applyDragForce();
      } else {
        for (let j = 0; j < this.mjData.qfrc_applied.length; j++) {
          this.mjData.qfrc_applied[j] = 0;
        }
      }

      this.mujoco.mj_step1(this.mjModel, this.mjData);
      this.onControlStep(this.mjModel, this.mjData);
      this.mujoco.mj_step2(this.mjModel, this.mjData);
    }

    this.syncBodies();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
