import { MuJoCoViewer } from './mujoco_viewer.js';
import { loadPolicyBundle } from './urma_policy.js';
import { PolicyRunner } from './policy_runner.js';

const ITER_COUNT = 51;
const SEED_COUNT = 10;
const DEMO_BACKGROUND = '#6aaee1';

function sceneEntry(robotDir, mode, seed, iter) {
    const base = `demo/scenes/${robotDir}`;
    if (mode === 'reference') {
        return { xml: `${base}/reference.xml`, meta: `${base}/reference.json` };
    }
    const tag = `s${seed}_iter_${String(iter).padStart(2, '0')}`;
    return { xml: `${base}/${tag}.xml`, meta: `${base}/${tag}.json` };
}

const ROBOTS = {
    unitree_go2: {
        name: 'Unitree Go2',
        dir: 'unitree_go2',
        policyId: 'unitree_go2',
        cameraPos: [2.05, 0.88, 2.25],
        cameraTarget: [0, 0.25, 0],
    },
    mit_humanoid: {
        name: 'MIT Humanoid',
        dir: 'mit_humanoid',
        policyId: 'mit_humanoid',
        cameraPos: [2.75, 1.5, 3.05],
        cameraTarget: [0, 0.45, 0],
    },
    golem: {
        name: 'Golem',
        dir: 'golem',
        policyId: 'golem',
        cameraPos: [2.05, 0.78, 2.25],
        cameraTarget: [0, 0.1, 0],
    },
    anymal_c: {
        name: 'ANYmal C',
        dir: 'anymal_c',
        policyId: 'all50',
        cameraPos: [2.5, 1.1, 2.7],
        cameraTarget: [0, 0.4, 0],
    },
    booster_t1: {
        name: 'Booster T1',
        dir: 'booster_t1',
        policyId: 'all50',
        cameraPos: [2.7, 1.4, 3.0],
        cameraTarget: [0, 0.5, 0],
    },
    mini_pi: {
        name: 'Mini PI',
        dir: 'mini_pi',
        policyId: 'all50',
        cameraPos: [1.5, 0.7, 1.65],
        cameraTarget: [0, 0.2, 0],
    },
    fourier_gr1t2: {
        name: 'Fourier GR1-T2',
        dir: 'fourier_gr1t2',
        policyId: 'all50',
        cameraPos: [3.4, 1.9, 3.7],
        cameraTarget: [0, 0.7, 0],
    },
};

const state = {
    robot: 'mit_humanoid',
    mode: 'codesign',            // 'reference' | 'codesign'
    seed: 4,                     // 0..9
    iter: 0,                     // 0..50
    playing: true,
    follow: true,
};
let viewer = null;
let loadToken = 0;
let iterDebounceTimer = null;
let lastTrunk = null;            // [tx, ty, tz] in Three.js coords, for camera-follow delta

const runner = new PolicyRunner();
const policyCache = new Map();

const containerEl = document.getElementById('syb-canvas');
const statusEl = document.getElementById('syb-status');
const robotTabsEl = document.getElementById('syb-robot-tabs');
const modeTabsEl = document.getElementById('syb-mode-tabs');
const seedTabsEl = document.getElementById('syb-seed-tabs');
const iterInputEl = document.getElementById('syb-iter');
const iterValueEl = document.getElementById('syb-iter-value');
const cmdVxEl = document.getElementById('syb-vx');
const cmdVyEl = document.getElementById('syb-vy');
const cmdVyawEl = document.getElementById('syb-vyaw');
const cmdVxValEl = document.getElementById('syb-vx-val');
const cmdVyValEl = document.getElementById('syb-vy-val');
const cmdVyawValEl = document.getElementById('syb-vyaw-val');
const playBtnEl = document.getElementById('syb-play');
const resetBtnEl = document.getElementById('syb-reset');
const followBtnEl = document.getElementById('syb-follow');

function applyHomeKeyframe(mujoco, mjModel, mjData, meta) {
    // Zero dynamic state in-place (do NOT call mj_resetData — it also clears
    // the constraint-solver warmstart and the kinematics).  Then write the
    // keyframe pose and seed ctrl with the nominal joint targets.
    const nkey = mjModel.nkey | 0;
    const keyQpos = mjModel.key_qpos;
    const nq = mjModel.nq | 0;
    if (nkey > 0 && keyQpos) {
        for (let i = 0; i < nq; i++) mjData.qpos[i] = keyQpos[i];
        if (mjData.ctrl && mjData.ctrl.length) {
            for (let i = 0; i < mjData.ctrl.length; i++) {
                mjData.ctrl[i] = keyQpos[7 + i] ?? 0;
            }
        }
    }
    // Per-design correction: shift trunk z so the lowest foot rests on the floor.
    // The XML's home keyframe was authored for the nominal embodiment; perturbed
    // designs change body lengths, so trunk z must adapt or the robot either
    // penetrates the floor (spring up) or hovers (fall down).
    if (meta && typeof meta.home_qpos_z === 'number') {
        mjData.qpos[2] = meta.home_qpos_z;
    }
    const nv = mjModel.nv | 0;
    if (mjData.qvel)         for (let i = 0; i < nv; i++) mjData.qvel[i] = 0;
    if (mjData.qacc)         for (let i = 0; i < nv; i++) mjData.qacc[i] = 0;
    if (mjData.qfrc_applied) for (let i = 0; i < nv; i++) mjData.qfrc_applied[i] = 0;
    if (mjData.act)          for (let i = 0; i < mjData.act.length; i++) mjData.act[i] = 0;
    mjData.time = 0;
    mujoco.mj_forward(mjModel, mjData);
}

async function ensurePolicy(robot) {
    const policyId = ROBOTS[robot].policyId;
    if (policyCache.has(policyId)) return policyCache.get(policyId);
    statusEl.textContent = `Loading policy for ${ROBOTS[robot].name}…`;
    const p = await loadPolicyBundle(policyId, 'demo/weights');
    policyCache.set(policyId, p);
    return p;
}

async function loadCurrent() {
    const myToken = ++loadToken;
    const robot = ROBOTS[state.robot];
    const entry = sceneEntry(robot.dir, state.mode, state.seed, state.iter);

    if (viewer) {
        viewer.dispose();
        viewer = null;
    }
    runner.policy = null;
    runner.meta = null;
    lastTrunk = null;
    containerEl.innerHTML = '';
    statusEl.style.display = '';
    statusEl.textContent = `Loading ${robot.name}…`;

    try {
        const [policy, metaResp] = await Promise.all([
            ensurePolicy(state.robot),
            fetch(entry.meta),
        ]);
        if (myToken !== loadToken) return;
        if (!metaResp.ok) throw new Error(`Failed to fetch ${entry.meta}: ${metaResp.status}`);
        const meta = await metaResp.json();
        runner.attach(policy, meta);

        const v = new MuJoCoViewer(containerEl, {
            sceneXML: entry.xml,
            wasmURL: './dist/mujoco_wasm.js',
            background: DEMO_BACKGROUND,
            cameraPos: robot.cameraPos,
            cameraTarget: robot.cameraTarget,
            enableDrag: false,
            paused: !state.playing,
            onStatus: (s) => {
                if (myToken !== loadToken) return;
                statusEl.textContent = s;
            },
            onReady: (mjModel, mjData, mujoco) => {
                if (myToken !== loadToken) return;
                applyHomeKeyframe(mujoco, mjModel, mjData, meta);
                runner.setCommand(
                    Number(cmdVxEl.value),
                    Number(cmdVyEl.value),
                    Number(cmdVyawEl.value),
                );
                statusEl.style.display = 'none';
            },
        });

        v.onControlStep = (mjModel, mjData) => {
            const trunkZ = mjData.qpos[2];
            if (!Number.isFinite(trunkZ) || trunkZ < 0) {
                resetPose();
                return;
            }
            runner.onPhysicsStep(mjModel, mjData);
            if (state.follow) {
                // MuJoCo (x, y, z up) -> Three.js (x, z, -y) (y up)
                const tx = mjData.qpos[0];
                const ty = mjData.qpos[2];
                const tz = -mjData.qpos[1];
                if (lastTrunk === null) {
                    // First frame after a (re)load: snap the orbit target onto
                    // the trunk and translate the camera by the same delta so
                    // the camera's relative offset to the robot is preserved.
                    const dx = tx - v.controls.target.x;
                    const dy = ty - v.controls.target.y;
                    const dz = tz - v.controls.target.z;
                    v.camera.position.x += dx;
                    v.camera.position.y += dy;
                    v.camera.position.z += dz;
                    v.controls.target.set(tx, ty, tz);
                    lastTrunk = [tx, ty, tz];
                } else {
                    // Exponential moving average: lastTrunk follows the trunk with
                    // an ~80 ms time constant at 200 Hz physics rate.  Camera
                    // translates by the same delta so the offset stays constant.
                    const alpha = 0.06;
                    const dx = (tx - lastTrunk[0]) * alpha;
                    const dy = (ty - lastTrunk[1]) * alpha;
                    const dz = (tz - lastTrunk[2]) * alpha;
                    lastTrunk[0] += dx;
                    lastTrunk[1] += dy;
                    lastTrunk[2] += dz;
                    v.camera.position.x += dx;
                    v.camera.position.y += dy;
                    v.camera.position.z += dz;
                    v.controls.target.set(lastTrunk[0], lastTrunk[1], lastTrunk[2]);
                }
            }
        };

        await v.init();
        if (myToken !== loadToken) {
            v.dispose();
            return;
        }
        viewer = v;
    } catch (err) {
        if (myToken !== loadToken) return;
        console.error('[syb-demo] load failed:', err);
        statusEl.style.display = '';
        const desc = (err && (err.message || (err.toString && err.toString()))) || String(err);
        statusEl.textContent = `Failed to load ${robot.name}: ${desc}`;
    }
}

function setRobot(robot) {
    if (!ROBOTS[robot] || state.robot === robot) return;
    state.robot = robot;
    refreshTabs();
    loadCurrent();
}

function setMode(mode) {
    if (mode !== 'reference' && mode !== 'codesign') return;
    if (state.mode === mode) return;
    state.mode = mode;
    refreshTabs();
    loadCurrent();
}

function setSeed(seed) {
    seed = Number(seed);
    if (!Number.isInteger(seed) || seed < 0 || seed >= SEED_COUNT) return;
    if (state.seed === seed) return;
    state.seed = seed;
    refreshTabs();
    if (state.mode === 'codesign') loadCurrent();
}

function refreshTabs() {
    for (const btn of robotTabsEl.querySelectorAll('button')) {
        btn.classList.toggle('active', btn.dataset.robot === state.robot);
    }
    for (const btn of modeTabsEl.querySelectorAll('button')) {
        btn.classList.toggle('active', btn.dataset.mode === state.mode);
    }
    const codesignActive = state.mode === 'codesign';
    for (const btn of seedTabsEl.querySelectorAll('button')) {
        btn.classList.toggle('active', codesignActive && Number(btn.dataset.seed) === state.seed);
        btn.disabled = !codesignActive;
    }
    iterInputEl.disabled = !codesignActive;
    iterInputEl.max = String(ITER_COUNT - 1);
    iterValueEl.textContent = codesignActive
        ? `iter ${state.iter} / ${ITER_COUNT - 1}`
        : 'reference (slider disabled)';

    playBtnEl.classList.toggle('active', !state.playing);
    followBtnEl.classList.toggle('active', state.follow);
}

function updateCommandFromUI() {
    const vx = Number(cmdVxEl.value);
    const vy = Number(cmdVyEl.value);
    const vyaw = Number(cmdVyawEl.value);
    cmdVxValEl.textContent = vx.toFixed(2);
    cmdVyValEl.textContent = vy.toFixed(2);
    cmdVyawValEl.textContent = vyaw.toFixed(2);
    runner.setCommand(vx, vy, vyaw);
}

robotTabsEl.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-robot]');
    if (btn) setRobot(btn.dataset.robot);
});

modeTabsEl.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-mode]');
    if (btn) setMode(btn.dataset.mode);
});

seedTabsEl.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-seed]');
    if (btn && !btn.disabled) setSeed(btn.dataset.seed);
});

iterInputEl.addEventListener('input', (ev) => {
    if (iterInputEl.disabled) return;
    state.iter = Number(ev.target.value);
    iterValueEl.textContent = `iter ${state.iter} / ${ITER_COUNT - 1}`;
    if (iterDebounceTimer) clearTimeout(iterDebounceTimer);
    iterDebounceTimer = setTimeout(loadCurrent, 80);
});

for (const el of [cmdVxEl, cmdVyEl, cmdVyawEl]) {
    el.addEventListener('input', updateCommandFromUI);
}

function togglePlay() {
    state.playing = !state.playing;
    if (viewer) viewer.paused = !state.playing;
    refreshTabs();
}

playBtnEl.addEventListener('click', togglePlay);

window.addEventListener('keydown', (ev) => {
    if (ev.code !== 'Space') return;
    const tag = (ev.target && ev.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    ev.preventDefault();
    togglePlay();
});

function resetPose() {
    if (!viewer) return;
    applyHomeKeyframe(viewer.mujoco, viewer.mjModel, viewer.mjData, runner ? runner.meta : null);
    if (runner && runner.meta) {
        runner.subStep = 0;
        runner.prevAction.fill(0);
        runner.phase = runner.meta.gait.phase_offset.slice();
    }
    lastTrunk = null;
    viewer.controls.target.set(...ROBOTS[state.robot].cameraTarget);
    viewer.camera.position.set(...ROBOTS[state.robot].cameraPos);
}

resetBtnEl.addEventListener('click', resetPose);

followBtnEl.addEventListener('click', () => {
    state.follow = !state.follow;
    lastTrunk = null;   // camera stays where it is; just stop tracking
    refreshTabs();
});

updateCommandFromUI();
refreshTabs();
loadCurrent();
