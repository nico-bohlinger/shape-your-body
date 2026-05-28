// PolicyRunner — drives a URMA policy inside the MuJoCo viewer's
// per-physics-step callback.  Builds joint + general observations from mjData,
// runs the policy, and writes mjData.ctrl.  Decimates from physics rate
// (~200 Hz) to control rate (50 Hz).

export class PolicyRunner {
    constructor() {
        this.policy = null;
        this.meta = null;
        this.command = { vx: 0, vy: 0, vyaw: 0 };
        this.subStep = 0;
        this.controlDecimation = 4;   // 200 Hz physics / 50 Hz control
        this.phase = [0.0, -Math.PI];
        this.phaseDt = 0;
        this.prevAction = null;
        this.jointState = null;
        this.jointDescFlat = null;
        this.generalState = null;
        this.action = null;
    }

    attach(policy, meta) {
        this.policy = policy;
        this.meta = meta;

        const n = meta.n_joints;
        this.prevAction = new Float32Array(n);
        this.jointState = new Float32Array(n * 4);
        this.generalState = new Float32Array(27);
        this.action = new Float32Array(n);

        const desc = meta.joint_descriptions;
        const dDesc = meta.input_dims.joint_description;
        this.jointDescFlat = new Float32Array(n * dDesc);
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dDesc; d++) {
                this.jointDescFlat[i * dDesc + d] = desc[i][d];
            }
        }

        this.phase = meta.gait.phase_offset.slice();
        const gaitFreq = 1.0 / meta.gait.period;
        this.phaseDt = 2 * Math.PI * meta.control_dt * gaitFreq;
        this.subStep = 0;
        this.prevAction.fill(0);
    }

    setCommand(vx, vy, vyaw) {
        this.command.vx = vx;
        this.command.vy = vy;
        this.command.vyaw = vyaw;
    }

    onPhysicsStep(mjModel, mjData) {
        if (!this.policy || !this.meta) return;
        this.subStep++;
        if (this.subStep < this.controlDecimation) return;
        this.subStep = 0;

        const meta = this.meta;
        const n = meta.n_joints;
        const norm = meta.observation_normalization;

        // Build joint observations:  [pos-nom, vel, prev_action, keep_nominal]
        for (let i = 0; i < n; i++) {
            const qpos = mjData.qpos[meta.actuator_qpos_idx[i]];
            const qvel = mjData.qvel[meta.actuator_qvel_idx[i]];
            const off = i * 4;
            this.jointState[off + 0] = (qpos - meta.nominal_qpos[i]) / norm.joint_pos;
            this.jointState[off + 1] = qvel / norm.joint_vel;
            this.jointState[off + 2] = this.prevAction[i] / norm.prev_action;
            this.jointState[off + 3] = meta.keep_nominal[i];
        }

        // Advance gait phase and compute phase features  (sin/cos of phase + dt)
        const p0 = wrapPi(this.phase[0] + this.phaseDt);
        const p1 = wrapPi(this.phase[1] + this.phaseDt);
        this.phase[0] = p0;
        this.phase[1] = p1;

        // IMU angular velocity (from mjData.sensordata)
        const imu = meta.imu_angular_velocity;
        const sensorData = mjData.sensordata;
        let imuX = sensorData[imu.adr + 0];
        let imuY = sensorData[imu.adr + 1];
        let imuZ = sensorData[imu.adr + 2];
        const imuScale = 1 / norm.imu_ang_vel;
        imuX = clamp(imuX * imuScale, -1, 1);
        imuY = clamp(imuY * imuScale, -1, 1);
        imuZ = clamp(imuZ * imuScale, -1, 1);

        // Projected gravity in body frame: v_body = R^T * (0,0,-1) where R is the
        // body-to-world rotation matrix from the trunk freejoint quaternion at qpos[3..7] (w, x, y, z).
        const qw = mjData.qpos[3], qx = mjData.qpos[4], qy = mjData.qpos[5], qz = mjData.qpos[6];
        const gx =  2 * (qw * qy - qx * qz);
        const gy = -2 * (qy * qz + qw * qx);
        const gz = -1 + 2 * (qx * qx + qy * qy);

        // Pack generalState (27): cat([imu_ang_vel(3), goal_vel(3), phase_feat(4), proj_gravity(3), gains_scale(3), mass(1), dims(3), trunk_attrs(7)])
        const gs = this.generalState;
        gs[0] = imuX;  gs[1] = imuY;  gs[2] = imuZ;
        gs[3] = this.command.vx;  gs[4] = this.command.vy;  gs[5] = this.command.vyaw;
        gs[6] = Math.sin(p0);  gs[7] = Math.sin(p1);
        gs[8] = Math.cos(p0);  gs[9] = Math.cos(p1);
        gs[10] = gx;  gs[11] = gy;  gs[12] = gz;
        const stat = meta.general_static;
        gs[13] = stat.gains_and_scale_normalized[0];
        gs[14] = stat.gains_and_scale_normalized[1];
        gs[15] = stat.gains_and_scale_normalized[2];
        gs[16] = stat.mass_normalized[0];
        gs[17] = stat.robot_dimensions_normalized[0];
        gs[18] = stat.robot_dimensions_normalized[1];
        gs[19] = stat.robot_dimensions_normalized[2];
        gs[20] = stat.trunk_attributes_normalized[0];
        gs[21] = stat.trunk_attributes_normalized[1];
        gs[22] = stat.trunk_attributes_normalized[2];
        gs[23] = stat.trunk_attributes_normalized[3];
        gs[24] = stat.trunk_attributes_normalized[4];
        gs[25] = stat.trunk_attributes_normalized[5];
        gs[26] = stat.trunk_attributes_normalized[6];

        // Forward pass
        this.policy.forward(this.jointDescFlat, this.jointState, gs, n, this.action);

        // Apply: ctrl[i] = nominal_qpos[i] + scaling_factor * action[i]
        const sigma = meta.scaling_factor;
        for (let i = 0; i < n; i++) {
            mjData.ctrl[i] = meta.nominal_qpos[i] + sigma * this.action[i];
            this.prevAction[i] = this.action[i];
        }
    }
}

function wrapPi(x) {
    return ((x + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function clamp(x, lo, hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
