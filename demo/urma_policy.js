// URMA policy forward pass — direct port of the Flax module at
// loco_mjx/algorithms/uniform/single_robot_urma/mujoco/policy.py.
// Uses Float32Array views into a single contiguous weights buffer.
// WeightNorm scaling has been merged into the kernels at export time.

export async function loadPolicyBundle(robotName, basePath = './weights') {
    const manifestUrl = `${basePath}/${robotName}.weights.json`;
    const binUrl = `${basePath}/${robotName}.weights.bin`;
    const [manifestResp, binResp] = await Promise.all([
        fetch(manifestUrl),
        fetch(binUrl),
    ]);
    if (!manifestResp.ok) throw new Error(`Failed to fetch ${manifestUrl}: ${manifestResp.status}`);
    if (!binResp.ok) throw new Error(`Failed to fetch ${binUrl}: ${binResp.status}`);
    const manifest = await manifestResp.json();
    const buffer = await binResp.arrayBuffer();
    const weights = {};
    for (const layer of manifest.layers) {
        const flat = new Float32Array(buffer, layer.offset, layer.size_bytes / 4);
        weights[layer.name] = { data: flat, shape: layer.shape };
    }
    return new URMAPolicy(weights, manifest.hyperparams);
}

export class URMAPolicy {
    constructor(weights, hyperparams) {
        this.w = weights;
        this.softmax_temperature_min = hyperparams.softmax_temperature_min;
        this.stability_epsilon = hyperparams.stability_epsilon;
        this.policy_mean_abs_clip = hyperparams.policy_mean_abs_clip;
        const temp = this.w.joint_log_softmax_temperature.data[0];
        this.softmax_temperature = Math.exp(temp) + this.softmax_temperature_min;
        this._scratch = {};
    }

    scratch(name, size) {
        let s = this._scratch[name];
        if (!s || s.length < size) {
            s = new Float32Array(size);
            this._scratch[name] = s;
        }
        return s.subarray(0, size);
    }

    // out[i,k] = sum_j x[i,j] * W[j,k] + b[k]  ; x:[N,J], W:[J,K], b:[K]
    denseBatch(out, x, N, W, b, J, K) {
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < K; k++) {
                let s = b[k];
                const xRow = i * J;
                for (let j = 0; j < J; j++) s += x[xRow + j] * W[j * K + k];
                out[i * K + k] = s;
            }
        }
    }

    denseSingle(out, x, W, b, J, K) {
        for (let k = 0; k < K; k++) {
            let s = b[k];
            for (let j = 0; j < J; j++) s += x[j] * W[j * K + k];
            out[k] = s;
        }
    }

    elu(x) {
        for (let i = 0; i < x.length; i++) {
            const v = x[i];
            if (v < 0) x[i] = Math.exp(v) - 1;
        }
    }

    tanhClip(x, eps) {
        const lo = -1 + eps, hi = 1 - eps;
        for (let i = 0; i < x.length; i++) {
            const t = Math.tanh(x[i]);
            x[i] = t < lo ? lo : (t > hi ? hi : t);
        }
    }

    // Layer norm with per-feature gain `gamma` and bias `beta`, length D.
    layerNormBatch(x, gamma, beta, N, D, eps = 1e-6) {
        for (let i = 0; i < N; i++) {
            const off = i * D;
            let mean = 0;
            for (let d = 0; d < D; d++) mean += x[off + d];
            mean /= D;
            let varSum = 0;
            for (let d = 0; d < D; d++) {
                const c = x[off + d] - mean;
                varSum += c * c;
            }
            const invStd = 1 / Math.sqrt(varSum / D + eps);
            for (let d = 0; d < D; d++) {
                x[off + d] = (x[off + d] - mean) * invStd * gamma[d] + beta[d];
            }
        }
    }

    layerNormSingle(x, gamma, beta, D, eps = 1e-6) {
        let mean = 0;
        for (let d = 0; d < D; d++) mean += x[d];
        mean /= D;
        let varSum = 0;
        for (let d = 0; d < D; d++) {
            const c = x[d] - mean;
            varSum += c * c;
        }
        const invStd = 1 / Math.sqrt(varSum / D + eps);
        for (let d = 0; d < D; d++) x[d] = (x[d] - mean) * invStd * gamma[d] + beta[d];
    }

    // Inputs:
    //   jointDesc        : Float32Array of length nJoints * 30  (joint description, static per design)
    //   jointState       : Float32Array of length nJoints * 4   (per-step joint observations)
    //   generalState     : Float32Array of length 27            (per-step general observations)
    //   nJoints          : number of actuated joints
    // Output: actionOut[nJoints] populated with policy mean (clipped to ±policy_mean_abs_clip).
    forward(jointDesc, jointState, generalState, nJoints, actionOut) {
        const eps = this.stability_epsilon;
        const dDesc = 30, dState = 4;
        const dDescState = dDesc + dState;  // 34

        // Build [jointDesc, jointState] per joint  ([N, 34])
        const concatIn = this.scratch('concatIn', nJoints * dDescState);
        for (let i = 0; i < nJoints; i++) {
            const outOff = i * dDescState;
            const descOff = i * dDesc;
            const stateOff = i * dState;
            for (let d = 0; d < dDesc; d++) concatIn[outOff + d] = jointDesc[descOff + d];
            for (let d = 0; d < dState; d++) concatIn[outOff + dDesc + d] = jointState[stateOff + d];
        }

        // Dense_0: 34 -> 256
        const d0 = this.scratch('d0', nJoints * 256);
        this.denseBatch(d0, concatIn, nJoints, this.w['Dense_0.kernel'].data, this.w['Dense_0.bias'].data, dDescState, 256);
        // LayerNorm_0
        this.layerNormBatch(d0, this.w['LayerNorm_0.scale'].data, this.w['LayerNorm_0.bias'].data, nJoints, 256);
        this.elu(d0);

        // Dense_1: 256 -> 256
        const d1 = this.scratch('d1', nJoints * 256);
        this.denseBatch(d1, d0, nJoints, this.w['Dense_1.kernel'].data, this.w['Dense_1.bias'].data, 256, 256);

        // joint_state_mask (raw) = clip(tanh(d1)), per joint [N, 256]
        this.tanhClip(d1, eps);
        // d1 is now joint_state_mask_raw (before softmax)

        // Dense_2: 4 -> 4 for latent_joint_state
        const dlatent = this.scratch('dlatent', nJoints * 4);
        this.denseBatch(dlatent, jointState, nJoints, this.w['Dense_2.kernel'].data, this.w['Dense_2.bias'].data, 4, 4);
        this.elu(dlatent);

        // softmax-with-temperature over the 256 axis  (per joint, normalize columns)
        const temp = this.softmax_temperature;
        const mask = d1; // shape [N, 256]
        for (let i = 0; i < nJoints; i++) {
            const off = i * 256;
            // Subtract max for numerical stability (the original code doesn't but it's mathematically equivalent here)
            let mx = mask[off];
            for (let d = 1; d < 256; d++) if (mask[off + d] > mx) mx = mask[off + d];
            let sum = 0;
            for (let d = 0; d < 256; d++) {
                const e = Math.exp((mask[off + d] - mx) / temp);
                mask[off + d] = e;
                sum += e;
            }
            const inv = 1 / (sum + eps);
            for (let d = 0; d < 256; d++) mask[off + d] *= inv;
        }
        // mask is now joint_state_mask (softmaxed), per [N, 256]

        // joint_latent[k*4 + l] = sum_i mask[i, k] * latent_joint_state[i, l]
        // Result shape: [256 * 4] = [1024], summed over joints
        const jointLatent = this.scratch('jointLatent', 1024);
        jointLatent.fill(0);
        for (let i = 0; i < nJoints; i++) {
            const maskOff = i * 256;
            const latOff = i * 4;
            for (let k = 0; k < 256; k++) {
                const m = mask[maskOff + k];
                const outOff = k * 4;
                for (let l = 0; l < 4; l++) jointLatent[outOff + l] += m * dlatent[latOff + l];
            }
        }

        // combined = [jointLatent (1024), generalState (27)] -> 1051
        const combined = this.scratch('combined', 1024 + 27);
        combined.set(jointLatent, 0);
        combined.set(generalState, 1024);

        // Dense_3: 1051 -> 512 (WeightNorm merged)
        const a0 = this.scratch('a0', 512);
        this.denseSingle(a0, combined, this.w['Dense_3.kernel'].data, this.w['Dense_3.bias'].data, 1051, 512);
        this.layerNormSingle(a0, this.w['LayerNorm_1.scale'].data, this.w['LayerNorm_1.bias'].data, 512);
        this.elu(a0);

        // Dense_4: 512 -> 256
        const a1 = this.scratch('a1', 256);
        this.denseSingle(a1, a0, this.w['Dense_4.kernel'].data, this.w['Dense_4.bias'].data, 512, 256);
        this.elu(a1);

        // Dense_5: 256 -> 256
        const a2 = this.scratch('a2', 256);
        this.denseSingle(a2, a1, this.w['Dense_5.kernel'].data, this.w['Dense_5.bias'].data, 256, 256);
        this.elu(a2);

        // Dense_6: 256 -> 256
        const a3 = this.scratch('a3', 256);
        this.denseSingle(a3, a2, this.w['Dense_6.kernel'].data, this.w['Dense_6.bias'].data, 256, 256);
        this.elu(a3);

        // Dense_7: 256 -> 256  (no activation, no layernorm)
        const a4 = this.scratch('a4', 256);
        this.denseSingle(a4, a3, this.w['Dense_7.kernel'].data, this.w['Dense_7.bias'].data, 256, 256);

        // policy_mean[i] = sum_k a4[k] * mask[i, k]; then clip
        const clip = this.policy_mean_abs_clip;
        for (let i = 0; i < nJoints; i++) {
            const maskOff = i * 256;
            let s = 0;
            for (let k = 0; k < 256; k++) s += a4[k] * mask[maskOff + k];
            actionOut[i] = s > clip ? clip : (s < -clip ? -clip : s);
        }
    }
}
