# Feedback Loop Structure & Adaptive Parameters
### For Zibo Yuan — Re: Hybrid Image Encoding Integration
**Prepared by: Unubold Tuguldur | Date: 18 Mar 2026**

---

## 1. Overview

The feedback loop runs **per-frame** inside `closed_loop_sim.m`. After each frame is transmitted, received, and reconstructed, the RX measures quality and updates a **policy** that controls the next frame's TX settings. The policy is written to `policy.csv` and read by `PolicyController.m` at the start of each frame.

```
Frame N:  PolicyController → TX Pipeline → Channel → RX Pipeline → Quality Metrics
                ↑                                                         |
                └──── rx_policy_update() ← rx_feedback() ← SSIM/PSNR ──┘
```

---

## 2. The Policy — All Adjustable Parameters

The policy struct (defined in `PolicyController.defaultPolicy()` and stored in `policy.csv`) contains these fields:

### Parameters that the feedback loop ACTIVELY adjusts each frame:

| Parameter    | Type   | Range   | What it controls                                   |
|-------------|--------|---------|---------------------------------------------------|
| `K`         | uint8  | 5–60    | DCT zigzag split point: first K coeffs → base layer, rest → enhancement |
| `Qb_idx`    | uint8  | 0–3*    | Base layer quantization step index (maps to [0.01, 0.02, 0.05, 0.1]) |
| `Qe_idx`    | uint8  | 0–3*    | Enhancement layer quantization step index (same mapping) |
| `mcs_base`  | uint8  | 0–7     | MCS index for base layer (0=BPSK R1/2, 7=64QAM R3/4) |
| `mcs_enh`   | uint8  | 0–7     | MCS index for enhancement layer |

*Note: Qb_idx and Qe_idx are now part of the GA search space (range 0-3, mapped to DCT quantisation lists). MCS is **not** in the GA — rules + LSTM ceiling handle MCS exclusively.

### Parameters that are fixed / rarely changed:

| Parameter        | Type   | Default | What it is                          |
|-----------------|--------|---------|-------------------------------------|
| `tx_mode`       | string | "split" | "combined" (1 packet) or "split" (2 packets) |
| `mux_gap_us`    | double | 0       | Gap in microseconds between base and enh packets |
| `crc_flag`      | uint8  | 1       | Enable CRC in packet header         |
| `endian_flag`   | uint8  | 1       | Big-endian byte order               |
| `codec_id`      | uint8  | 0       | Codec identifier (reserved)         |
| `reserved_flag` | uint8  | 0       | Reserved                            |
| `version`       | uint32 | auto    | Monotonically increasing policy version |

---

## 3. How the Feedback Loop Decides (rx_feedback.m → rx_policy_update.m)

The decision engine has **three layers** (any or all can be active):

### Layer 1: Rule-Based (always active, fallback)
Located in `rx_policy_update.m`:
- **SSIM < 0.5 (urgent):** Drop MCS by 2 steps, increase K by 5
- **SSIM < 0.7:** Drop MCS by 1, increase K by 2
- **SSIM < 0.85:** Drop enhancement MCS by 1
- **SSIM > 0.95 AND targets met:** Raise enhancement MCS by 1, decrease K by 1
- **PSNR < 20:** Decrease Qb/Qe indices
- **PSNR > 40 AND targets met:** Increase Qe index

### Layer 2: GA Optimizer (optional, in ga_parameter_optimizer.m)
- Evolves population of **[K, Qb_idx, Qe_idx]** chromosomes (MCS not in search space — rules handle it)
- Fitness = f(quantisation distortion, compression, SNR reliability, **measured SSIM/enhFailRatio**)
- Result is **blended 70/30** with current params in Step 2 of rx_policy_update.m (after rules run in Step 1)

### Layer 3: Channel Predictor (optional, in channel_predictor.m / bold_channel_lstm.m)
- Predicts next-frame SNR from history
- If predicted SNR drop > 5 dB: pre-emptively lower both MCS
- If predicted drop > 3 dB: lower enhancement MCS only

---

## 4. What I (Bold) Added — ML Enhancement Layer

My additions sit in `ml_addons/` and are prefixed `bold_*`. They **do not modify** any existing files. They plug in via `bold_rx_feedback.m` which wraps the original `rx_feedback.m`:

| Model | File | What it does | Integration point |
|-------|------|-------------|-------------------|
| MLP Quality Predictor | `bold_quality_predictor.m` | Predicts SSIM/PSNR from DCT coefficients **without the original image** | Feeds into feedback when reference isn't available |
| LSTM Channel Predictor | `bold_channel_lstm.m` | Replaces EMA+trend with trained LSTM for SNR prediction | Replaces `channel_predictor.m` when model is trained |
| MLP Symbol Denoiser | `bold_symbol_denoiser.m` | Denoises QAM constellation symbols before hard-decision demapping | Operates at physical layer inside RX |

Supporting infrastructure: `SimpleNet.m` (MLP from scratch), `SimpleLSTM.m` (LSTM with full BPTT from scratch), `bold_train_all.m`, `bold_generate_training_data.m`.

---

## 5. What Zibo Needs for Hybrid Encoding Integration

### If your DWT/wavelet encoder produces similar two-layer output:
The feedback loop will work as-is. Your encoder just needs to output:
- **Base layer coefficients** (protected, robust)
- **Enhancement layer coefficients** (higher detail, less protected)
- A **K-equivalent** parameter that controls the base/enhancement split

The policy parameters `K`, `Qb_idx`, `Qe_idx` directly map to:
- `K` → how many coefficients go in the base layer
- `Qb_idx` → quantization coarseness for base
- `Qe_idx` → quantization coarseness for enhancement

### Parameters you should keep fixed for now:
As agreed in the group chat, **quantization step is fixed** (both Qb and Qe at index 1) in both the GA and BO optimisers. The optimisers focus on `K`, `mcs_base`, and `mcs_enh` since those directly relate to SNR adaptation.

### Parameters you may want to add:
If your wavelet scheme has its own split-level parameter (e.g., decomposition level or subband selection), you could add it to the policy struct in `PolicyController.m` and the CSV format. The feedback loop in `rx_policy_update.m` would then need a rule or GA gene for it.

---

## 6. File Map for Reference

```
FEEDBACK LOOP FILES:
  rx/rx_feedback.m          ← Main feedback orchestrator
  rx/rx_policy_update.m     ← Policy decision logic (rules + GA blend)
  tx/src/PolicyController.m ← Reads/writes policy.csv, manages state
  ml_addons/ga_parameter_optimizer.m  ← GA search over [K, Qb_idx, Qe_idx]  (MCS not evolved) (Angela)
  ml_addons/bo_parameter_optimizer.py ← Bayesian optimiser (Gaussian Process) over [K, MCS_b, MCS_e]; offline GA vs BO comparison tool for report (Angela)
  ml_addons/channel_predictor.m       ← EMA+trend SNR prediction
  
BOLD'S ML ADDITIONS:
  ml_addons/bold_rx_feedback.m        ← Enhanced feedback with 3 ML models
  ml_addons/bold_quality_predictor.m  ← No-reference quality estimation
  ml_addons/bold_channel_lstm.m       ← LSTM channel prediction
  ml_addons/bold_symbol_denoiser.m    ← QAM symbol denoising

TX/RX PIPELINE:
  tx/src/two_layer_tx_pipeline.m      ← Full TX: split→quant→modem
  rx/two_layer_rx_v2.m                ← Full RX: demod→dequant→reconstruct
  channel/channel_model_v1.m          ← Rician fading channel (Jie Du)
```