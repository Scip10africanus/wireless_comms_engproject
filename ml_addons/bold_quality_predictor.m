function [predicted_ssim, predicted_psnr, confidence] = bold_quality_predictor(dctCoeffs, K, policy, mode, modelPath)
% BOLD_QUALITY_PREDICTOR  No-reference image quality predictor using MLP
% Predicts SSIM and PSNR from received DCT coefficients WITHOUT needing
% the original image. This is critical for real-world deployment where
% the receiver has no access to the reference.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Inputs:
%   dctCoeffs  - Received DCT coefficient matrix [Hp x Wp] (after IDCT input)
%   K          - Layer split point (from policy)
%   policy     - Current TX policy struct
%   mode       - 'predict' (default) | 'extract_features'
%   modelPath  - Path to trained model .mat (default: 'quality_predictor.mat')
%
% Outputs:
%   predicted_ssim - Estimated SSIM [0, 1]
%   predicted_psnr - Estimated PSNR (dB)
%   confidence     - Prediction confidence [0, 1]
%
% Feature vector (12 features):
%   1. Total DCT energy (log scale)
%   2. Base layer energy ratio (base_energy / total_energy)
%   3. Enhancement layer energy ratio
%   4. Variance of base coefficients
%   5. Variance of enhancement coefficients
%   6. DC coefficient mean (average block brightness)
%   7. DC coefficient variance (brightness variation)
%   8. K / 64 (normalised split point)
%   9. Qb_idx (normalised)
%  10. Qe_idx (normalised)
%  11. MCS base (normalised)
%  12. MCS enhancement (normalised)

    if nargin < 4, mode = 'predict'; end
    if nargin < 5, modelPath = fullfile(fileparts(mfilename('fullpath')), 'quality_predictor.mat'); end

    N = 8;  % block size

    % --- Extract features from DCT coefficients ---
    features = extract_quality_features(dctCoeffs, K, N, policy);

    if strcmp(mode, 'extract_features')
        predicted_ssim = features;  % return raw features
        predicted_psnr = [];
        confidence = [];
        return;
    end

    % --- Load trained model ---
    persistent qNet qNetPath
    if isempty(qNet) || isempty(qNetPath) || ~strcmp(qNetPath, modelPath)
        if exist(modelPath, 'file')
            qNet = SimpleNet([12 64 32 2], 1e-3);
            qNet.load_model(modelPath);
            qNetPath = modelPath;
        else
            warning('Quality predictor model not found at %s. Using heuristic fallback.', modelPath);
            [predicted_ssim, predicted_psnr, confidence] = heuristic_quality(features);
            return;
        end
    end

    % --- Predict ---
    feat_col = features(:);  % [12 x 1]
    out = qNet.predict(feat_col);  % [2 x 1]: [ssim_pred; psnr_pred]

    predicted_ssim = max(0, min(1, out(1)));
    predicted_psnr = max(5, min(60, out(2)));

    % Confidence based on feature stability
    energy_ratio = features(2);
    if energy_ratio > 0.5
        confidence = 0.85;
    elseif energy_ratio > 0.2
        confidence = 0.65;
    else
        confidence = 0.4;
    end
end

%% Feature extraction
function feat = extract_quality_features(dctCoeffs, K, N, policy)
    [Hp, Wp] = size(dctCoeffs);
    nBlocksH = Hp / N;
    nBlocksW = Wp / N;
    nBlocks = nBlocksH * nBlocksW;

    zigIdx = Zigzag_index(N);

    total_base_energy = 0;
    total_enh_energy  = 0;
    dc_vals = zeros(nBlocks, 1);
    base_vals = [];
    enh_vals  = [];

    bIdx = 0;
    for br = 1:nBlocksH
        for bc = 1:nBlocksW
            bIdx = bIdx + 1;
            rStart = (br-1)*N + 1;
            cStart = (bc-1)*N + 1;
            block = dctCoeffs(rStart:rStart+N-1, cStart:cStart+N-1);

            vec = block(zigIdx);
            dc_vals(bIdx) = vec(1);

            base = vec(1:K);
            enh  = vec(K+1:end);

            total_base_energy = total_base_energy + sum(base.^2);
            total_enh_energy  = total_enh_energy  + sum(enh.^2);

            base_vals = [base_vals; base(:)];
            enh_vals  = [enh_vals;  enh(:)];
        end
    end

    total_energy = total_base_energy + total_enh_energy + 1e-12;

    feat = zeros(12, 1);
    feat(1)  = log10(total_energy + 1);                 % log total energy
    feat(2)  = total_base_energy / total_energy;         % base energy ratio
    feat(3)  = total_enh_energy / total_energy;          % enh energy ratio
    feat(4)  = std(base_vals) / (mean(abs(base_vals)) + 1e-6);  % base CV
    feat(5)  = std(enh_vals)  / (mean(abs(enh_vals))  + 1e-6);  % enh CV
    feat(6)  = mean(dc_vals);                            % DC mean
    feat(7)  = std(dc_vals);                             % DC variance
    feat(8)  = double(K) / 64;                           % normalised K
    feat(9)  = double(policy.Qb_idx) / 3;               % normalised Qb
    feat(10) = double(policy.Qe_idx) / 3;               % normalised Qe
    feat(11) = double(policy.mcs_base) / 7;              % normalised MCS base
    feat(12) = double(policy.mcs_enh) / 7;               % normalised MCS enh
end

%% Heuristic fallback (when model not trained yet)
function [ssim_est, psnr_est, conf] = heuristic_quality(feat)
    energy_ratio = feat(2);
    k_norm = feat(8);
    mcs_e  = feat(12);

    % Higher base energy ratio + higher K = better quality
    ssim_est = 0.3 + 0.5 * energy_ratio + 0.15 * k_norm - 0.1 * mcs_e;
    ssim_est = max(0.1, min(0.99, ssim_est));

    psnr_est = 10 + 30 * ssim_est;
    conf = 0.3;
end
