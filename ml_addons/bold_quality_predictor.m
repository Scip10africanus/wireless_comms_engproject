function [predicted_ssim, predicted_psnr, confidence] = bold_quality_predictor(dctCoeffs, K, policy, mode, modelPath, dwtFeatures, noiseVar)
% BOLD_QUALITY_PREDICTOR  No-reference image quality predictor using MLP
%
% CHANGES v3 (Bold - Apr 2026):
%   - Added feature 16: log10(noiseVar) from Jie Du's LLTF channel estimation
%     (Zibo requested this: "noiseVar estimated by Rx done by Jie Du,
%     Hope it could help ML to improve its decision")
%   - NFEAT: 15 -> 16
%
% Bold (Unubold) - EEME30003 Major Project
%
% Feature vector (16 features):
%   --- DCT domain ---
%    1-12. (unchanged — see v2 header)
%   --- DWT domain (Zibo Yuan) ---
%   13. E_LL_ratio
%   14. E_HH_ratio
%   15. base_share
%   --- Channel estimation (Jie Du) ---
%   16. log10(noiseVar) from LLTF noise estimate
%       Range: ~[-4, 0] for SNR 40->0 dB. Prior: -2 (20 dB SNR).

    if nargin < 4 || isempty(mode),      mode = 'predict'; end
    if nargin < 5 || isempty(modelPath)
        modelPath = fullfile(fileparts(mfilename('fullpath')), 'quality_predictor.mat');
    end
    if nargin < 6, dwtFeatures = struct(); end
    if nargin < 7 || isempty(noiseVar) || ~isfinite(noiseVar)
        noiseVar = 0.01;  % prior: ~20 dB SNR
    end

    N     = 8;
    NFEAT = 16;

    %% Extract features
    dctFeat  = extract_dct_features(dctCoeffs, K, N, policy);  % [12 x 1]
    dwtFeat  = extract_dwt_features(dwtFeatures, policy, K);   % [3  x 1]
    chanFeat = log10(max(noiseVar, 1e-12));                     % scalar (feat 16)

    features = [dctFeat; dwtFeat; chanFeat];  % [16 x 1]

    if strcmp(mode, 'extract_features')
        predicted_ssim = features;
        predicted_psnr = [];
        confidence     = [];
        return;
    end

    %% Load model and normalisation stats
    persistent qNet qNetPath featMean featStd
    if isempty(qNet) || isempty(qNetPath) || ~strcmp(qNetPath, modelPath)
        if exist(modelPath, 'file')
            qNet = SimpleNet([NFEAT 64 32 16 2], 1e-3);
            qNet.load_model(modelPath);
            qNetPath = modelPath;

            % Load feature normalisation stats saved during training
            normPath = strrep(modelPath, 'quality_predictor.mat', 'quality_predictor_norm.mat');
            if exist(normPath, 'file')
                nm = load(normPath);
                if isfield(nm,'feat_mean') && isfield(nm,'feat_std')
                    featMean = nm.feat_mean(:);
                    featStd  = nm.feat_std(:);
                else
                    featMean = zeros(NFEAT,1);
                    featStd  = ones(NFEAT,1);
                end
            else
                featMean = zeros(NFEAT,1);
                featStd  = ones(NFEAT,1);
            end
        else
            warning('[QualityPredictor] Model not found at %s. Using heuristic.', modelPath);
            [predicted_ssim, predicted_psnr, confidence] = heuristic_quality(features);
            return;
        end
    end

    %% Normalise features then predict
    feat_norm = (features - featMean) ./ (featStd + 1e-8);
    out = qNet.predict(feat_norm);   % [2 x 1]: [ssim_pred; psnr_pred_norm]

    % De-normalise PSNR (SSIM target was kept in [0,1] so no scaling needed)
    normPath = strrep(modelPath, 'quality_predictor.mat', 'quality_predictor_norm.mat');
    psnr_mean = 30; psnr_std = 8;  % safe defaults if norm file unavailable
    if exist(normPath,'file')
        nm = load(normPath);
        if isfield(nm,'psnr_mean'), psnr_mean = nm.psnr_mean; end
        if isfield(nm,'psnr_std'),  psnr_std  = nm.psnr_std;  end
    end

    % Sigmoid on SSIM output instead of hard clamp.
    % Linear output layer can produce values > 1.0 when training SSIM targets
    % are concentrated near 1.0 (training distribution imbalance). Sigmoid
    % maps the unbounded output smoothly into (0,1) so predictions are
    % meaningful rather than pinned at the boundary.
    predicted_ssim = 1 / (1 + exp(-out(1)));
    predicted_ssim = max(0, min(1, predicted_ssim));
    predicted_psnr = max(5, min(60, out(2) * psnr_std + psnr_mean));

    %% Confidence from base-layer energy ratio (feature 2)
    energy_ratio = features(2);
    if energy_ratio > 0.5
        confidence = 0.85;
    elseif energy_ratio > 0.2
        confidence = 0.65;
    else
        confidence = 0.4;
    end
end

%% -----------------------------------------------------------------------
%%  DCT feature extraction (12 features — unchanged from v1)
%% -----------------------------------------------------------------------
function feat = extract_dct_features(dctCoeffs, K, N, policy)
    [Hp, Wp] = size(dctCoeffs);
    nBlocksH = Hp / N;
    nBlocksW = Wp / N;
    nBlocks  = nBlocksH * nBlocksW;

    zigIdx = Zigzag_index(N);

    total_base_energy = 0;
    total_enh_energy  = 0;
    dc_vals   = zeros(nBlocks, 1);
    base_vals = zeros(nBlocks * K, 1);
    enh_vals  = zeros(nBlocks * (N*N - K), 1);

    bIdx = 0;
    bPtr = 0;  ePtr = 0;
    for br = 1:nBlocksH
        for bc = 1:nBlocksW
            bIdx = bIdx + 1;
            rS = (br-1)*N + 1;  cS = (bc-1)*N + 1;
            block = dctCoeffs(rS:rS+N-1, cS:cS+N-1);
            vec   = block(zigIdx);

            dc_vals(bIdx) = vec(1);

            base = vec(1:K);
            enh  = vec(K+1:end);

            total_base_energy = total_base_energy + sum(base.^2);
            total_enh_energy  = total_enh_energy  + sum(enh.^2);

            base_vals(bPtr+1 : bPtr+K) = base(:);
            enh_vals (ePtr+1 : ePtr+length(enh)) = enh(:);
            bPtr = bPtr + K;
            ePtr = ePtr + length(enh);
        end
    end

    total_energy = total_base_energy + total_enh_energy + 1e-12;

    feat = zeros(12, 1);
    feat(1)  = log10(total_energy + 1);
    feat(2)  = total_base_energy / total_energy;
    feat(3)  = total_enh_energy  / total_energy;
    feat(4)  = std(base_vals) / (mean(abs(base_vals)) + 1e-6);
    feat(5)  = std(enh_vals)  / (mean(abs(enh_vals))  + 1e-6);
    feat(6)  = mean(dc_vals);
    feat(7)  = std(dc_vals);
    feat(8)  = double(K) / 64;
    feat(9)  = double(policy.Qb_idx) / 3;
    feat(10) = double(policy.Qe_idx) / 3;
    feat(11) = double(policy.mcs_base) / 7;
    feat(12) = double(policy.mcs_enh)  / 7;
end

%% -----------------------------------------------------------------------
%%  DWT feature extraction (3 features — Zibo Yuan, Apr 2026)
%%  Uses dwt_extract_features output struct when available.
%%  Falls back to DCT-domain proxies or neutral priors if not.
%% -----------------------------------------------------------------------
function feat = extract_dwt_features(dwtFeatures, policy, K)
    feat = zeros(3, 1);

    if isstruct(dwtFeatures) && isfield(dwtFeatures,'E_LL_ratio') && ...
            ~isnan(dwtFeatures.E_LL_ratio)
        %% DWT features available from Zibo's dwt_extract_features()
        feat(1) = double(dwtFeatures.E_LL_ratio);
        feat(2) = double(dwtFeatures.E_HH_ratio);
        if isfield(dwtFeatures,'base_share') && ~isnan(dwtFeatures.base_share)
            feat(3) = double(dwtFeatures.base_share);
        else
            feat(3) = double(K) / 64;  % DCT proxy: K ratio
        end
    else
        %% Fallback: DCT-domain proxies for the DWT features
        %  E_LL_ratio proxy: assume moderate low-frequency dominance
        feat(1) = 0.5;
        %  E_HH_ratio proxy: assume low texture for compressed image
        feat(2) = 0.1;
        %  base_share proxy: use K-based split ratio
        feat(3) = double(K) / 64;
    end

    % Clamp to valid range
    feat = max(0, min(1, feat));
end

%% -----------------------------------------------------------------------
%%  Heuristic fallback (when model .mat not trained yet)
%% -----------------------------------------------------------------------
function [ssim_est, psnr_est, conf] = heuristic_quality(feat)
    energy_ratio = feat(2);
    k_norm       = feat(8);
    mcs_e        = feat(12);
    ll_ratio = feat(13);  % LL energy ratio
    nv_feat   = feat(16);  % log10 noiseVar: more negative = cleaner channel

    ssim_est = 0.3 + 0.35*energy_ratio + 0.1*k_norm + 0.1*ll_ratio ...
               - 0.1*mcs_e - 0.1*(nv_feat + 2);  % noiseVar centred at -2 (20dB prior)
    ssim_est = max(0.1, min(0.99, ssim_est));
    psnr_est = 10 + 30 * ssim_est;
    conf     = 0.3;
end
