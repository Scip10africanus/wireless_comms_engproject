function feedback = content_aware_feedback(rxImage, refImage, rxMetrics, params)
% CONTENT_AWARE_FEEDBACK - Generate feedback for adaptive UEP transmission
%
% Produces feedback specifically designed for DCT-based content-aware
% transmission with Unequal Error Protection. Outputs:
%   - SSIM/PSNR quality metrics
%   - Recommended DCT frequency threshold adjustment
%   - Per-band modulation recommendations
%   - Urgency flags for rapid adaptation
%
% =========================================================================
% TODO: Adapt after meeting with TX team:
%   - Confirm feedback message format/bit allocation
%   - Confirm update rate (per frame? per N frames?)
%   - Confirm DCT threshold adjustment granularity
%   - Confirm if TX can handle per-band modulation changes
% =========================================================================
%
% Inputs:
%   rxImage   - Received reconstructed image
%   refImage  - Reference image (if available, otherwise [])
%   rxMetrics - Structure from RX processing:
%       .demapMetrics  - Per-band EVM/BER from uep_demapper
%       .coeffInfo     - DCT coefficient stats from dct_reconstructor
%       .channelEst    - Channel estimate (optional)
%   params    - Configuration:
%       .targetSSIM       - Target SSIM quality (default: 0.9)
%       .targetPSNR       - Target PSNR in dB (default: 30)
%       .feedbackBits     - Total feedback bits (default: 16)
%       .currentThreshold - Current DCT freq threshold at TX
%       .currentMods      - Current modulation orders [critical, normal, noncrit]
%
% Outputs:
%   feedback - Structure containing:
%       .ssim            - Measured SSIM (if ref available)
%       .psnr            - Measured PSNR (if ref available)
%       .qualityMet      - true if targets achieved
%       .thresholdDelta  - Recommended change to DCT threshold
%       .modRecommend    - Recommended modulations per band
%       .urgentAdapt     - Flag for immediate adaptation needed
%       .feedbackBits    - Encoded feedback for transmission
%       .diagnostics     - Detailed diagnostic info

    %% Initialize
    feedback = struct();
    feedback.timestamp = now;

    %% Default parameters
    if nargin < 4
        params = struct();
    end
    params = set_defaults(params);

    %% Calculate image quality metrics (if reference available)
    if ~isempty(refImage) && ~isempty(rxImage)
        [feedback.ssim, feedback.psnr] = calculate_quality(rxImage, refImage);
        feedback.hasQualityMetrics = true;

        % Check if targets met
        feedback.qualityMet = (feedback.ssim >= params.targetSSIM) && ...
                              (feedback.psnr >= params.targetPSNR);
    else
        feedback.ssim = NaN;
        feedback.psnr = NaN;
        feedback.hasQualityMetrics = false;
        feedback.qualityMet = false;
    end

    %% Analyze per-band performance
    bandAnalysis = analyze_band_performance(rxMetrics, params);
    feedback.bandAnalysis = bandAnalysis;

    %% Generate threshold recommendation
    feedback.thresholdDelta = recommend_threshold_change(feedback, bandAnalysis, params);

    %% Generate per-band modulation recommendations
    feedback.modRecommend = recommend_modulations(bandAnalysis, feedback, params);

    %% Determine urgency
    feedback.urgentAdapt = check_urgency(feedback, bandAnalysis, params);

    %% Encode feedback for transmission
    feedback.feedbackBits = encode_feedback_bits(feedback, params);

    %% Generate diagnostics
    feedback.diagnostics = generate_diagnostics(feedback, bandAnalysis, params);
end

%% Parameter Functions

function params = set_defaults(params)
% SET_DEFAULTS - Apply default parameter values

    if ~isfield(params, 'targetSSIM')
        params.targetSSIM = 0.9;
    end
    if ~isfield(params, 'targetPSNR')
        params.targetPSNR = 30;
    end
    if ~isfield(params, 'feedbackBits')
        params.feedbackBits = 16;
    end
    if ~isfield(params, 'currentThreshold')
        params.currentThreshold = 21;  % Default for 8x8 DCT
    end
    if ~isfield(params, 'currentMods')
        params.currentMods = [4, 16, 64];  % [critical, normal, noncritical]
    end

    % Quality change thresholds for adaptation
    if ~isfield(params, 'ssimUrgentThresh')
        params.ssimUrgentThresh = 0.7;  % Below this = urgent
    end
    if ~isfield(params, 'evmUrgentThresh')
        params.evmUrgentThresh = 25;    % Above this % = urgent
    end
end

%% Quality Calculation

function [ssim_val, psnr_val] = calculate_quality(rxImg, refImg)
% CALCULATE_QUALITY - Compute SSIM and PSNR

    rxImg = double(rxImg);
    refImg = double(refImg);

    % PSNR
    mse = mean((rxImg(:) - refImg(:)).^2);
    if mse == 0
        psnr_val = 100;
    else
        psnr_val = 10 * log10(255^2 / mse);
    end

    % Simplified SSIM
    K1 = 0.01; K2 = 0.03; L = 255;
    C1 = (K1*L)^2; C2 = (K2*L)^2;

    if ndims(rxImg) == 3
        rxImg = 0.299*rxImg(:,:,1) + 0.587*rxImg(:,:,2) + 0.114*rxImg(:,:,3);
        refImg = 0.299*refImg(:,:,1) + 0.587*refImg(:,:,2) + 0.114*refImg(:,:,3);
    end

    mu_rx = mean(rxImg(:));
    mu_ref = mean(refImg(:));
    sigma_rx = std(rxImg(:));
    sigma_ref = std(refImg(:));
    sigma_rxref = mean((rxImg(:) - mu_rx) .* (refImg(:) - mu_ref));

    ssim_val = ((2*mu_rx*mu_ref + C1) * (2*sigma_rxref + C2)) / ...
               ((mu_rx^2 + mu_ref^2 + C1) * (sigma_rx^2 + sigma_ref^2 + C2));
end

%% Band Analysis

function analysis = analyze_band_performance(rxMetrics, params)
% ANALYZE_BAND_PERFORMANCE - Analyze quality per frequency band

    analysis = struct();

    % Extract per-band metrics if available
    if isfield(rxMetrics, 'demapMetrics')
        dm = rxMetrics.demapMetrics;

        analysis.critical.evm = getfield_safe(dm.critical, 'avgEVM', NaN);
        analysis.critical.ber = getfield_safe(dm.critical, 'ber', NaN);
        analysis.critical.count = getfield_safe(dm.critical, 'count', 0);

        analysis.normal.evm = getfield_safe(dm.normal, 'avgEVM', NaN);
        analysis.normal.ber = getfield_safe(dm.normal, 'ber', NaN);
        analysis.normal.count = getfield_safe(dm.normal, 'count', 0);

        analysis.noncritical.evm = getfield_safe(dm.noncritical, 'avgEVM', NaN);
        analysis.noncritical.ber = getfield_safe(dm.noncritical, 'ber', NaN);
        analysis.noncritical.count = getfield_safe(dm.noncritical, 'count', 0);
    else
        % Default/unknown
        analysis.critical = struct('evm', NaN, 'ber', NaN, 'count', 0);
        analysis.normal = struct('evm', NaN, 'ber', NaN, 'count', 0);
        analysis.noncritical = struct('evm', NaN, 'ber', NaN, 'count', 0);
    end

    % Extract DCT coefficient info if available
    if isfield(rxMetrics, 'coeffInfo')
        ci = rxMetrics.coeffInfo;
        analysis.lowFreqEnergy = getfield_safe(ci.lowFreq, 'energy', NaN);
        analysis.highFreqEnergy = getfield_safe(ci.highFreq, 'energy', NaN);
        analysis.energyRatio = getfield_safe(ci, 'lowFreqEnergyRatio', NaN);
    else
        analysis.lowFreqEnergy = NaN;
        analysis.highFreqEnergy = NaN;
        analysis.energyRatio = NaN;
    end

    % Determine band health
    analysis.critical.health = assess_band_health(analysis.critical.evm);
    analysis.normal.health = assess_band_health(analysis.normal.evm);
    analysis.noncritical.health = assess_band_health(analysis.noncritical.evm);
end

function health = assess_band_health(evm)
% ASSESS_BAND_HEALTH - Convert EVM to health rating

    if isnan(evm)
        health = 'unknown';
    elseif evm < 10
        health = 'excellent';
    elseif evm < 20
        health = 'good';
    elseif evm < 30
        health = 'fair';
    else
        health = 'poor';
    end
end

%% Recommendation Functions

function delta = recommend_threshold_change(feedback, bandAnalysis, params)
% RECOMMEND_THRESHOLD_CHANGE - Suggest DCT threshold adjustment
%
% Positive delta = include more coefficients as "critical" (more protection)
% Negative delta = fewer critical coefficients (higher throughput)
%
% TODO: Tune these heuristics based on actual system performance

    delta = 0;

    % If quality targets not met
    if feedback.hasQualityMetrics && ~feedback.qualityMet
        if feedback.ssim < params.targetSSIM
            % Need more protection - expand critical region
            ssimGap = params.targetSSIM - feedback.ssim;
            delta = delta + round(ssimGap * 10);  % Scale factor TBD
        end
    end

    % If critical band has high EVM, expand protection
    if ~isnan(bandAnalysis.critical.evm) && bandAnalysis.critical.evm > 15
        delta = delta + 2;
    end

    % If non-critical band is very clean, could reduce protection
    if ~isnan(bandAnalysis.noncritical.evm) && bandAnalysis.noncritical.evm < 10
        if feedback.qualityMet
            delta = delta - 1;  % Can afford to reduce protection
        end
    end

    % Clamp to reasonable range
    % TODO: Adjust range based on actual DCT block size
    delta = max(-5, min(5, delta));
end

function mods = recommend_modulations(bandAnalysis, feedback, params)
% RECOMMEND_MODULATIONS - Suggest modulation order per band
%
% Returns [critical_mod, normal_mod, noncritical_mod]
%
% TODO: Fine-tune thresholds based on actual BER/quality relationships

    currentMods = params.currentMods;
    mods = currentMods;  % Start with current

    % Critical band adjustments
    if strcmp(bandAnalysis.critical.health, 'poor')
        % Reduce modulation for more robustness
        mods(1) = max(2, currentMods(1) / 2);  % Don't go below BPSK
    elseif strcmp(bandAnalysis.critical.health, 'excellent') && feedback.qualityMet
        % Could increase if channel allows
        mods(1) = min(16, currentMods(1) * 2);
    end

    % Normal band adjustments
    if strcmp(bandAnalysis.normal.health, 'poor')
        mods(2) = max(4, currentMods(2) / 2);
    elseif strcmp(bandAnalysis.normal.health, 'excellent') && feedback.qualityMet
        mods(2) = min(64, currentMods(2) * 2);
    end

    % Non-critical band adjustments
    if strcmp(bandAnalysis.noncritical.health, 'poor')
        mods(3) = max(16, currentMods(3) / 2);
    elseif strcmp(bandAnalysis.noncritical.health, 'excellent')
        mods(3) = min(256, currentMods(3) * 2);
    end

    % Ensure ordering (critical <= normal <= noncritical)
    mods(2) = max(mods(2), mods(1));
    mods(3) = max(mods(3), mods(2));
end

function urgent = check_urgency(feedback, bandAnalysis, params)
% CHECK_URGENCY - Determine if immediate adaptation is needed

    urgent = false;

    % SSIM critically low
    if feedback.hasQualityMetrics && feedback.ssim < params.ssimUrgentThresh
        urgent = true;
    end

    % Critical band failing
    if ~isnan(bandAnalysis.critical.evm) && ...
       bandAnalysis.critical.evm > params.evmUrgentThresh
        urgent = true;
    end

    % Critical band in poor health
    if strcmp(bandAnalysis.critical.health, 'poor')
        urgent = true;
    end
end

%% Feedback Encoding

function bits = encode_feedback_bits(feedback, params)
% ENCODE_FEEDBACK_BITS - Encode feedback into compact bit representation
%
% 16-bit format (TODO: adjust based on TX requirements):
%   Bits 1-4:   SSIM quantized (0-15, maps to 0.0-1.0)
%   Bits 5-8:   PSNR quantized (0-15, maps to 0-60 dB)
%   Bits 9-11:  Threshold delta (-4 to +3, 3 bits signed)
%   Bits 12-13: Critical mod recommendation (00=BPSK, 01=QPSK, 10=16QAM, 11=64QAM)
%   Bit 14:     Urgent flag
%   Bits 15-16: Reserved/parity

    nBits = params.feedbackBits;
    bits = zeros(nBits, 1);

    % SSIM (4 bits: 0-15)
    if ~isnan(feedback.ssim)
        ssimQ = round(feedback.ssim * 15);
        ssimQ = max(0, min(15, ssimQ));
    else
        ssimQ = 0;
    end
    bits(1:4) = de2bi(ssimQ, 4, 'left-msb')';

    % PSNR (4 bits: 0-15 maps to 0-60 dB)
    if ~isnan(feedback.psnr)
        psnrQ = round(feedback.psnr / 4);  % 4 dB steps
        psnrQ = max(0, min(15, psnrQ));
    else
        psnrQ = 0;
    end
    bits(5:8) = de2bi(psnrQ, 4, 'left-msb')';

    % Threshold delta (3 bits signed: -4 to +3)
    deltaBiased = feedback.thresholdDelta + 4;  % Bias to make positive
    deltaBiased = max(0, min(7, deltaBiased));
    bits(9:11) = de2bi(deltaBiased, 3, 'left-msb')';

    % Critical modulation (2 bits)
    critMod = feedback.modRecommend(1);
    switch critMod
        case 2
            modBits = [0 0];
        case 4
            modBits = [0 1];
        case 16
            modBits = [1 0];
        otherwise
            modBits = [1 1];
    end
    bits(12:13) = modBits';

    % Urgent flag
    bits(14) = feedback.urgentAdapt;

    % Parity (even parity over bits 1-14)
    bits(15) = mod(sum(bits(1:14)), 2);

    % Reserved
    bits(16) = 0;
end

%% Diagnostics

function diag = generate_diagnostics(feedback, bandAnalysis, params)
% GENERATE_DIAGNOSTICS - Human-readable diagnostic summary

    diag = struct();

    % Quality summary
    if feedback.hasQualityMetrics
        diag.qualitySummary = sprintf('SSIM=%.3f (target %.2f), PSNR=%.1f dB (target %.0f)', ...
            feedback.ssim, params.targetSSIM, feedback.psnr, params.targetPSNR);
        if feedback.qualityMet
            diag.qualityStatus = 'TARGETS MET';
        else
            diag.qualityStatus = 'TARGETS NOT MET';
        end
    else
        diag.qualitySummary = 'No reference image - quality unknown';
        diag.qualityStatus = 'UNKNOWN';
    end

    % Band summary
    diag.bandSummary = sprintf('Critical:%s, Normal:%s, NonCrit:%s', ...
        bandAnalysis.critical.health, bandAnalysis.normal.health, ...
        bandAnalysis.noncritical.health);

    % Recommendation summary
    diag.recommendation = sprintf('Threshold delta: %+d, Mods: [%d,%d,%d]->[%d,%d,%d]', ...
        feedback.thresholdDelta, ...
        params.currentMods(1), params.currentMods(2), params.currentMods(3), ...
        feedback.modRecommend(1), feedback.modRecommend(2), feedback.modRecommend(3));

    if feedback.urgentAdapt
        diag.urgency = 'URGENT - Immediate adaptation recommended';
    else
        diag.urgency = 'Normal - Gradual adaptation OK';
    end
end

%% Utility Functions

function val = getfield_safe(s, field, default)
% GETFIELD_SAFE - Get field with default if missing

    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
