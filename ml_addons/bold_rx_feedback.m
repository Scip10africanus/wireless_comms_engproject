function feedback = bold_rx_feedback(quality, results, currentPolicy, settings)
% BOLD_RX_FEEDBACK  Enhanced RX feedback with 3 ML models
% Drop-in replacement for rx_feedback.m that additionally uses:
%   1. MLP quality predictor (no-reference SSIM/PSNR estimation)
%   2. LSTM channel predictor (replaces EMA+trend)
%   3. Symbol denoiser metrics (if available)
%
% Bold (Unubold) - EEME30003 Major Project
%
% Inputs:
%   quality        - struct with .ssim, .psnr, .hasMetrics, .mse
%   results        - struct with .success, .blkOK, .nBlocks
%   currentPolicy  - current PolicyController policy struct
%   settings       - struct with control flags:
%       .originalImage          (optional - if absent, uses ML prediction)
%       .dctCoeffsReceived      (for quality predictor)
%       .useMLQualityPredictor  (default: true)
%       .useMLChannelPredictor  (default: true)
%       .channelHistory         (struct with .snr vector)
%       .predictorParams        (horizon, windowSize, etc.)
%       .useGaOptimizer         (default: FALSE -- see GA note below)
%       .gaSettings
%       .policyCsvPath
%       .verbose
%
% GA NOTE (updated Apr 2026):
%   GA is disabled by default. When enabled, it optimises [K, Qb_idx,
%   Qe_idx] — layer split and quantisation. MCS is NOT in the GA search
%   space; rules + LSTM ceiling handle MCS exclusively.
%   rx_policy_update.m runs rules first (Step 1), then blends GA
%   suggestions for K/Qb/Qe in Step 2 (70/30). The previous if/else
%   structure that made GA and rules mutually exclusive has been replaced
%   by this sequential pipeline.
%   GA's calc_fitness now receives measured prevSSIM and prevEnhFailRatio
%   via chCond, closing the feedback loop for K and quantisation.
%
% Outputs:
%   feedback - struct with all metrics, predictions, and updated policy

    if nargin < 4 || ~isstruct(settings), settings = struct(); end
    if ~isfield(settings, 'targetSsim'),            settings.targetSsim = 0.9; end
    if ~isfield(settings, 'targetPsnr'),            settings.targetPsnr = 30; end
    if ~isfield(settings, 'verbose'),               settings.verbose = false; end
    if ~isfield(settings, 'useMLQualityPredictor'), settings.useMLQualityPredictor = true; end
    if ~isfield(settings, 'useMLChannelPredictor'), settings.useMLChannelPredictor = true; end
    if ~isfield(settings, 'useGaOptimizer'),        settings.useGaOptimizer = false; end
    if ~isfield(settings, 'useChannelPredictor'),   settings.useChannelPredictor = false; end
    if ~isfield(settings, 'predictorParams'),       settings.predictorParams = struct(); end

    feedback = struct();
    feedback.timestamp = now;
    feedback.hasQualityMetrics = quality.hasMetrics;

    %% ━━━ ML Model 1: Quality Predictor ━━━
    feedback.mlQualityUsed = false;

    if settings.useMLQualityPredictor && isfield(settings, 'dctCoeffsReceived') ...
       && ~isempty(settings.dctCoeffsReceived)
        try
            % Pass noiseVar from Jie's LLTF estimation as feature 16
            nv = 0.01;  % default prior (20 dB SNR)
            if isfield(settings,'noiseVar') && isfinite(settings.noiseVar)
                nv = settings.noiseVar;
            end

            % DWT features (Zibo, Apr 2026): when running the DWT closed
            % loop, the caller passes real DWT features via settings.dwtFeatures.
            % For the DCT path this stays empty and the quality predictor
            % falls back to DCT-domain proxies for features 13-15.
            dwtFeat = struct();
            if isfield(settings,'dwtFeatures') && isstruct(settings.dwtFeatures)
                dwtFeat = settings.dwtFeatures;
            end

            [ml_ssim, ml_psnr, ml_conf] = bold_quality_predictor( ...
                settings.dctCoeffsReceived, double(currentPolicy.K), currentPolicy, ...
                'predict', [], dwtFeat, nv);

            feedback.ml_ssim = ml_ssim;
            feedback.ml_psnr = ml_psnr;
            feedback.ml_quality_confidence = ml_conf;
            feedback.mlQualityUsed = true;

            if settings.verbose
                fprintf('[ML-Quality] Predicted SSIM=%.4f PSNR=%.2f (conf=%.2f)', ...
                    ml_ssim, ml_psnr, ml_conf);
                if quality.hasMetrics
                    fprintf(' | Actual SSIM=%.4f PSNR=%.2f', quality.ssim, quality.psnr);
                end
                fprintf('\n');
            end
        catch ME
            if settings.verbose
                fprintf('[ML-Quality] Failed: %s\n', ME.message);
            end
        end
    end

    % Use actual metrics if available, otherwise use ML predictions
    if quality.hasMetrics
        feedback.ssim = quality.ssim;
        feedback.psnr = quality.psnr;
    elseif feedback.mlQualityUsed
        feedback.ssim = feedback.ml_ssim;
        feedback.psnr = feedback.ml_psnr;
        quality.ssim = feedback.ml_ssim;
        quality.psnr = feedback.ml_psnr;
        quality.hasMetrics = true;
    else
        feedback.ssim = 0.5;
        feedback.psnr = 20;
    end

    feedback.meetsQualityTargets = (feedback.ssim >= settings.targetSsim) && ...
                                   (feedback.psnr >= settings.targetPsnr);

    %% ━━━ Layer health + channel condition estimate ━━━
    if ~quality.hasMetrics
        feedback.baseLayerStatus  = 'marginal';
        feedback.detailLayerStatus = 'marginal';
    elseif quality.ssim > 0.85
        feedback.baseLayerStatus  = 'good';
        feedback.detailLayerStatus = 'good';
    elseif quality.ssim > 0.5
        feedback.baseLayerStatus  = 'good';
        feedback.detailLayerStatus = 'marginal';
    else
        feedback.baseLayerStatus  = 'poor';
        feedback.detailLayerStatus = 'poor';
    end

    chCond = struct('snr_dB', 20, 'estimatedBaseEvm', 10, 'estimatedEnhEvm', 20);
    % Prefer the REAL channel SNR over PSNR-derived SNR when we have it.
    % PSNR-derived SNR is dangerously optimistic when reconstruction happens
    % to succeed at a high-MCS setting that is one bad fade away from failure.
    % The MCS ceiling downstream needs the true channel state, not a lagging
    % quality proxy. Restored from original rx_feedback.m (Unubold, Apr 2026).
    if isfield(settings, 'channelHistory') && ~isempty(settings.channelHistory) && ...
       isfield(settings.channelHistory, 'snr') && ~isempty(settings.channelHistory.snr)
        chCond.snr_dB = settings.channelHistory.snr(end);
    elseif quality.hasMetrics
        chCond.snr_dB = min(40, max(5, quality.psnr - 5));
    end
    if quality.hasMetrics
        if quality.ssim > 0.9
            chCond.estimatedBaseEvm = 5;  chCond.estimatedEnhEvm = 10;
        elseif quality.ssim > 0.7
            chCond.estimatedBaseEvm = 15; chCond.estimatedEnhEvm = 25;
        else
            chCond.estimatedBaseEvm = 30; chCond.estimatedEnhEvm = 40;
        end
    end
    feedback.baseLayerError  = chCond.estimatedBaseEvm;
    feedback.detailLayerError = chCond.estimatedEnhEvm;

    %% ━━━ Urgency detection ━━━
    feedback.needsUrgentChange = false;
    feedback.urgentReason = '';
    if quality.hasMetrics && quality.ssim < 0.5
        feedback.needsUrgentChange = true;
        feedback.urgentReason = 'Critical quality (SSIM < 0.5)';
    elseif quality.hasMetrics && quality.psnr < 15
        feedback.needsUrgentChange = true;
        feedback.urgentReason = 'Very low PSNR (< 15 dB)';
    end
    if ~results.success
        feedback.needsUrgentChange = true;
        feedback.urgentReason = 'Decoding failure';
    end

    %% ━━━ ML Model 2: LSTM Channel Predictor ━━━
    feedback.predictedSNR = NaN;
    feedback.predictionConfidence = 0;
    feedback.channelTrend = 'stable';
    feedback.mlChannelUsed = false;

    if settings.useMLChannelPredictor && isfield(settings, 'channelHistory') ...
       && ~isempty(settings.channelHistory)
        try
            [predSNR, ~, predConf] = bold_channel_lstm( ...
                settings.channelHistory, settings.predictorParams);

            feedback.predictedSNR = predSNR;
            feedback.predictionConfidence = predConf;
            feedback.mlChannelUsed = true;

            if predSNR(1) > chCond.snr_dB + 2
                feedback.channelTrend = 'improving';
            elseif predSNR(1) < chCond.snr_dB - 2
                feedback.channelTrend = 'degrading';
            end

            if settings.verbose
                fprintf('[ML-Channel] Predicted SNR=[%.1f] conf=%.2f trend=%s\n', ...
                    predSNR(1), predConf(1), feedback.channelTrend);
            end
        catch ME
            if settings.verbose
                fprintf('[ML-Channel] Failed: %s\n', ME.message);
            end
        end
    elseif isfield(settings,'useChannelPredictor') && settings.useChannelPredictor ...
       && isfield(settings,'channelHistory') && ~isempty(settings.channelHistory)
        % Fall back to EMA predictor
        try
            [predSNR, ~, predConf] = channel_predictor(settings.channelHistory, settings.predictorParams);
            feedback.predictedSNR = predSNR;
            feedback.predictionConfidence = predConf;
            if predSNR(1) > chCond.snr_dB + 2
                feedback.channelTrend = 'improving';
            elseif predSNR(1) < chCond.snr_dB - 2
                feedback.channelTrend = 'degrading';
            end
        catch
        end
    end

    %% ━━━ Suggestions ━━━
    feedback.suggestions = struct('forBaseLayer','keep_current','forDetailLayer','keep_current');
    if quality.hasMetrics
        if quality.ssim < 0.7
            feedback.suggestions.forBaseLayer = 'use_lower_modulation';
            feedback.suggestions.forDetailLayer = 'use_lower_modulation';
        elseif quality.ssim > 0.95 && feedback.meetsQualityTargets
            feedback.suggestions.forDetailLayer = 'can_use_higher_modulation';
        end
    end

    %% ━━━ GA Optimizer (Angela's domain - keep compatibility) ━━━
    % Default OFF. When enabled, populates feedback.gaOptimalParams with
    % [K, Qb_idx, Qe_idx]. rx_policy_update.m blends these in Step 2
    % after rules have already run in Step 1.
    feedback.gaOptimalParams = struct();
    feedback.gaMetrics = struct();
    if isfield(settings,'useGaOptimizer') && settings.useGaOptimizer
        try
            cur = struct('K',double(currentPolicy.K), 'Qb_idx',double(currentPolicy.Qb_idx), ...
                         'Qe_idx',double(currentPolicy.Qe_idx), 'mcs_base',double(currentPolicy.mcs_base), ...
                         'mcs_enh',double(currentPolicy.mcs_enh));
            gaS = struct();
            if isfield(settings,'gaSettings'), gaS = settings.gaSettings; end
            gaChCond = chCond;
            if quality.hasMetrics
                gaChCond.prevSSIM = quality.ssim;
            end
            if isstruct(results) && isfield(results,'enhFailRatio')
                gaChCond.prevEnhFailRatio = double(results.enhFailRatio);
            end
            if ~isnan(feedback.predictedSNR(1)) && feedback.predictionConfidence(1) > 0.5
                gaChCond.snr_dB = feedback.predictedSNR(1);
            end
            [feedback.gaOptimalParams, feedback.gaMetrics] = ga_parameter_optimizer(gaChCond, cur, gaS);
        catch
        end
    end

    %% ━━━ Policy Update ━━━
    feedback.policy = rx_policy_update(currentPolicy, quality, feedback, chCond);

    %% ━━━ Write CSV ━━━
    if isfield(settings,'policyCsvPath') && ~isempty(settings.policyCsvPath)
        write_csv(settings.policyCsvPath, feedback.policy);
    end

    %% ━━━ ML summary ━━━
    feedback.mlModelsActive = struct( ...
        'qualityPredictor', feedback.mlQualityUsed, ...
        'channelLSTM',      feedback.mlChannelUsed, ...
        'symbolDenoiser',   isfield(settings,'useML') && settings.useML ...
    );
end

function write_csv(path, p)
    fid = fopen(path, 'w');
    if fid == -1, return; end
    fprintf(fid, 'version,timestamp,tx_mode,K,Qb_idx,Qe_idx,crc_flag,endian_flag,codec_id,reserved_flag,mcs_base,mcs_enh,mux_gap_us\n');
    fprintf(fid, '%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%g\n', ...
        p.version, p.timestamp, p.tx_mode, p.K, p.Qb_idx, p.Qe_idx, ...
        p.crc_flag, p.endian_flag, p.codec_id, p.reserved_flag, ...
        p.mcs_base, p.mcs_enh, p.mux_gap_us);
    fclose(fid);
end
