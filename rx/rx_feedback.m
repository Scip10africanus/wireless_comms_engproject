function feedback = rx_feedback(quality, results, currentPolicy, settings)
% RX_FEEDBACK  Generate feedback struct from quality metrics
% Also runs GA optimizer and channel predictor if enabled in settings

    feedback = struct();
    feedback.timestamp = now;
    feedback.ssim = quality.ssim;
    feedback.psnr = quality.psnr;
    feedback.hasQualityMetrics = quality.hasMetrics;

    if quality.hasMetrics
        feedback.meetsQualityTargets = (quality.ssim >= settings.targetSsim) && ...
                                       (quality.psnr >= settings.targetPsnr);
    else
        feedback.meetsQualityTargets = false;
    end

    %% layer health estimate (rough, based on overall SSIM)
    if ~quality.hasMetrics
        feedback.baseLayerStatus = 'marginal';
        feedback.detailLayerStatus = 'marginal';
    elseif quality.ssim > 0.85
        feedback.baseLayerStatus = 'good';
        feedback.detailLayerStatus = 'good';
    elseif quality.ssim > 0.5
        feedback.baseLayerStatus = 'good';
        feedback.detailLayerStatus = 'marginal';
    else
        feedback.baseLayerStatus = 'poor';
        feedback.detailLayerStatus = 'poor';
    end

    % Channel condition: prefer true SNR from history over PSNR-based estimate
    chCond = struct('snr_dB', 20, 'estimatedBaseEvm', 10, 'estimatedEnhEvm', 20);
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

    %% urgency
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

    %% suggestions (legacy format for old RX compat)
    feedback.suggestions = struct('forBaseLayer','keep_current','forDetailLayer','keep_current');
    if quality.hasMetrics
        if quality.ssim < 0.7
            feedback.suggestions.forBaseLayer = 'use_lower_modulation';
            feedback.suggestions.forDetailLayer = 'use_lower_modulation';
        elseif quality.ssim > 0.95 && feedback.meetsQualityTargets
            feedback.suggestions.forDetailLayer = 'can_use_higher_modulation';
        end
    end

    %% channel predictor (if enabled)
    feedback.predictedSNR = NaN;
    feedback.predictionConfidence = 0;
    feedback.channelTrend = 'stable';

    if isfield(settings,'useChannelPredictor') && settings.useChannelPredictor ...
       && isfield(settings,'channelHistory') && ~isempty(settings.channelHistory)
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
            % predictor failed, keep defaults
        end
    end

    %% GA optimizer (if enabled)
    feedback.gaOptimalParams = struct();
    feedback.gaMetrics = struct();

    if isfield(settings,'useGaOptimizer') && settings.useGaOptimizer
        try
            cur = struct('K',double(currentPolicy.K), 'Qb_idx',double(currentPolicy.Qb_idx), ...
                         'Qe_idx',double(currentPolicy.Qe_idx), 'mcs_base',double(currentPolicy.mcs_base), ...
                         'mcs_enh',double(currentPolicy.mcs_enh));
            gaS = struct();
            if isfield(settings,'gaSettings'), gaS = settings.gaSettings; end
            % Optimize for the predicted next-frame SNR rather than the current one.
            % This gives the GA a one-frame head-start when the channel is changing.
            gaChCond = chCond;
            if ~isnan(feedback.predictedSNR(1)) && feedback.predictionConfidence(1) > 0.5
                gaChCond.snr_dB = feedback.predictedSNR(1);
            end
            [feedback.gaOptimalParams, feedback.gaMetrics] = ga_parameter_optimizer(gaChCond, cur, gaS);
        catch
        end
    end

    %% wavelet-specific feedback fields
    feedback.waveletLevels = max(1, min(4, floor(double(currentPolicy.K) / 10)));
    feedback.baseQuant = double(currentPolicy.Qb_idx);
    feedback.enhQuant = double(currentPolicy.Qe_idx);

    %% write updated policy for next frame
    feedback.policy = rx_policy_update(currentPolicy, quality, feedback, chCond);

    if isfield(settings,'policyCsvPath') && ~isempty(settings.policyCsvPath)
        write_csv(settings.policyCsvPath, feedback.policy);
    end
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
