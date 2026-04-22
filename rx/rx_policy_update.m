function policy = rx_policy_update(currentPolicy, quality, feedback, chCond)
% RX_POLICY_UPDATE  Decide next-frame policy based on quality + feedback
% Uses GA blend if available, otherwise rule-based adaptation

    policy = struct();
    policy.version   = uint32(double(currentPolicy.version) + 1);
    policy.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    policy.tx_mode   = 'split';
    policy.crc_flag      = currentPolicy.crc_flag;
    policy.endian_flag   = currentPolicy.endian_flag;
    policy.codec_id      = currentPolicy.codec_id;
    policy.reserved_flag = uint8(0);

    cK  = double(currentPolicy.K);
    cQb = double(currentPolicy.Qb_idx);
    cQe = double(currentPolicy.Qe_idx);
    cMb = double(currentPolicy.mcs_base);
    cMe = double(currentPolicy.mcs_enh);
    cGap = double(currentPolicy.mux_gap_us);

    %% Step 1: Rule-based (always runs first)
    nK = cK;  nQb = cQb;  nQe = cQe;  nMb = cMb;  nMe = cMe;

    if quality.hasMetrics
        if feedback.needsUrgentChange
            nMb = max(0, cMb - 2);
            nMe = max(0, cMe - 2);
            nK  = min(60, cK + 5);
        elseif quality.ssim < 0.7
            nMb = max(0, cMb - 1);
            nMe = max(0, cMe - 1);
            nK  = min(60, cK + 2);
        elseif quality.ssim < 0.85
            nMe = max(0, cMe - 1);
        elseif quality.ssim > 0.95 && feedback.meetsQualityTargets
            nMb = min(7, cMb + 1);   % push base MCS up (ceiling will cap it)
            nMe = min(7, cMe + 1);
            % K stays stable when quality is good — no aggressive shrink
        end

        if quality.psnr < 20
            nQb = max(0, cQb - 1);
            nQe = max(0, cQe - 1);
        elseif quality.psnr > 40 && feedback.meetsQualityTargets
            nQb = min(3, cQb + 1);
            nQe = min(3, cQe + 1);
        end
    end

    %% SNR-driven proactive MCS push (runs even without quality metrics)
    % Drive nMb toward the ceiling rather than waiting for quality to be > 0.95
    snr_now = chCond.snr_dB;
    if     snr_now >= 33, target_mb = 4;
    elseif snr_now >= 25, target_mb = 3;
    elseif snr_now >= 18, target_mb = 2;
    elseif snr_now >= 13, target_mb = 1;
    else,                 target_mb = 0;
    end
    if nMb < target_mb
        nMb = nMb + 1;   % step up at most 1 per frame
    end

    %% Step 2: GA blend for K/Qb/Qe only — MCS stays with rules + ceiling
    if isfield(feedback,'gaOptimalParams') && isstruct(feedback.gaOptimalParams) && ~isempty(fieldnames(feedback.gaOptimalParams))
        g = feedback.gaOptimalParams;
        nK  = round(0.7*nK  + 0.3*double(g.K));
        nQb = round(0.7*nQb + 0.3*double(g.Qb_idx));
        nQe = round(0.7*nQe + 0.3*double(g.Qe_idx));
    end

    %% channel prediction influence
    if ~isnan(feedback.predictedSNR(1)) && feedback.predictionConfidence(1) > 0.5
        drop = chCond.snr_dB - feedback.predictedSNR(1);
        if drop > 5
            nMe = max(0, nMe - 1);
            nMb = max(0, nMb - 1);
        elseif drop > 3
            nMe = max(0, nMe - 1);
        end
    end

    %% Rician channel MCS ceiling (SNR-adaptive for both layers)
    % Base layer uses a conservative SNR-based ceiling (2 MCS levels below enh)
    % to protect the core image quality. Enhancement layer uses a higher ceiling
    % since failures are tolerated gracefully by the blockOK guard.
    snr = chCond.snr_dB;
    if     snr >= 40, base_ceil = 5;
    elseif snr >= 33, base_ceil = 4;
    elseif snr >= 25, base_ceil = 3;
    elseif snr >= 18, base_ceil = 2;
    elseif snr >= 13, base_ceil = 1;
    else,             base_ceil = 0;
    end
    nMb = min(nMb, base_ceil);

    if     snr >= 33, enh_ceil = 6;
    elseif snr >= 30, enh_ceil = 5;
    elseif snr >= 25, enh_ceil = 4;
    elseif snr >= 18, enh_ceil = 3;
    elseif snr >= 13, enh_ceil = 2;
    elseif snr >= 8,  enh_ceil = 1;
    else,             enh_ceil = 0;
    end
    nMe = min(nMe, enh_ceil);

    % clamp everything
    policy.K        = uint8(max(5,  min(32, nK)));   % cap K at 32 (max half of 64 block)
    policy.Qb_idx   = uint8(max(0,  min(3, nQb)));
    policy.Qe_idx   = uint8(max(0,  min(3, nQe)));
    policy.mcs_base = uint8(max(0,  min(7,  nMb)));
    policy.mcs_enh  = uint8(max(0,  min(7,  nMe)));
    policy.mux_gap_us = cGap;
end
