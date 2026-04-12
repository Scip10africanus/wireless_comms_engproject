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

    %% GA blend (if GA gave us something)
    if isfield(feedback,'gaOptimalParams') && ~isempty(fieldnames(feedback.gaOptimalParams))
        g = feedback.gaOptimalParams;
        % 70/30 blend - dont jump too hard
        nK  = round(0.7*cK  + 0.3*double(g.K));
        nQb = round(0.7*cQb + 0.3*double(g.Qb_idx));
        nQe = round(0.7*cQe + 0.3*double(g.Qe_idx));
        nMb = round(0.7*cMb + 0.3*double(g.mcs_base));
        nMe = round(0.7*cMe + 0.3*double(g.mcs_enh));
    else
        %% rule-based fallback
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
                nMe = min(7, cMe + 1);
                nK  = max(5, cK - 1);
            end

            if quality.psnr < 20
                nQb = max(0, cQb - 1);
                nQe = max(0, cQe - 1);
            elseif quality.psnr > 40 && feedback.meetsQualityTargets
                nQe = min(15, cQe + 1);
            end
        end
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

    if isfield(feedback,'enhFailRatio') && isfinite(feedback.enhFailRatio) && feedback.enhFailRatio > 0.5
        nK = max(nK, cK);
        nMe = min(nMe, cMe);
        nMe = max(0, nMe - 1);
    end

    %% Rician channel MCS ceiling
    % Only BPSK (mcs=0) and QPSK-1/2 (mcs=1) survive this
    % channel's fading variance reliably. Higher base MCS causes huge
    % reconstruction failure even at high nominal SNR. The enhancement layer
    % can try higher MCS, failures are caught by the blockOK guard.
    snr = chCond.snr_dB;
    nMb = min(nMb, 1);    % base: hard cap at QPSK 1/2 for channel reliability
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
    policy.Qb_idx   = uint8(max(0,  min(15, nQb)));
    policy.Qe_idx   = uint8(max(0,  min(15, nQe)));
    policy.mcs_base = uint8(max(0,  min(7,  nMb)));
    policy.mcs_enh  = uint8(max(0,  min(7,  nMe)));
    policy.mux_gap_us = cGap;
end
