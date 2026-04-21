function policy = rx_policy_update(currentPolicy, quality, feedback, chCond)
% RX_POLICY_UPDATE  Decide next-frame policy based on quality + feedback
% 4-step pipeline (sequential, not mutually exclusive):
%   Step 1: rules always run (MCS from SSIM/PSNR, Qb/Qe from extreme PSNR)
%   Step 2: GA blends K + Qb_idx + Qe_idx if available (MCS stays with rules)
%   Step 3: LSTM-predicted SNR drops MCS pre-emptively
%   Step 4: Rician MCS ceiling hard-caps based on SNR

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

    %% Step 1: rule-based MCS adaptation (always runs — measured SSIM/PSNR feeds back)
    % GA does NOT override this path. Rules react to actual reconstruction quality.
    % Fixes the open-loop issue where the GA branch silenced SSIM-triggered MCS decisions.
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
        elseif quality.ssim > 0.98 && quality.psnr > 40 && feedback.meetsQualityTargets
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

    %% Step 2: GA nudges K, Qb_idx, Qe_idx (MCS left to rule branch above)
    % GA optimises layer-split and quantisation; rules + ceiling handle MCS.
    if isfield(feedback,'gaOptimalParams') && ~isempty(fieldnames(feedback.gaOptimalParams))
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
    % Thresholds tightened empirically: uncoded modes (MCS 2, 5) are fragile
    % on Rician fading because deep fades corrupt entire packets with no FEC
    % recovery. MCS 2 (QPSK R=1.0) was observed failing even at SNR=38 dB.
    snr = chCond.snr_dB;
    if     snr >= 42, base_ceil = 4;
    elseif snr >= 35, base_ceil = 3;
    elseif snr >= 28, base_ceil = 2;
    elseif snr >= 18, base_ceil = 1;
    else,             base_ceil = 0;
    end
    nMb = min(nMb, base_ceil);

    if     snr >= 42, enh_ceil = 5;
    elseif snr >= 40, enh_ceil = 4;
    elseif snr >= 38, enh_ceil = 3;
    elseif snr >= 35, enh_ceil = 2;
    elseif snr >= 13, enh_ceil = 1;
    else,             enh_ceil = 0;
    end
    nMe = min(nMe, enh_ceil);

    % clamp everything
    policy.K        = uint8(max(5,  min(32, nK)));   % cap K at 32 (max half of 64 block)
    policy.Qb_idx   = uint8(max(0,  min(3,  nQb)));
    policy.Qe_idx   = uint8(max(0,  min(3,  nQe)));
    policy.mcs_base = uint8(max(0,  min(7,  nMb)));
    policy.mcs_enh  = uint8(max(0,  min(7,  nMe)));
    policy.mux_gap_us = cGap;
end
