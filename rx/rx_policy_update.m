function policy = rx_policy_update(currentPolicy, quality, feedback, chCond)
% RX_POLICY_UPDATE  Decide next-frame policy based on quality + feedback
% 4-step pipeline (sequential, not mutually exclusive):
%   Step 1: rules always run (MCS from SSIM/PSNR, Qb/Qe from extreme PSNR)
%   Step 2: GA blends K + Qb_idx + Qe_idx if available (MCS stays with rules)
%   Step 3: channel-predicted SNR may trim MCS before drops
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

    %% Step 2: GA nudges K, Qb_idx, Qe_idx (MCS left to rule branch above)
    % GA optimises layer-split and quantisation; rules + ceiling handle MCS.
    % wGa = GA weight in convex blend (higher -> more GA pull vs rules-only baseline).
    wGa = 0.38;
    if isfield(feedback,'gaBlendWeight') && ~isempty(feedback.gaBlendWeight)
        wGa = max(0, min(1, double(feedback.gaBlendWeight(1))));
    end
    if isfield(feedback,'gaOptimalParams') && isa(feedback.gaOptimalParams,'struct') ...
            && isfield(feedback.gaOptimalParams,'K')
        % When the channel predictor flags a trend (improving/degrading), lean a bit
        % harder on GA so K/Q track the *next* frame better — helps rapid SNR swing
        % scenarios edge out rules-only on average.
        % Skip LSTM-driven wGa bump: trend label is noisy OOD and pulled GA away from rules.
        if ~(isfield(feedback,'mlChannelUsed') && logical(feedback.mlChannelUsed)) ...
                && isfield(feedback,'channelTrend') && strlength(string(feedback.channelTrend)) > 0 ...
                && string(feedback.channelTrend) ~= "stable"
            wGa = min(1, wGa + 0.04);
        end
        g = feedback.gaOptimalParams;
        nK  = round((1-wGa)*nK  + wGa*double(g.K));
        nQb = round((1-wGa)*nQb + wGa*double(g.Qb_idx));
        nQe = round((1-wGa)*nQe + wGa*double(g.Qe_idx));
    end

    %% channel prediction influence (pessimistic trim when forecast SNR drops)
    skipPessim = isfield(feedback,'skipPessimisticChanTrim') && logical(feedback.skipPessimisticChanTrim);
    if ~skipPessim && ~isnan(feedback.predictedSNR(1)) && feedback.predictionConfidence(1) > 0.78
        drop = chCond.snr_dB - feedback.predictedSNR(1);
        if drop > 7
            nMe = max(0, nMe - 1);
            nMb = max(0, nMb - 1);
        elseif drop > 5
            nMe = max(0, nMe - 1);
        end
    end

    %% Step 3b: confident improving forecast + already-strong recon → one enh MCS nudge (Bold ML)
    % Require nominal SNR high enough; nudging enh MCS from LSTM in the stress band caused tails.
    if quality.hasMetrics && quality.ssim >= 0.955 && quality.psnr >= 34 ...
            && chCond.snr_dB >= 20 ...
            && ~isnan(feedback.predictedSNR(1)) && feedback.predictionConfidence(1) > 0.88
        rise = feedback.predictedSNR(1) - chCond.snr_dB;
        if rise >= 3.5 && rise <= 8
            nMe = min(7, nMe + 1);
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
    policy.Qb_idx   = uint8(max(0,  min(3,  nQb)));
    policy.Qe_idx   = uint8(max(0,  min(3,  nQe)));
    policy.mcs_base = uint8(max(0,  min(7,  nMb)));
    policy.mcs_enh  = uint8(max(0,  min(7,  nMe)));
    policy.mux_gap_us = cGap;
end
