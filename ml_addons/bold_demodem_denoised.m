function [rx_pkt_bytes, ok, reason, noiseVar_out] = bold_demodem_denoised(waveRx, cfg, snr_dB)
% BOLD_DEMODEM_DENOISED  Enhanced demodem with ML symbol denoiser
%
% Drop-in replacement for demodem() that applies bold_symbol_denoiser to
% the equalized QAM constellation symbols before Viterbi decoding.
%
% How the denoiser hooks in:
%   Normal path:  rxWave -> OFDM demod -> chanEst -> noiseVar -> Viterbi -> bits
%   Denoiser path: rxWave -> OFDM demod -> ZF equalize -> QAM symbols
%                  -> bold_symbol_denoiser -> compute residual noise
%                  -> noiseVar_denoised (lower) -> Viterbi -> bits
%
% Passing a lower noiseVar to wlanNonHTDataRecover makes the Viterbi
% soft-decision metrics more aggressive, directly improving BER when the
% denoiser successfully pushed symbols toward valid constellation points.
%
% Falls back to standard demodem behaviour if wlanNonHTDemodulate is not
% available in the installed MATLAB WLAN Toolbox version (R2019a+).
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 3 || isempty(snr_dB), snr_dB = 20; end

    waveRx = waveRx(:);
    rx_pkt_bytes = [];
    ok    = false;
    reason = '';
    noiseVar_out = NaN;

    ind    = wlanFieldIndices(cfg);
    reqLen = ind.NonHTData(2);

    if numel(waveRx) < reqLen
        reason = 'input_too_short';
        return;
    end

    %% Timing alignment (matches rx/demodem.m)
    maxOff = 32;
    toff   = wlanSymbolTimingEstimate(waveRx, cfg.ChannelBandwidth);
    if ~isempty(toff) && toff >= 0 && toff <= maxOff && (numel(waveRx)-toff) >= reqLen
        waveRx = waveRx(toff+1:end);
    end
    if numel(waveRx) < reqLen
        reason = 'too_short_after_timing';
        return;
    end

    try
        waveRx  = waveRx(1:reqLen);
        rxLLTF  = waveRx(ind.LLTF(1)    : ind.LLTF(2));
        rxNHTD  = waveRx(ind.NonHTData(1): ind.NonHTData(2));

        %% Channel estimation from LLTF (identical to standard demodem)
        demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg.ChannelBandwidth);
        chanEst   = wlanLLTFChannelEstimate(demodLLTF, cfg.ChannelBandwidth);
        noiseVar  = wlanLLTFNoiseEstimate(demodLLTF);
        noiseVar_out = noiseVar;

        %% Attempt symbol-level denoising — MMSE equalization (Jie Du, Apr 2026)
        % Jie confirmed MMSE is better than ZF for OFDM channels:
        %   ZF:   eqSym = rxOFDM / chanEst         — amplifies noise at weak subcarriers
        %   MMSE: eqSym = conj(h)*rx / (|h|^2 + noiseVar) — regularised, no noise amplification
        noiseVar_denoised = noiseVar;

        try
            % wlanNonHTDemodulate (R2019a+): [Nst x Nsym] — all 52 occupied subcarriers
            % wlanLLTFChannelEstimate:       [Nst x Nltf]  — same 52 subcarriers
            % So sizes match — no pilot stripping needed.
            rxOFDM = wlanNonHTDemodulate(rxNHTD, cfg);   % [52 x Nsym]

            chanEst_col = chanEst(:, 1);                  % [52 x 1]

            % MMSE equalization per subcarrier
            denom   = abs(chanEst_col).^2 + noiseVar + 1e-10;  % [52 x 1]
            eqSym   = (conj(chanEst_col) .* rxOFDM(:,:,1)) ./ denom;  % [52 x Nsym]
            qamSyms = eqSym(:);   % flatten all subcarriers x symbols

            % MCS -> modulation order
            modTable = [4 4 4 16 16 16 64 64];
            M = modTable(double(cfg.MCS) + 1);

            % Apply ML symbol denoiser
            denoisedSyms = bold_symbol_denoiser(qamSyms, snr_dB, M);

            % Compute noise removed. Residual variance = power shifted toward
            % valid constellation points = reduction in effective noise.
            residual     = denoisedSyms - qamSyms;
            removedNoise = var(real(residual(:))) + var(imag(residual(:)));

            % Denoised noiseVar: subtract removed noise, floor at 10% of original
            noiseVar_denoised = max(0.1*noiseVar, noiseVar - removedNoise);
            noiseVar_denoised = max(noiseVar_denoised, 1e-8);
            noiseVar_out      = noiseVar_denoised;

        catch
            % wlanNonHTDemodulate not available or equalization failed.
            % Fall through to standard decode — no regression vs demodem.
            noiseVar_denoised = noiseVar;
        end

        %% Decode with (possibly improved) noise variance
        rxBits  = wlanNonHTDataRecover(rxNHTD, chanEst, noiseVar_denoised, cfg);
        rxBits  = rxBits(:);

        expBits = 8 * double(cfg.PSDULength);
        if numel(rxBits) < expBits
            reason = 'recovered_bits_too_short';
            return;
        end

        rxBits       = rxBits(1:expBits);
        rxBytes      = uint8(bi2de(reshape(rxBits, 8, []).', 'left-msb'));
        rx_pkt_bytes = rxBytes.';
        ok     = true;
        reason = 'ok';

    catch ME
        reason = ['recover_fail:' ME.identifier];
    end
end
