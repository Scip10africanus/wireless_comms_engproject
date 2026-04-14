% function rx_pkt_bytes = demodem(waveRx, cfg)
%     waveRx = waveRx(:);
% 
%     ind = wlanFieldIndices(cfg);
%     reqLen = ind.NonHTData(2);
% 
%     % --- Fast path: assume waveform already starts at packet boundary
%     if numel(waveRx) >= reqLen
%         pktOffset = 0;
%         timingOff = 0;
%     else
%         % If it's shorter than required, we MUST be missing samples anyway
%         error("Input waveform shorter than required packet length: need %d got %d", reqLen, numel(waveRx));
%     end
% 
%     % --- Optional: try packet detect only as a small correction (guarded)
%     maxOffsetAllowed = 200; % keep small: demux-by-length should be near 0
%     off = wlanPacketDetect(waveRx, cfg.ChannelBandwidth);
%     if ~isempty(off) && off >= 0 && off <= maxOffsetAllowed
%         pktOffset = off;
%     end
%     waveRx = waveRx(pktOffset+1:end);
% 
%     % Symbol timing estimate (also guarded)
%     toff = wlanSymbolTimingEstimate(waveRx, cfg.ChannelBandwidth);
%     if ~isempty(toff) && toff >= 0 && toff <= maxOffsetAllowed && (numel(waveRx)-toff) >= reqLen
%         timingOff = toff;
%     else
%         timingOff = 0;
%     end
%     waveRx = waveRx(timingOff+1:end);
% 
%     % Guard
%     assert(numel(waveRx) >= reqLen, "Waveform too short after timing. Need %d, got %d", reqLen, numel(waveRx));
% 
%     rxLLTF  = waveRx(ind.LLTF(1):ind.LLTF(2));
%     rxData  = waveRx(ind.NonHTData(1):ind.NonHTData(2));
% 
%     demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg.ChannelBandwidth);
%     chanEst   = wlanLLTFChannelEstimate(demodLLTF, cfg.ChannelBandwidth);
% 
%     noiseVar = wlanLLTFNoiseEstimate(demodLLTF);
%     rxBits = wlanNonHTDataRecover(rxData, chanEst, noiseVar, cfg);
%     rxBits = rxBits(:);
% 
%     expBits = 8 * double(cfg.PSDULength);
%     assert(numel(rxBits) >= expBits, "Recovered bits too short: got %d need %d", numel(rxBits), expBits);
%     rxBits = rxBits(1:expBits);
% 
%     rxBytes = uint8(bi2de(reshape(rxBits, 8, []).', 'left-msb'));
%     rx_pkt_bytes = rxBytes.';
% end
% 
function [rx_pkt_bytes, ok, reason, dbg] = demodem(waveRx, cfg)
% DEMODEM  Custom OFDM receiver following the pipeline of reference OFDM.m
%
%   Pipeline (reverses modem.m):
%     waveform -> reshape -> remove CP -> FFT
%              -> extract pilots -> LS channel estimate -> linear interp
%              -> equalize data subcarriers -> [despread]
%              -> demodulate -> [Viterbi decode] -> bits -> bytes
%
%   Optional 4th output: dbg struct containing
%     .eqSym        - equalized data symbols  (for constellation plot)
%     .H_est        - estimated channel response
%     .rxBitsCoded  - coded bits before Viterbi decoding
%     .rxBitsSrc    - source bits after Viterbi decoding
%     .rxBytes      - final received bytes
 
    rx_pkt_bytes = [];
    ok = false;
    reason = '';
    dbg = struct('eqSym', [], 'H_est', [], ...
                 'rxBitsCoded', [], 'rxBitsSrc', [], 'rxBytes', []);
 
    p         = cfg.params;
    nOFDMSyms = cfg.nOFDMSyms;
    expLen    = nOFDMSyms * p.N_symbol;
 
    waveRx = waveRx(:);
    if numel(waveRx) < expLen
        reason = 'input_too_short';
        return;
    end
 
    try
        % ==================== 1. Reshape into OFDM symbols ====================
        waveRx  = waveRx(1:expLen);
        timeMat = reshape(waveRx, p.N_symbol, nOFDMSyms);
 
        % ==================== 2. Remove CP ====================
        timeNoCP = timeMat(p.N_cp + 1 : end, :);         % N_fft x nOFDMSyms
 
        % ==================== 3. FFT to frequency domain ====================
        freqGrid = fft(timeNoCP, p.N_fft);               % N_fft x nOFDMSyms
 
        % ==================== 4. Channel Estimation ====================
        %   LS at pilots + linear interpolation, same as reference OFDM.m
        rxPilots = freqGrid(p.pilot_pos, :);             % N_pilot x nOFDMSyms
        H_pilot  = rxPilots ./ p.pilot_value;            % element-wise LS
 
        H_full = zeros(p.N_fft, nOFDMSyms);
        for k = 1:nOFDMSyms
            H_full(:, k) = interp1(p.pilot_pos.', H_pilot(:, k), ...
                                   (1:p.N_fft).', 'linear', 'extrap');
        end
 
        % ==================== 5. Equalization ====================
        %   Reference OFDM.m uses ZF (Y/H). We default to MMSE for
        %   robustness at high MCS, with noise variance estimated
        %   from pilot-to-estimate residual.
        H_data = H_full(p.data_pos, :);
        Y_data = freqGrid(p.data_pos, :);
 
        noiseVar = estimate_noise_var(rxPilots, H_pilot, p.pilot_value);
        eqData   = (conj(H_data) .* Y_data) ./ (abs(H_data).^2 + noiseVar);
 
        % Serialize in same order as TX (column-major)
        eqSymVec = eqData(:);
        % Remove padding symbols
        eqSymVec = eqSymVec(1 : cfg.nModSyms);
 
        % ==================== 6. Despreading (if enabled) ====================
        if p.spread_factor > 1
            sf = p.spread_factor;
            % Reference OFDM.m uses despread() from comm toolbox.
            % For reproducibility we use a correlate-with-code approach.
            nSyms0 = numel(eqSymVec) / sf;
            eqSymVec = reshape(eqSymVec, sf, nSyms0);
            % Note: the spreading code must be shared with TX via a seed
            % or global. With spread_factor = 1 this block is skipped.
            code = 2 * (randi([0 1], sf, 1)) - 1;
            eqSymVec = (code.' * eqSymVec).' / sf;
        end
 
        % ==================== 7. Demodulation ====================
        switch cfg.modName
            case 'BPSK'
                symInts = pskdemod(eqSymVec, 2, 0);
            case 'QPSK'
                symInts = pskdemod(eqSymVec, 4, pi/4, 'gray');
            case '16QAM'
                symInts = qamdemod(eqSymVec, 16, 'gray', 'UnitAveragePower', true);
            case '64QAM'
                symInts = qamdemod(eqSymVec, 64, 'gray', 'UnitAveragePower', true);
        end
 
        % ==================== 8. Symbols -> coded bits ====================
        rxBitsP = de2bi(symInts, cfg.bits_per_sym, 'left-msb').';
        rxBitsP = rxBitsP(:);
        rxBitsCoded = rxBitsP(1 : cfg.nCodedBits);    % strip modulation padding
 
        % ==================== 9. Viterbi Decoding ====================
        if p.useConvCode
            trellis = poly2trellis(p.convConstraint, p.convPoly);
            rxBitsSrc = vitdec(rxBitsCoded, trellis, p.tblen, 'trunc', 'hard');
        else
            rxBitsSrc = rxBitsCoded;
        end
        rxBitsSrc = rxBitsSrc(1 : cfg.nSrcBits);      % strip any tail
 
        % ==================== 10. Bits -> bytes ====================
        nBytes  = cfg.nDataBytes;
        rxBytes = uint8(bi2de(reshape(rxBitsSrc, 8, []).', 'left-msb'));
        rx_pkt_bytes = rxBytes(1:nBytes).';
 
        % ==================== 11. Debug info ====================
        dbg.eqSym       = eqSymVec;
        dbg.H_est       = H_full;
        dbg.rxBitsCoded = rxBitsCoded;
        dbg.rxBitsSrc   = rxBitsSrc;
        dbg.rxBytes     = rxBytes;
 
        ok = true;
        reason = 'ok';
 
    catch ME
        reason = ['recover_fail:' ME.identifier];
        return;
    end
end
 
 
function nVar = estimate_noise_var(rxPilots, H_pilot, pilot_val)
% Estimate noise variance from the residual between the LS-reconstructed
% pilot symbols and the actual received pilots. Simple and self-contained.
    reconPilots = H_pilot .* pilot_val;
    resid       = rxPilots - reconPilots;
    nVar        = max(var(resid(:)), 1e-10);
end