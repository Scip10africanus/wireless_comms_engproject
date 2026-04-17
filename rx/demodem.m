function [rx_pkt_bytes, ok, reason, noiseVar] = demodem(waveRx, cfg)
    waveRx = waveRx(:);
    rx_pkt_bytes = [];
    ok = false;
    reason = ''; %#ok<NASGU>
    noiseVar = NaN;

    ind = wlanFieldIndices(cfg);
    reqLen = ind.NonHTData(2);
    if numel(waveRx) < reqLen
        reason = 'input_too_short';
        return;
    end

    % For per-packet waveform, disable packet detect first
    usePktDetect = false;
    maxOffsetAllowed = 32;

    if usePktDetect
        off = wlanPacketDetect(waveRx, cfg.ChannelBandwidth); %#ok<*UNRCH>
        if ~isempty(off) && off >= 0 && off <= maxOffsetAllowed ...
                && (numel(waveRx) - off) >= reqLen
            waveRx = waveRx(off+1:end);
        end
    end

    toff = wlanSymbolTimingEstimate(waveRx, cfg.ChannelBandwidth);
    if ~isempty(toff) && toff >= 0 && toff <= maxOffsetAllowed ...
            && (numel(waveRx) - toff) >= reqLen
        waveRx = waveRx(toff+1:end);
    end

    if numel(waveRx) < reqLen
        reason = 'too_short_after_timing';
        return;
    end

    try
        waveRx = waveRx(1:reqLen);

        rxLLTF = waveRx(ind.LLTF(1):ind.LLTF(2));
        rxData = waveRx(ind.NonHTData(1):ind.NonHTData(2));

        demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg.ChannelBandwidth);
        chanEst   = wlanLLTFChannelEstimate(demodLLTF, cfg.ChannelBandwidth);

        noiseVar = wlanLLTFNoiseEstimate(demodLLTF);
        rxBits   = wlanNonHTDataRecover(rxData, chanEst, noiseVar, cfg);
        rxBits   = rxBits(:);

        expBits = 8 * double(cfg.PSDULength);
        if numel(rxBits) < expBits
            reason = 'recovered_bits_too_short';
            return;
        end

        rxBits = rxBits(1:expBits);
        rxBytes = uint8(bi2de(reshape(rxBits, 8, []).', 'left-msb'));
        rx_pkt_bytes = rxBytes.';
        ok = true;
        reason = 'ok';

    catch ME
        reason = ['recover_fail:' ME.identifier];
        return;
    end
end
