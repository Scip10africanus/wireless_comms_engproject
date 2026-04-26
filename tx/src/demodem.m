function rx_pkt_bytes = demodem(waveRx, cfg)
    waveRx = waveRx(:);

    ind = wlanFieldIndices(cfg);
    reqLen = ind.NonHTData(2);

    % --- Fast path: assume waveform already starts at packet boundary
    if numel(waveRx) >= reqLen
        pktOffset = 0;
        timingOff = 0;
    else
        % If it's shorter than required, we MUST be missing samples anyway
        error("Input waveform shorter than required packet length: need %d got %d", reqLen, numel(waveRx));
    end

    % --- Optional: try packet detect only as a small correction (guarded)
    maxOffsetAllowed = 200; % keep small: demux-by-length should be near 0
    off = wlanPacketDetect(waveRx, cfg.ChannelBandwidth);
    if ~isempty(off) && off >= 0 && off <= maxOffsetAllowed
        pktOffset = off;
    end
    waveRx = waveRx(pktOffset+1:end);

    % Symbol timing estimate (also guarded)
    toff = wlanSymbolTimingEstimate(waveRx, cfg.ChannelBandwidth);
    if ~isempty(toff) && toff >= 0 && toff <= maxOffsetAllowed && (numel(waveRx)-toff) >= reqLen
        timingOff = toff;
    else
        timingOff = 0;
    end
    waveRx = waveRx(timingOff+1:end);

    % Guard
    assert(numel(waveRx) >= reqLen, "Waveform too short after timing. Need %d, got %d", reqLen, numel(waveRx));

    rxLLTF  = waveRx(ind.LLTF(1):ind.LLTF(2));
    rxData  = waveRx(ind.NonHTData(1):ind.NonHTData(2));

    demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg.ChannelBandwidth);
    chanEst   = wlanLLTFChannelEstimate(demodLLTF, cfg.ChannelBandwidth);

    noiseVar = wlanLLTFNoiseEstimate(demodLLTF);
    rxBits   = wlanNonHTDataRecover(rxData, chanEst, noiseVar, cfg);
    rxBits   = rxBits(:);

    expBits = 8 * double(cfg.PSDULength);
    assert(numel(rxBits) >= expBits, "Recovered bits too short: got %d need %d", numel(rxBits), expBits);
    rxBits = rxBits(1:expBits);

    rxBytes = uint8(bi2de(reshape(rxBits, 8, []).', 'left-msb'));
    rx_pkt_bytes = rxBytes.';
end
