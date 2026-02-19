function rx_pkt_bytes = demodem(waveRx, cfg)
% demodem: Recover PSDU BYTES from a Non-HT waveform using field indices
% waveRx: column complex waveform of ONE packet (already demuxed)
% cfg: wlanNonHTConfig (PSDULength in BYTES, MCS in double)

    waveRx = waveRx(:);

    % 1) Packet detect (find L-STF start)
    pktOffset = wlanPacketDetect(waveRx, cfg.ChannelBandwidth);
    assert(~isempty(pktOffset) && pktOffset >= 0, "Packet not detected.");
    waveRx = waveRx(pktOffset+1:end);

    % 2) Symbol timing (align to symbol boundary)
    timingOff = wlanSymbolTimingEstimate(waveRx, cfg.ChannelBandwidth);
    waveRx = waveRx(timingOff+1:end);

    % 3) Use field indices to slice L-LTF and Data (this avoids "wrong segment" decode)
    ind = wlanFieldIndices(cfg);

    % Guard: ensure waveform long enough
    reqLen = ind.NonHTData(2);
    assert(numel(waveRx) >= reqLen, "Waveform too short after timing. Need %d, got %d", reqLen, numel(waveRx));

    rxLLTF  = waveRx(ind.LLTF(1):ind.LLTF(2));
    rxData  = waveRx(ind.NonHTData(1):ind.NonHTData(2));

    % 4) Channel estimate from L-LTF
    demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg.ChannelBandwidth);
    chanEst   = wlanLLTFChannelEstimate(demodLLTF, cfg.ChannelBandwidth);

    % 5) Data recover (noiseVar ~ 0 for wire baseline)
    noiseVar = 1e-10;
    rxBits = wlanNonHTDataRecover(rxData, chanEst, noiseVar, cfg);

    rxBits = rxBits(:);

    % 6) Bits -> bytes (MSB-first, must match modem()'s de2bi(...,'left-msb'))
    expBits = 8 * double(cfg.PSDULength);
    assert(numel(rxBits) >= expBits, "Recovered bits too short: got %d need %d", numel(rxBits), expBits);
    rxBits = rxBits(1:expBits);

    rxBytes = uint8(bi2de(reshape(rxBits, 8, []).', 'left-msb')); % column bytes
    rx_pkt_bytes = rxBytes.'; % row bytes
end
