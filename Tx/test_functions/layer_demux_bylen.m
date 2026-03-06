function [rxBaseWave, rxEnhWave] = layer_demux_bylen(rxWave, lenBase, lenEnh, mux_gap_us, fs)
% LAYER_DEMUX_BYLEN split by known waveform lengths (wire-mode friendly)
    gapSamp = round(double(mux_gap_us) * 1e-6 * fs);
    rxBaseWave = rxWave(1:lenBase);
    start2 = lenBase + gapSamp + 1;
    rxEnhWave  = rxWave(start2 : start2 + lenEnh - 1);
end
