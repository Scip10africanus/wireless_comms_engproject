function txWave = layer_mux(waveBase, waveEnh, mux_gap_us, fs)
% LAYER_MUX concatenate two waveforms with optional zero-gap
    if nargin < 4 || isempty(fs)
        fs = wlanSampleRate(wlanNonHTConfig); % CBW20 default
    end
    gapSamp = round(double(mux_gap_us) * 1e-6 * fs);
    txWave = [waveBase; zeros(gapSamp,1); waveEnh];
end
