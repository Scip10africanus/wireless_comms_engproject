function txWave = layer_mux(waveBase, waveEnh, mux_gap_us, fs)
    if nargin < 1 || isempty(waveBase), waveBase = complex(zeros(0,1)); end
    if nargin < 2 || isempty(waveEnh),  waveEnh  = complex(zeros(0,1)); end
    if nargin < 3 || isempty(mux_gap_us), mux_gap_us = 0; end
    if nargin < 4 || isempty(fs), fs = 20e6; end

    gapSamp = round(double(mux_gap_us) * 1e-6 * fs);
    gap = complex(zeros(gapSamp,1));

    if isempty(waveBase)
        txWave = waveEnh;
    elseif isempty(waveEnh)
        txWave = waveBase;
    else
        txWave = [waveBase; gap; waveEnh];
    end
end