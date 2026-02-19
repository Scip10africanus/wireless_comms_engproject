function rx_pkt_bytes = demodem_simple(waveRx, cfg)
% DEMODEM_SIMPLE - Simple OFDM demodulator (no WLAN Toolbox required)
%
% Drop-in replacement for demodem.m when WLAN Toolbox is not available.
% Matches modem_simple.m output format.
%
% Inputs:
%   waveRx - Received OFDM waveform (complex column vector)
%   cfg    - Config struct from modem_simple:
%       .PSDULength, .MCS, .nFFT, .nCP, .nDataSubcarriers, .modOrder, .nOFDMSymbols
%
% Outputs:
%   rx_pkt_bytes - Recovered uint8 bytes (row vector)

    waveRx = waveRx(:);

    nFFT = cfg.nFFT;
    nCP = cfg.nCP;
    nDataSubcarriers = cfg.nDataSubcarriers;
    modOrder = cfg.modOrder;
    bitsPerSymbol = log2(modOrder);
    nOFDMSymbols = cfg.nOFDMSymbols;
    nBytes = double(cfg.PSDULength);
    symLen = nFFT + nCP;

    dataSubIdx = [2:27, 39:64];  % Must match modem_simple

    % Demodulate OFDM symbols
    allSymbols = [];
    for s = 1:nOFDMSymbols
        symStart = (s-1)*symLen + nCP + 1;
        symEnd = symStart + nFFT - 1;

        if symEnd > length(waveRx)
            break;
        end

        freqDomain = fft(waveRx(symStart:symEnd)) / sqrt(nFFT);
        dataSyms = freqDomain(dataSubIdx);
        allSymbols = [allSymbols; dataSyms(:)];
    end

    % QAM demap to bits
    nSymbols = length(allSymbols);
    allBits = zeros(nSymbols * bitsPerSymbol, 1);

    for i = 1:nSymbols
        idx = qam_demap(allSymbols(i), modOrder);
        for k = 1:bitsPerSymbol
            allBits((i-1)*bitsPerSymbol + k) = bitand(floor(idx / 2^(bitsPerSymbol - k)), 1);
        end
    end

    % Convert bits to bytes
    totalBits = nBytes * 8;
    if length(allBits) < totalBits
        allBits = [allBits; zeros(totalBits - length(allBits), 1)];
    end
    allBits = allBits(1:totalBits);

    rx_pkt_bytes = zeros(1, nBytes, 'uint8');
    for b = 1:nBytes
        val = 0;
        for k = 1:8
            val = val + allBits((b-1)*8 + k) * 2^(8-k);
        end
        rx_pkt_bytes(b) = uint8(val);
    end
end

function idx = qam_demap(sym, M)
% Simple hard-decision QAM demapper
    switch M
        case 2  % BPSK
            idx = double(real(sym) > 0);
        case 4  % QPSK
            sym = sym * sqrt(2);
            I = double(real(sym) > 0);
            Q = double(imag(sym) > 0);
            idx = I + 2*Q;
        case 16  % 16-QAM
            sqrtM = 4;
            sym = sym * sqrt(10);
            I = round((real(sym) + (sqrtM-1)) / 2);
            Q = round((imag(sym) + (sqrtM-1)) / 2);
            I = max(0, min(sqrtM-1, I));
            Q = max(0, min(sqrtM-1, Q));
            idx = I + sqrtM * Q;
        case 64  % 64-QAM
            sqrtM = 8;
            sym = sym * sqrt(42);
            I = round((real(sym) + (sqrtM-1)) / 2);
            Q = round((imag(sym) + (sqrtM-1)) / 2);
            I = max(0, min(sqrtM-1, I));
            Q = max(0, min(sqrtM-1, Q));
            idx = I + sqrtM * Q;
        otherwise
            idx = 0;
    end
end
