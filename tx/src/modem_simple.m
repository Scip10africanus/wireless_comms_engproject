function [waveTx, cfg] = modem_simple(pkt_bytes, mcs)
% MODEM_SIMPLE - Simple OFDM modulator (no WLAN Toolbox required)
%
% Drop-in replacement for modem.m when WLAN Toolbox is not available.
% Uses basic OFDM with BPSK/QPSK/16QAM based on MCS index.
%
% Inputs:
%   pkt_bytes - uint8 row/column of payload bytes
%   mcs       - MCS index 0-7 (maps to modulation + code rate)
%
% Outputs:
%   waveTx - Time-domain OFDM waveform (complex column vector)
%   cfg    - Config struct with PSDULength and MCS

    bytes = uint8(pkt_bytes(:)).';
    nBytes = numel(bytes);

    % MCS to modulation order
    mcsModOrder = [2, 2, 4, 4, 16, 16, 64, 64];  % BPSK, QPSK, 16QAM, 64QAM
    modOrder = mcsModOrder(double(mcs) + 1);
    bitsPerSymbol = log2(modOrder);

    % Convert bytes to bits (MSB first)
    bits = zeros(nBytes * 8, 1);
    for i = 1:nBytes
        b = bytes(i);
        for k = 1:8
            bits((i-1)*8 + k) = bitand(bitshift(b, -(8-k)), 1);
        end
    end

    % Pad bits to fill OFDM symbols
    nFFT = 64;
    nCP = 16;
    % Data subcarrier indices: [2:27, 39:64] = 26 + 26 = 52 subcarriers
    nDataSubcarriers = 52;
    bitsPerOFDMSymbol = nDataSubcarriers * bitsPerSymbol;
    nOFDMSymbols = ceil(length(bits) / bitsPerOFDMSymbol);
    totalBits = nOFDMSymbols * bitsPerOFDMSymbol;
    bits = [bits; zeros(totalBits - length(bits), 1)];

    % Map bits to QAM symbols
    nSymbols = totalBits / bitsPerSymbol;
    symbols = zeros(nSymbols, 1);

    for i = 1:nSymbols
        bitGroup = bits((i-1)*bitsPerSymbol+1 : i*bitsPerSymbol);
        idx = 0;
        for k = 1:bitsPerSymbol
            idx = idx + bitGroup(k) * 2^(bitsPerSymbol - k);
        end
        symbols(i) = qam_map(idx, modOrder);
    end

    % Build OFDM symbols
    waveTx = [];
    dataSubIdx = [2:27, 39:64];  % Standard 802.11a data subcarrier positions (1-indexed)

    for s = 1:nOFDMSymbols
        symStart = (s-1)*nDataSubcarriers + 1;
        symEnd = min(s*nDataSubcarriers, nSymbols);
        dataSyms = symbols(symStart:symEnd);

        % Zero-pad if needed
        if length(dataSyms) < nDataSubcarriers
            dataSyms = [dataSyms; zeros(nDataSubcarriers - length(dataSyms), 1)];
        end

        % Place into FFT bin
        freqDomain = zeros(nFFT, 1);
        freqDomain(dataSubIdx) = dataSyms;

        % IFFT
        timeDomain = ifft(freqDomain) * sqrt(nFFT);

        % Add cyclic prefix
        withCP = [timeDomain(end-nCP+1:end); timeDomain];
        waveTx = [waveTx; withCP];
    end

    % Config struct (compatible with demodem_simple expectations)
    cfg = struct();
    cfg.PSDULength = nBytes;
    cfg.MCS = double(mcs);
    cfg.nFFT = nFFT;
    cfg.nCP = nCP;
    cfg.nDataSubcarriers = 52;
    cfg.modOrder = modOrder;
    cfg.nOFDMSymbols = nOFDMSymbols;
    cfg.ChannelBandwidth = 'CBW20';
end

function sym = qam_map(idx, M)
% Simple QAM mapper
    switch M
        case 2  % BPSK
            sym = 2*idx - 1;  % -1 or +1
        case 4  % QPSK
            I = 2*mod(idx, 2) - 1;
            Q = 2*floor(idx/2) - 1;
            sym = (I + 1j*Q) / sqrt(2);
        case 16  % 16-QAM
            sqrtM = 4;
            I = 2*mod(idx, sqrtM) - (sqrtM-1);
            Q = 2*floor(idx/sqrtM) - (sqrtM-1);
            sym = (I + 1j*Q) / sqrt(10);
        case 64  % 64-QAM
            sqrtM = 8;
            I = 2*mod(idx, sqrtM) - (sqrtM-1);
            Q = 2*floor(idx/sqrtM) - (sqrtM-1);
            sym = (I + 1j*Q) / sqrt(42);
        otherwise
            sym = 1;
    end
end
