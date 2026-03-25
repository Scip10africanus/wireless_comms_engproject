function [baseBits, enhBits, headerParams, demodInfo] = ofdm_demod(rxWave, ofdmInfo)
% ofdm_demod - reverse of ofdm_mod
% strips CP, does FFT, extracts base/enh bits from their subcarriers, decodes header

    nFFT = ofdmInfo.nFFT;
    nCP = ofdmInfo.nCP;
    nOFDMSymbols = ofdmInfo.nOFDMSymbols;
    criticalIdx = ofdmInfo.criticalIdx;
    noncriticalIdx = ofdmInfo.noncriticalIdx;
    baseBitsPerOFDM = ofdmInfo.baseBitsPerOFDM;
    enhBitsPerOFDM = ofdmInfo.enhBitsPerOFDM;
    % Read modulation orders stored by ofdm_mod; fall back to original defaults
    if isfield(ofdmInfo, 'baseMod'), baseMod = ofdmInfo.baseMod; else baseMod = 4;  end
    if isfield(ofdmInfo, 'enhMod'),  enhMod  = ofdmInfo.enhMod;  else enhMod  = 64; end

    rxWave = rxWave(:);

    allBaseBits = zeros(nOFDMSymbols * baseBitsPerOFDM, 1);
    allEnhBits = zeros(nOFDMSymbols * enhBitsPerOFDM, 1);
    baseEVM = 0;
    enhEVM = 0;

    for sym = 1:nOFDMSymbols
        startIdx = (sym-1) * (nFFT + nCP) + 1;
        endIdx = startIdx + nFFT + nCP - 1;
        if endIdx > length(rxWave), break; end

        symWithCP = rxWave(startIdx:endIdx);
        symTime = symWithCP(nCP+1:end);  % strip the cyclic prefix
        freqDomain = fft(symTime, nFFT);

        % base layer - critical subcarriers
        baseSymbols = freqDomain(criticalIdx);
        baseBitsChunk = qamdemod(baseSymbols, baseMod, 'OutputType', 'bit', 'UnitAveragePower', true);
        bStart = (sym-1) * baseBitsPerOFDM + 1;
        allBaseBits(bStart:bStart + baseBitsPerOFDM - 1) = baseBitsChunk;

        % enhancement layer - noncritical subcarriers
        enhSymbols = freqDomain(noncriticalIdx);
        enhBitsChunk = qamdemod(enhSymbols, enhMod, 'OutputType', 'bit', 'UnitAveragePower', true);
        eStart = (sym-1) * enhBitsPerOFDM + 1;
        allEnhBits(eStart:eStart + enhBitsPerOFDM - 1) = enhBitsChunk;

        % EVM - compare received symbols to ideal reference
        baseRef = qammod(baseBitsChunk, baseMod, 'InputType', 'bit', 'UnitAveragePower', true);
        baseEVM = baseEVM + mean(abs(baseSymbols - baseRef).^2);
        enhRef = qammod(enhBitsChunk, enhMod, 'InputType', 'bit', 'UnitAveragePower', true);
        enhEVM = enhEVM + mean(abs(enhSymbols - enhRef).^2);
    end

    baseEVM = baseEVM / nOFDMSymbols;
    enhEVM = enhEVM / nOFDMSymbols;

    % decode header from start of base bits
    headerParams = struct();
    headerValid = false;
    nHeaderBits = ofdmInfo.nHeaderBits;
    if nHeaderBits > 0 && length(allBaseBits) >= nHeaderBits
        [headerParams, headerValid] = header_codec('decode', allBaseBits(1:nHeaderBits));
    end

    % strip header then trim to original bit lengths
    if nHeaderBits > 0
        baseBitsAll = allBaseBits(nHeaderBits+1:end);
    else
        baseBitsAll = allBaseBits;
    end

    origBaseLen = ofdmInfo.origBaseBitsLen;
    origEnhLen = ofdmInfo.origEnhBitsLen;

    if length(baseBitsAll) >= origBaseLen
        baseBits = baseBitsAll(1:origBaseLen);
    else
        baseBits = [baseBitsAll; zeros(origBaseLen - length(baseBitsAll), 1)];
    end

    if length(allEnhBits) >= origEnhLen
        enhBits = allEnhBits(1:origEnhLen);
    else
        enhBits = [allEnhBits; zeros(origEnhLen - length(allEnhBits), 1)];
    end

    demodInfo = struct();
    demodInfo.baseEVM = sqrt(baseEVM) * 100;
    demodInfo.enhEVM = sqrt(enhEVM) * 100;
    demodInfo.headerValid = headerValid;
    demodInfo.nOFDMSymbols = nOFDMSymbols;
end
