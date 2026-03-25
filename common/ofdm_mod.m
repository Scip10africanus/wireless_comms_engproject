function [txWave, ofdmInfo] = ofdm_mod(baseBits, enhBits, params)
% ofdm_mod - OFDM modulator with unequal error protection
% base layer goes on critical subcarriers (QPSK), enh on the rest (64-QAM)
% header is prepended to base bits so RX knows what to expect

    if nargin < 3, params = struct(); end
    if ~isfield(params, 'nFFT'),     params.nFFT = 64; end
    if ~isfield(params, 'nCP'),      params.nCP = 16; end
    if ~isfield(params, 'mcs_base'), params.mcs_base = 2; end
    if ~isfield(params, 'mcs_enh'),  params.mcs_enh  = 6; end

    % MCS table - indices 0-2 are QPSK, 3-5 are 16QAM, 6-7 are 64QAM
    % need to keep this in sync with ga_parameter_optimizer
    MCS_MOD = [4, 4, 4, 16, 16, 16, 64, 64];
    baseMod = MCS_MOD(max(1, min(8, double(params.mcs_base) + 1)));
    enhMod  = MCS_MOD(max(1, min(8, double(params.mcs_enh)  + 1)));

    nFFT = params.nFFT;
    nCP = params.nCP;

    baseBits = baseBits(:);
    enhBits = enhBits(:);

    % set up the subcarrier map - 40% critical for base, rest for enhancement
    scConfig = struct();
    scConfig.criticalMod    = baseMod;
    scConfig.noncriticalMod = enhMod;
    scConfig.normalMod = 16;
    scConfig.criticalRatio = 0.4;
    scConfig.normalRatio = 0.0;
    scMap = subcarrier_mapper(nFFT, scConfig);

    % figure out which indices are critical vs noncritical
    criticalIdx = [];
    noncriticalIdx = [];
    for k = 1:nFFT
        if strcmp(scMap.importance{k}, 'critical')
            criticalIdx = [criticalIdx; k];
        elseif strcmp(scMap.importance{k}, 'noncritical')
            noncriticalIdx = [noncriticalIdx; k];
        end
    end

    nCritical = length(criticalIdx);
    nNoncritical = length(noncriticalIdx);

    bitsPerBaseSymbol = log2(baseMod);
    bitsPerEnhSymbol  = log2(enhMod);

    baseBitsPerOFDM = nCritical * bitsPerBaseSymbol;
    enhBitsPerOFDM = nNoncritical * bitsPerEnhSymbol;

    % encode header and prepend to base bits (header is also on critical SCs)
    headerBits = [];
    if isfield(params, 'headerParams')
        headerBits = header_codec('encode', params.headerParams);
        headerBits = headerBits(:);
    end
    allBaseBits = [headerBits; baseBits];

    % work out how many OFDM symbols we need
    nBaseSymbols = ceil(length(allBaseBits) / baseBitsPerOFDM);
    nEnhSymbols = ceil(length(enhBits) / enhBitsPerOFDM);
    nOFDMSymbols = max(nBaseSymbols, nEnhSymbols);
    if nOFDMSymbols == 0, nOFDMSymbols = 1; end

    % save original lengths before zero-padding
    origBaseBitsLen = length(baseBits);
    origEnhBitsLen = length(enhBits);
    origAllBaseBitsLen = length(allBaseBits);

    % zero-pad both bit streams to fill complete symbols
    totalBaseBits = nOFDMSymbols * baseBitsPerOFDM;
    totalEnhBits = nOFDMSymbols * enhBitsPerOFDM;
    allBaseBits = [allBaseBits; zeros(totalBaseBits - length(allBaseBits), 1)];
    enhBits = [enhBits; zeros(totalEnhBits - length(enhBits), 1)];

    pilotValue = 1 + 0j;

    % build each OFDM symbol
    txWave = zeros(nOFDMSymbols * (nFFT + nCP), 1);

    for sym = 1:nOFDMSymbols
        freqDomain = zeros(nFFT, 1);

        % base bits onto critical subcarriers
        bStart = (sym-1) * baseBitsPerOFDM + 1;
        bEnd = bStart + baseBitsPerOFDM - 1;
        baseSymbols = qammod(allBaseBits(bStart:bEnd), baseMod, 'InputType', 'bit', 'UnitAveragePower', true);
        freqDomain(criticalIdx) = baseSymbols;

        % enhancement bits onto noncritical subcarriers
        eStart = (sym-1) * enhBitsPerOFDM + 1;
        eEnd = eStart + enhBitsPerOFDM - 1;
        enhSymbols = qammod(enhBits(eStart:eEnd), enhMod, 'InputType', 'bit', 'UnitAveragePower', true);
        freqDomain(noncriticalIdx) = enhSymbols;

        for p = scMap.pilotIndices
            freqDomain(p) = pilotValue;
        end

        timeDomain = ifft(freqDomain, nFFT);
        symWithCP = [timeDomain(end-nCP+1:end); timeDomain];

        outStart = (sym-1) * (nFFT + nCP) + 1;
        txWave(outStart:outStart + nFFT + nCP - 1) = symWithCP;
    end

    % pack everything the demodulator will need into ofdmInfo
    ofdmInfo = struct();
    ofdmInfo.nFFT = nFFT;
    ofdmInfo.nCP = nCP;
    ofdmInfo.nOFDMSymbols = nOFDMSymbols;
    ofdmInfo.nHeaderBits = length(headerBits);
    ofdmInfo.nBaseBits = origBaseBitsLen;
    ofdmInfo.nEnhBits = origEnhBitsLen;
    ofdmInfo.criticalIdx = criticalIdx;
    ofdmInfo.noncriticalIdx = noncriticalIdx;
    ofdmInfo.pilotIndices = scMap.pilotIndices;
    ofdmInfo.scMap = scMap;
    ofdmInfo.baseBitsPerOFDM = baseBitsPerOFDM;
    ofdmInfo.enhBitsPerOFDM = enhBitsPerOFDM;
    ofdmInfo.origBaseBitsLen = origBaseBitsLen;
    ofdmInfo.origEnhBitsLen = origEnhBitsLen;
    ofdmInfo.origAllBaseBitsLen = origAllBaseBitsLen;
    ofdmInfo.baseMod = baseMod;
    ofdmInfo.enhMod  = enhMod;
end
