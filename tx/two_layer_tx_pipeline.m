function [txSignal, txInfo] = two_layer_tx_pipeline(image, params)
% TWO_LAYER_TX_PIPELINE - Complete transmitter for two-layer residual coding
%
% Implements the transmitter chain:
%   1. Wavelet decomposition → Extract LL subband (base layer)
%   2. Reconstruct base at TX
%   3. Calculate residual: R = Original - Base
%   4. DCT on residual (enhancement layer)
%   5. Map layers to different subcarriers with UEP:
%      - Base → QPSK → Strong FEC → Priority subcarriers
%      - Enhancement → 64-QAM → Weak FEC → Remaining subcarriers
%   6. OFDM modulation (IFFT + CP)
%
% Inputs:
%   image  - Input image (uint8 or double)
%   params - Configuration structure (see config_two_layer.m)
%
% Outputs:
%   txSignal - Transmitted time-domain OFDM signal
%   txInfo   - Structure with all transmission info (needed by RX)

    %% Initialize
    txInfo = struct();
    txInfo.success = true;
    startTime = tic;

    image = double(image);
    [height, width, nChannels] = size(image);

    txInfo.imageSize = [height, width, nChannels];

    if params.verbose
        print_tx_header(params);
    end

    %% ==================== STAGE 1: HEADER GENERATION ====================
    if params.verbose
        fprintf('\n[1/6] Header Generation...\n');
    end

    % Header will be finalized after we know data lengths
    % (placeholder for now, updated after encoding)

    %% ==================== STAGE 2: BASE LAYER ENCODING ====================
    if params.verbose
        fprintf('\n[2/6] Base Layer (Wavelet) Encoding...\n');
    end

    waveletParams = struct();
    waveletParams.waveletType = params.waveletType;
    waveletParams.waveletLevels = params.waveletLevels;
    waveletParams.bitsPerCoeff = params.baseBitsPerCoeff;

    [baseBits, baseImage, baseEncInfo] = wavelet_base_encoder(image, waveletParams);
    txInfo.baseEncoder = baseEncInfo;

    if params.verbose
        fprintf('  ✓ LL subband: %dx%d\n', baseEncInfo.llSize(1), baseEncInfo.llSize(2));
        fprintf('  ✓ Base bits: %d (%.1f KB)\n', length(baseBits), length(baseBits)/8/1024);
    end

    %% ==================== STAGE 3: ENHANCEMENT LAYER ENCODING ====================
    if params.verbose
        fprintf('\n[3/6] Enhancement Layer (DCT Residual) Encoding...\n');
    end

    dctParams = struct();
    dctParams.dctBlockSize = params.dctBlockSize;
    dctParams.coeffOrdering = params.coeffOrdering;
    dctParams.bitsPerCoeff = params.enhancementBitsPerCoeff;
    dctParams.maxCoeffsPerBlock = params.maxDctCoeffs;
    dctParams.residualScale = params.residualScale;

    [enhancementBits, enhEncInfo] = residual_dct_encoder(image, baseImage, dctParams);
    txInfo.enhancementEncoder = enhEncInfo;

    if params.verbose
        fprintf('  ✓ Residual std: %.2f\n', enhEncInfo.residualStd);
        fprintf('  ✓ Enhancement bits: %d (%.1f KB)\n', length(enhancementBits), length(enhancementBits)/8/1024);
    end

    %% ==================== STAGE 4: FINALIZE HEADER ====================
    if params.verbose
        fprintf('\n[4/6] Finalizing Header...\n');
    end

    % Build header with all parameters and data lengths
    headerParams = params;
    headerParams.imageHeight = height;
    headerParams.imageWidth = width;
    headerParams.nChannels = nChannels;
    headerParams.baseDataLength = length(baseBits);
    headerParams.enhDataLength = length(enhancementBits);

    headerBits = header_codec('encode', headerParams);
    txInfo.headerBits = headerBits;

    % Prepend header to base layer (transmitted with same high protection)
    baseBitsWithHeader = [headerBits; baseBits];

    if params.verbose
        fprintf('  ✓ Header: %d bits\n', length(headerBits));
        fprintf('  ✓ Base + header: %d bits\n', length(baseBitsWithHeader));
    end

    %% ==================== STAGE 5: FEC ENCODING ====================
    if params.verbose
        fprintf('\n[5/6] FEC Encoding...\n');
    end

    if params.useFEC
        [baseBitsEncoded, baseFecInfo] = encode_fec(baseBitsWithHeader, params.baseFEC, params.baseFECRate);
        [enhBitsEncoded, enhFecInfo] = encode_fec(enhancementBits, params.enhancementFEC, params.enhancementFECRate);
        txInfo.baseFEC = baseFecInfo;
        txInfo.enhancementFEC = enhFecInfo;

        if params.verbose
            fprintf('  ✓ Base FEC: %s, rate %.2f\n', params.baseFEC, params.baseFECRate);
            fprintf('  ✓ Enhancement FEC: %s, rate %.2f\n', params.enhancementFEC, params.enhancementFECRate);
        end
    else
        baseBitsEncoded = baseBitsWithHeader;
        enhBitsEncoded = enhancementBits;
        if params.verbose
            fprintf('  (FEC disabled)\n');
        end
    end

    %% ==================== STAGE 6: MODULATION & OFDM ====================
    if params.verbose
        fprintf('\n[6/6] QAM Modulation & OFDM...\n');
    end

    % Calculate required OFDM symbols
    nBaseSubcarriers = length(params.baseSubcarriers);
    nEnhSubcarriers = length(params.enhancementSubcarriers);

    baseBitsPerSymbol = nBaseSubcarriers * log2(params.baseMod);
    enhBitsPerSymbol = nEnhSubcarriers * log2(params.enhancementMod);

    nSymbolsBase = ceil(length(baseBitsEncoded) / baseBitsPerSymbol);
    nSymbolsEnh = ceil(length(enhBitsEncoded) / enhBitsPerSymbol);
    nOFDMSymbols = max(nSymbolsBase, nSymbolsEnh);

    % Pad bits to fill symbols
    baseBitsPadded = pad_bits(baseBitsEncoded, nOFDMSymbols * baseBitsPerSymbol);
    enhBitsPadded = pad_bits(enhBitsEncoded, nOFDMSymbols * enhBitsPerSymbol);

    % QAM modulation
    baseSymbols = qammod(baseBitsPadded, params.baseMod, 'InputType', 'bit', 'UnitAveragePower', true);
    enhSymbols = qammod(enhBitsPadded, params.enhancementMod, 'InputType', 'bit', 'UnitAveragePower', true);

    % Reshape for subcarrier mapping
    baseSymbols = reshape(baseSymbols, nBaseSubcarriers, nOFDMSymbols);
    enhSymbols = reshape(enhSymbols, nEnhSubcarriers, nOFDMSymbols);

    % Create frequency-domain OFDM symbols
    ofdmFreq = zeros(params.nFFT, nOFDMSymbols);

    % Map base layer to priority subcarriers
    ofdmFreq(params.baseSubcarriers, :) = baseSymbols;

    % Map enhancement layer to remaining subcarriers
    ofdmFreq(params.enhancementSubcarriers, :) = enhSymbols;

    % Add pilots
    if ~isempty(params.pilotIndices)
        pilotSymbol = 1 + 0j;  % Simple pilot
        ofdmFreq(params.pilotIndices, :) = pilotSymbol;
    end

    txInfo.nOFDMSymbols = nOFDMSymbols;
    txInfo.baseBitsTransmitted = length(baseBitsPadded);
    txInfo.enhBitsTransmitted = length(enhBitsPadded);

    if params.verbose
        fprintf('  ✓ Base: %d-QAM on %d subcarriers\n', params.baseMod, nBaseSubcarriers);
        fprintf('  ✓ Enhancement: %d-QAM on %d subcarriers\n', params.enhancementMod, nEnhSubcarriers);
        fprintf('  ✓ Total OFDM symbols: %d\n', nOFDMSymbols);
    end

    %% ==================== STAGE 5: OFDM MODULATION ====================
    if params.verbose
        fprintf('\n[5/5] OFDM Modulation (IFFT + CP)...\n');
    end

    % IFFT
    ofdmTime = ifft(ofdmFreq, params.nFFT);

    % Add cyclic prefix
    ofdmWithCP = [ofdmTime(end-params.nCP+1:end, :); ofdmTime];

    % Serialize
    txSignal = ofdmWithCP(:);

    txInfo.signalLength = length(txSignal);
    txInfo.signalPower = mean(abs(txSignal).^2);

    if params.verbose
        fprintf('  ✓ Signal length: %d samples\n', length(txSignal));
        fprintf('  ✓ Signal power: %.4f\n', txInfo.signalPower);
    end

    %% Store additional info for RX
    txInfo.baseBits = baseBits;           % Original bits (for BER calculation)
    txInfo.enhancementBits = enhancementBits;
    txInfo.baseImage = baseImage;         % For debugging
    txInfo.processingTime = toc(startTime);

    if params.verbose
        print_tx_footer(txInfo);
    end
end

%% FEC Functions (Placeholder)

function [encodedBits, fecInfo] = encode_fec(bits, fecType, rate)
% ENCODE_FEC - FEC encoding placeholder
%
% TODO: Implement actual FEC encoding

    fecInfo = struct();
    fecInfo.type = fecType;
    fecInfo.rate = rate;
    fecInfo.inputBits = length(bits);

    switch fecType
        case 'none'
            encodedBits = bits;

        case 'repetition'
            % Simple repetition code (for testing)
            repFactor = round(1/rate);
            encodedBits = repmat(bits, repFactor, 1);
            encodedBits = encodedBits(:);

        case 'convolutional'
            % TODO: Implement convolutional encoder
            % For now, just add dummy parity bits
            nParity = round(length(bits) * (1/rate - 1));
            encodedBits = [bits; zeros(nParity, 1)];

        otherwise
            encodedBits = bits;
    end

    fecInfo.outputBits = length(encodedBits);
end

%% Utility Functions

function padded = pad_bits(bits, targetLength)
% PAD_BITS - Pad bit vector to target length

    if length(bits) < targetLength
        padded = [bits; zeros(targetLength - length(bits), 1)];
    else
        padded = bits(1:targetLength);
    end
end

function print_tx_header(params)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║          Two-Layer TX Pipeline (Wavelet + DCT)               ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\nConfiguration:\n');
    fprintf('  Image: %dx%d, FFT: %d, CP: %d\n', ...
        params.imageHeight, params.imageWidth, params.nFFT, params.nCP);
    fprintf('  Base: %s L%d → %d-QAM\n', ...
        params.waveletType, params.waveletLevels, params.baseMod);
    fprintf('  Enhancement: %dx%d DCT → %d-QAM\n', ...
        params.dctBlockSize, params.dctBlockSize, params.enhancementMod);
end

function print_tx_footer(txInfo)
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════════\n');
    fprintf('TX complete in %.3f seconds\n', txInfo.processingTime);
    fprintf('  Total bits: %d base + %d enhancement = %d\n', ...
        txInfo.baseBitsTransmitted, txInfo.enhBitsTransmitted, ...
        txInfo.baseBitsTransmitted + txInfo.enhBitsTransmitted);
    fprintf('════════════════════════════════════════════════════════════════\n');
end
