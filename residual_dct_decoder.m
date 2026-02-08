function [residual, residualInfo] = residual_dct_decoder(enhancementBits, params)
% RESIDUAL_DCT_DECODER - Decode DCT enhancement layer (residual)
%
% Reconstructs the residual image R = I_original - I_base from DCT
% coefficients transmitted with lower protection (64-QAM, weaker FEC).
%
% This is Layer 2 of the two-layer residual coding system:
%   Layer 1 (Base): DWT LL subband -> provides structure
%   Layer 2 (Enhancement): DCT of residual -> provides sharpness/detail
%
% The residual contains:
%   - Edge information
%   - Texture details
%   - High-frequency content
%
% =========================================================================
% TODO: Confirm with TX team:
%   - DCT block size (8x8 standard?)
%   - Coefficient ordering (zigzag?)
%   - Which DCT coefficients are transmitted (all? truncated?)
%   - Quantization table for residual
%   - Bits per coefficient
% =========================================================================
%
% Inputs:
%   enhancementBits - Received bits for enhancement layer (64-QAM)
%   params          - Structure with:
%       .imageHeight    - Original image height
%       .imageWidth     - Original image width
%       .nChannels      - Number of color channels
%       .dctBlockSize   - DCT block size (default: 8)
%       .coeffOrdering  - 'zigzag' or 'raster'
%       .bitsPerCoeff   - Bits per DCT coefficient
%       .maxCoeffsPerBlock - Max coefficients per block to decode (for truncation)
%       .quantTable     - Quantization table (optional)
%
% Outputs:
%   residual     - Reconstructed residual image (double, signed values)
%   residualInfo - Structure with decoding statistics

    %% Default parameters
    if ~isfield(params, 'nChannels')
        params.nChannels = 1;
    end
    if ~isfield(params, 'dctBlockSize')
        params.dctBlockSize = 8;
    end
    if ~isfield(params, 'coeffOrdering')
        params.coeffOrdering = 'zigzag';
    end
    if ~isfield(params, 'bitsPerCoeff')
        params.bitsPerCoeff = 8;
    end
    if ~isfield(params, 'maxCoeffsPerBlock')
        % Default: all coefficients in block
        params.maxCoeffsPerBlock = params.dctBlockSize^2;
    end

    blockSize = params.dctBlockSize;
    coeffsPerBlock = params.maxCoeffsPerBlock;

    %% Calculate dimensions
    nBlocksH = ceil(params.imageHeight / blockSize);
    nBlocksW = ceil(params.imageWidth / blockSize);
    nBlocks = nBlocksH * nBlocksW;

    residualInfo = struct();
    residualInfo.nBlocksH = nBlocksH;
    residualInfo.nBlocksW = nBlocksW;
    residualInfo.nBlocks = nBlocks;
    residualInfo.coeffsPerBlock = coeffsPerBlock;

    %% Convert bits to DCT coefficients
    enhancementBits = enhancementBits(:);
    totalCoeffs = nBlocks * coeffsPerBlock * params.nChannels;
    expectedBits = totalCoeffs * params.bitsPerCoeff;

    % Pad if necessary
    if length(enhancementBits) < expectedBits
        residualInfo.bitsPadded = expectedBits - length(enhancementBits);
        residualInfo.bitsReceived = length(enhancementBits);
        enhancementBits = [enhancementBits; zeros(expectedBits - length(enhancementBits), 1)];
    else
        residualInfo.bitsPadded = 0;
        residualInfo.bitsReceived = expectedBits;
    end
    enhancementBits = enhancementBits(1:expectedBits);

    %% Decode coefficients
    dctCoeffs = bits_to_signed_coefficients(enhancementBits, totalCoeffs, params);

    %% Dequantize if quantization table provided
    if isfield(params, 'quantTable') && ~isempty(params.quantTable)
        dctCoeffs = dequantize_coeffs(dctCoeffs, params.quantTable, nBlocks, params);
    end

    %% Reconstruct residual for each channel
    paddedH = nBlocksH * blockSize;
    paddedW = nBlocksW * blockSize;
    residual = zeros(paddedH, paddedW, params.nChannels);

    for ch = 1:params.nChannels
        chStart = (ch-1) * nBlocks * coeffsPerBlock + 1;
        chEnd = ch * nBlocks * coeffsPerBlock;
        chCoeffs = dctCoeffs(chStart:chEnd);

        % Reshape into blocks
        blockCoeffs = reshape(chCoeffs, coeffsPerBlock, nBlocks);

        % Process each block
        blockIdx = 1;
        for bh = 1:nBlocksH
            for bw = 1:nBlocksW
                % Get coefficients for this block
                coeffVector = blockCoeffs(:, blockIdx);

                % Create full block (pad with zeros if truncated)
                fullCoeffs = zeros(blockSize^2, 1);
                fullCoeffs(1:coeffsPerBlock) = coeffVector;

                % Reorder to 2D block
                dctBlock = reorder_to_block(fullCoeffs, blockSize, params.coeffOrdering);

                % Inverse DCT
                pixelBlock = idct2(dctBlock);

                % Place in output
                rowStart = (bh-1)*blockSize + 1;
                rowEnd = bh*blockSize;
                colStart = (bw-1)*blockSize + 1;
                colEnd = bw*blockSize;
                residual(rowStart:rowEnd, colStart:colEnd, ch) = pixelBlock;

                blockIdx = blockIdx + 1;
            end
        end
    end

    %% Crop to original size
    residual = residual(1:params.imageHeight, 1:params.imageWidth, :);

    %% Calculate residual statistics
    residualInfo.meanValue = mean(residual(:));
    residualInfo.stdValue = std(residual(:));
    residualInfo.minValue = min(residual(:));
    residualInfo.maxValue = max(residual(:));
    residualInfo.energy = sum(residual(:).^2);

    % The residual should be roughly zero-mean (difference signal)
    residualInfo.dcBias = abs(residualInfo.meanValue);
end

%% Helper Functions

function coeffs = bits_to_signed_coefficients(bits, nCoeffs, params)
% BITS_TO_SIGNED_COEFFICIENTS - Convert bits to signed coefficient values
%
% Residual coefficients can be positive or negative

    bitsPerCoeff = params.bitsPerCoeff;
    coeffs = zeros(nCoeffs, 1);

    for i = 1:nCoeffs
        startBit = (i-1) * bitsPerCoeff + 1;
        endBit = i * bitsPerCoeff;

        if endBit > length(bits)
            break;
        end

        coeffBits = bits(startBit:endBit);

        % Convert to integer (MSB first)
        val = 0;
        for b = 1:bitsPerCoeff
            val = val * 2 + coeffBits(b);
        end

        % Two's complement for signed values
        maxVal = 2^bitsPerCoeff;
        if val >= maxVal/2
            val = val - maxVal;
        end

        coeffs(i) = val;
    end
end

function coeffs = dequantize_coeffs(coeffs, quantTable, nBlocks, params)
% DEQUANTIZE_COEFFS - Apply inverse quantization

    blockSize = params.dctBlockSize;
    coeffsPerBlock = params.maxCoeffsPerBlock;

    % Flatten quantization table in same order as coefficients
    if strcmp(params.coeffOrdering, 'zigzag')
        zigzagIdx = generate_zigzag_indices(blockSize);
        quantFlat = quantTable(zigzagIdx);
    else
        quantFlat = quantTable(:);
    end
    quantFlat = quantFlat(1:coeffsPerBlock);

    % Apply dequantization to each block
    for block = 1:nBlocks
        startIdx = (block-1) * coeffsPerBlock + 1;
        endIdx = block * coeffsPerBlock;
        coeffs(startIdx:endIdx) = coeffs(startIdx:endIdx) .* quantFlat;
    end
end

function block = reorder_to_block(vector, blockSize, ordering)
% REORDER_TO_BLOCK - Convert coefficient vector to 2D DCT block

    block = zeros(blockSize, blockSize);

    switch ordering
        case 'zigzag'
            zigzagIdx = generate_zigzag_indices(blockSize);
            for i = 1:length(vector)
                [r, c] = ind2sub([blockSize, blockSize], zigzagIdx(i));
                block(r, c) = vector(i);
            end

        case 'raster'
            block = reshape(vector, blockSize, blockSize)';

        otherwise
            block = reshape(vector, blockSize, blockSize)';
    end
end

function indices = generate_zigzag_indices(n)
% GENERATE_ZIGZAG_INDICES - Generate zigzag scan order for nxn block

    indices = zeros(1, n*n);
    idx = 1;

    for s = 1:(2*n - 1)
        if mod(s, 2) == 1
            if s <= n
                row = s; col = 1;
            else
                row = n; col = s - n + 1;
            end
            while row >= 1 && col <= n
                indices(idx) = (row-1)*n + col;
                idx = idx + 1;
                row = row - 1;
                col = col + 1;
            end
        else
            if s <= n
                row = 1; col = s;
            else
                row = s - n + 1; col = n;
            end
            while row <= n && col >= 1
                indices(idx) = (row-1)*n + col;
                idx = idx + 1;
                row = row + 1;
                col = col - 1;
            end
        end
    end
end
