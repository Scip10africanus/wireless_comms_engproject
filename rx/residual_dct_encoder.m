function [enhancementBits, encInfo] = residual_dct_encoder(originalImage, baseImage, params)
% RESIDUAL_DCT_ENCODER - Encode enhancement layer (residual) using DCT
%
% Calculates residual R = I_original - I_base, then applies 8x8 DCT.
% The residual contains edge and texture information.
%
% The enhancement layer provides:
%   - Sharpness and fine details
%   - Lower priority data (64-QAM + weak FEC)
%
% Inputs:
%   originalImage - Original input image (double)
%   baseImage     - Reconstructed base image from wavelet encoder
%   params        - Structure with:
%       .dctBlockSize     - DCT block size (default: 8)
%       .coeffOrdering    - 'zigzag' or 'raster'
%       .bitsPerCoeff     - Bits per DCT coefficient (default: 8)
%       .maxCoeffsPerBlock- Max coefficients to encode per block (default: 64)
%       .quantTable       - Quantization table (optional)
%       .residualScale    - Scale factor before encoding (default: 1.0)
%
% Outputs:
%   enhancementBits - Encoded bits for transmission
%   encInfo         - Encoding statistics

    %% Default parameters
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
        params.maxCoeffsPerBlock = params.dctBlockSize^2;
    end
    if ~isfield(params, 'residualScale')
        params.residualScale = 1.0;
    end

    %% Convert to double
    originalImage = double(originalImage);
    baseImage = double(baseImage);

    [height, width, nChannels] = size(originalImage);
    blockSize = params.dctBlockSize;

    %% Calculate residual
    residual = originalImage - baseImage;

    % Scale residual
    residual = residual * params.residualScale;

    encInfo = struct();
    encInfo.residualMean = mean(residual(:));
    encInfo.residualStd = std(residual(:));
    encInfo.residualEnergy = sum(residual(:).^2);
    encInfo.residualMin = min(residual(:));
    encInfo.residualMax = max(residual(:));

    %% Calculate block dimensions
    nBlocksH = ceil(height / blockSize);
    nBlocksW = ceil(width / blockSize);
    nBlocks = nBlocksH * nBlocksW;

    encInfo.nBlocks = nBlocks;
    encInfo.nBlocksH = nBlocksH;
    encInfo.nBlocksW = nBlocksW;

    %% Pad residual to fit blocks
    paddedH = nBlocksH * blockSize;
    paddedW = nBlocksW * blockSize;

    %% Process each channel
    allDctCoeffs = [];

    for ch = 1:nChannels
        % Pad channel
        paddedResidual = zeros(paddedH, paddedW);
        paddedResidual(1:height, 1:width) = residual(:,:,ch);

        % Process each block
        for bh = 1:nBlocksH
            for bw = 1:nBlocksW
                % Extract block
                rowStart = (bh-1)*blockSize + 1;
                rowEnd = bh*blockSize;
                colStart = (bw-1)*blockSize + 1;
                colEnd = bw*blockSize;

                block = paddedResidual(rowStart:rowEnd, colStart:colEnd);

                % Apply DCT
                dctBlock = dct2(block);

                % Quantize if table provided
                if isfield(params, 'quantTable') && ~isempty(params.quantTable)
                    dctBlock = round(dctBlock ./ params.quantTable);
                end

                % Reorder to vector
                coeffVector = reorder_block_to_vector(dctBlock, params.coeffOrdering);

                % Truncate to max coefficients
                coeffVector = coeffVector(1:params.maxCoeffsPerBlock);

                % Clip to bit range
                maxVal = 2^(params.bitsPerCoeff - 1) - 1;
                coeffVector = max(-maxVal-1, min(maxVal, round(coeffVector)));

                allDctCoeffs = [allDctCoeffs; coeffVector];
            end
        end
    end

    %% Convert to bits
    enhancementBits = coefficients_to_bits(allDctCoeffs, params.bitsPerCoeff);

    %% Statistics
    encInfo.totalCoeffs = length(allDctCoeffs);
    encInfo.totalBits = length(enhancementBits);
    encInfo.coeffsPerBlock = params.maxCoeffsPerBlock;
    encInfo.dctMean = mean(allDctCoeffs);
    encInfo.dctStd = std(allDctCoeffs);

    % Energy distribution
    dcCoeffs = allDctCoeffs(1:params.maxCoeffsPerBlock:end);
    acCoeffs = allDctCoeffs;
    acCoeffs(1:params.maxCoeffsPerBlock:end) = [];
    encInfo.dcEnergy = sum(dcCoeffs.^2);
    encInfo.acEnergy = sum(acCoeffs.^2);
end

%% Coefficient Ordering

function vector = reorder_block_to_vector(block, ordering)
% REORDER_BLOCK_TO_VECTOR - Convert 2D block to 1D vector

    blockSize = size(block, 1);

    switch ordering
        case 'zigzag'
            zigzagIdx = generate_zigzag_indices(blockSize);
            vector = zeros(blockSize^2, 1);
            for i = 1:blockSize^2
                [r, c] = ind2sub([blockSize, blockSize], zigzagIdx(i));
                vector(i) = block(r, c);
            end

        case 'raster'
            vector = block(:);

        otherwise
            vector = block(:);
    end
end

function indices = generate_zigzag_indices(n)
% GENERATE_ZIGZAG_INDICES - Zigzag scan order for nxn block

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

%% Bit Conversion

function bits = coefficients_to_bits(coeffs, bitsPerCoeff)
% COEFFICIENTS_TO_BITS - Convert signed coefficients to bits

    nCoeffs = length(coeffs);
    bits = zeros(nCoeffs * bitsPerCoeff, 1);
    maxVal = 2^bitsPerCoeff;

    for i = 1:nCoeffs
        val = coeffs(i);

        % Two's complement for negative values
        if val < 0
            val = val + maxVal;
        end
        val = mod(val, maxVal);

        % MSB first
        for b = bitsPerCoeff:-1:1
            bits((i-1)*bitsPerCoeff + (bitsPerCoeff - b + 1)) = mod(val, 2);
            val = floor(val / 2);
        end
    end
end

%% Quantization Tables

function qtable = get_jpeg_qtable(quality)
% GET_JPEG_QTABLE - Standard JPEG quantization table scaled by quality
%
% quality: 1-100 (higher = better quality, less compression)

    if nargin < 1
        quality = 50;
    end

    baseTable = [
        16  11  10  16  24  40  51  61;
        12  12  14  19  26  58  60  55;
        14  13  16  24  40  57  69  56;
        14  17  22  29  51  87  80  62;
        18  22  37  56  68 109 103  77;
        24  35  55  64  81 104 113  92;
        49  64  78  87 103 121 120 101;
        72  92  95  98 112 100 103  99
    ];

    if quality < 50
        scale = 5000 / quality;
    else
        scale = 200 - 2 * quality;
    end

    qtable = round(baseTable * scale / 100);
    qtable = max(1, min(255, qtable));
end
