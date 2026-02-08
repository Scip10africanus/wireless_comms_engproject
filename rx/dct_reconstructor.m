function [image, coeffInfo] = dct_reconstructor(dctCoeffs, params)
% DCT_RECONSTRUCTOR - Reconstruct image from received DCT coefficients
%
% Performs inverse DCT to recover image from frequency-domain coefficients.
% Designed for content-aware transmission where low-freq and high-freq
% coefficients may have different error characteristics.
%
% =========================================================================
% TODO: Adapt these after meeting with TX team:
%   - Confirm DCT block size (8x8 is standard JPEG, but could differ)
%   - Confirm coefficient ordering (zigzag, raster, or custom)
%   - Confirm how coefficients are packed (interleaved vs sequential blocks)
%   - Confirm if any quantization table is used
% =========================================================================
%
% Inputs:
%   dctCoeffs - Received DCT coefficients (vector or matrix)
%   params    - Structure with:
%       .imageHeight   - Original image height
%       .imageWidth    - Original image width
%       .blockSize     - DCT block size (default: 8)
%       .nChannels     - Color channels (default: 1)
%       .coeffOrdering - 'zigzag', 'raster', or 'custom' (default: 'zigzag')
%       .freqThreshold - Index separating low/high freq (default: 21 for 8x8)
%       .quantTable    - Quantization table if used (optional)
%
% Outputs:
%   image     - Reconstructed image (uint8)
%   coeffInfo - Structure with coefficient statistics for feedback

    %% Default parameters
    % TODO: Update defaults based on TX implementation
    if ~isfield(params, 'blockSize')
        params.blockSize = 8;  % Standard 8x8 DCT block
    end
    if ~isfield(params, 'nChannels')
        params.nChannels = 1;
    end
    if ~isfield(params, 'coeffOrdering')
        params.coeffOrdering = 'zigzag';
    end
    if ~isfield(params, 'freqThreshold')
        % Default: first 21 coefficients are "low frequency" in 8x8 block
        % This is roughly the upper-left triangle
        params.freqThreshold = 21;
    end

    blockSize = params.blockSize;
    nBlocksH = ceil(params.imageHeight / blockSize);
    nBlocksW = ceil(params.imageWidth / blockSize);
    nBlocks = nBlocksH * nBlocksW;
    coeffsPerBlock = blockSize^2;

    %% Reshape coefficients into blocks
    % TODO: Adapt based on how TX packs the coefficients
    dctCoeffs = dctCoeffs(:);
    expectedCoeffs = nBlocks * coeffsPerBlock * params.nChannels;

    if length(dctCoeffs) < expectedCoeffs
        warning('Fewer coefficients than expected. Padding with zeros.');
        dctCoeffs = [dctCoeffs; zeros(expectedCoeffs - length(dctCoeffs), 1)];
    end
    dctCoeffs = dctCoeffs(1:expectedCoeffs);

    %% Initialize output
    paddedH = nBlocksH * blockSize;
    paddedW = nBlocksW * blockSize;
    reconstructed = zeros(paddedH, paddedW, params.nChannels);

    %% Process each channel
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

                % Reorder from zigzag/raster to 2D block
                dctBlock = reorder_to_block(coeffVector, blockSize, params.coeffOrdering);

                % Optional: Dequantize
                if isfield(params, 'quantTable') && ~isempty(params.quantTable)
                    dctBlock = dctBlock .* params.quantTable;
                end

                % Inverse DCT
                pixelBlock = idct2(dctBlock);

                % Place in output image
                rowStart = (bh-1)*blockSize + 1;
                rowEnd = bh*blockSize;
                colStart = (bw-1)*blockSize + 1;
                colEnd = bw*blockSize;
                reconstructed(rowStart:rowEnd, colStart:colEnd, ch) = pixelBlock;

                blockIdx = blockIdx + 1;
            end
        end
    end

    %% Crop to original size and convert to uint8
    reconstructed = reconstructed(1:params.imageHeight, 1:params.imageWidth, :);

    % Clip to valid range and convert
    reconstructed = max(0, min(255, reconstructed));
    image = uint8(round(reconstructed));

    %% Calculate coefficient statistics for feedback
    coeffInfo = analyze_coefficients(dctCoeffs, params, nBlocks);
end

%% Helper Functions

function block = reorder_to_block(vector, blockSize, ordering)
% REORDER_TO_BLOCK - Convert coefficient vector to 2D DCT block
%
% TODO: Verify this matches TX ordering

    block = zeros(blockSize, blockSize);

    switch ordering
        case 'zigzag'
            % Standard JPEG zigzag ordering
            zigzagIdx = generate_zigzag_indices(blockSize);
            for i = 1:length(vector)
                [r, c] = ind2sub([blockSize, blockSize], zigzagIdx(i));
                block(r, c) = vector(i);
            end

        case 'raster'
            % Simple row-by-row ordering
            block = reshape(vector, blockSize, blockSize)';

        case 'custom'
            % TODO: Implement custom ordering from TX team
            % Placeholder: use raster for now
            block = reshape(vector, blockSize, blockSize)';

        otherwise
            block = reshape(vector, blockSize, blockSize)';
    end
end

function indices = generate_zigzag_indices(n)
% GENERATE_ZIGZAG_INDICES - Generate zigzag scan indices for nxn block

    indices = zeros(1, n*n);
    idx = 1;

    for s = 1:(2*n - 1)
        if mod(s, 2) == 1
            % Moving up-right
            if s <= n
                row = s;
                col = 1;
            else
                row = n;
                col = s - n + 1;
            end
            while row >= 1 && col <= n
                indices(idx) = (row-1)*n + col;
                idx = idx + 1;
                row = row - 1;
                col = col + 1;
            end
        else
            % Moving down-left
            if s <= n
                row = 1;
                col = s;
            else
                row = s - n + 1;
                col = n;
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

function info = analyze_coefficients(coeffs, params, nBlocks)
% ANALYZE_COEFFICIENTS - Analyze coefficient statistics for feedback
%
% Separates low-freq and high-freq analysis for UEP feedback

    coeffsPerBlock = params.blockSize^2;
    threshold = params.freqThreshold;

    % Reshape into blocks
    blockCoeffs = reshape(coeffs(1:nBlocks*coeffsPerBlock), coeffsPerBlock, nBlocks);

    % Separate low and high frequency
    lowFreqCoeffs = blockCoeffs(1:threshold, :);
    highFreqCoeffs = blockCoeffs(threshold+1:end, :);

    % Low frequency statistics (critical data)
    info.lowFreq.mean = mean(abs(lowFreqCoeffs(:)));
    info.lowFreq.std = std(abs(lowFreqCoeffs(:)));
    info.lowFreq.energy = sum(lowFreqCoeffs(:).^2);
    info.lowFreq.count = numel(lowFreqCoeffs);

    % High frequency statistics (non-critical data)
    info.highFreq.mean = mean(abs(highFreqCoeffs(:)));
    info.highFreq.std = std(abs(highFreqCoeffs(:)));
    info.highFreq.energy = sum(highFreqCoeffs(:).^2);
    info.highFreq.count = numel(highFreqCoeffs);

    % Energy ratio (for adaptive threshold feedback)
    totalEnergy = info.lowFreq.energy + info.highFreq.energy;
    info.lowFreqEnergyRatio = info.lowFreq.energy / (totalEnergy + eps);

    % DC component analysis (most critical)
    dcCoeffs = blockCoeffs(1, :);
    info.dc.mean = mean(dcCoeffs);
    info.dc.std = std(dcCoeffs);
end
