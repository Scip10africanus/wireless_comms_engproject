function [baseImage, baseInfo] = wavelet_base_decoder(baseBits, params)
% WAVELET_BASE_DECODER - Decode wavelet (DWT) base layer
%
% Reconstructs the low-resolution base image from the LL subband
% coefficients transmitted with high protection (QPSK + strong FEC).
%
% This is Layer 1 of the two-layer residual coding system:
%   Layer 1 (Base): DWT LL subband -> provides structure/recognizability
%   Layer 2 (Enhancement): DCT residual -> provides sharpness/detail
%
% =========================================================================
% TODO: Confirm with TX team:
%   - Wavelet type (Haar, Daubechies db2/db4, etc.)
%   - Number of decomposition levels
%   - LL coefficient bit depth
%   - Any quantization applied
% =========================================================================
%
% Inputs:
%   baseBits - Received bits for base layer (high priority, QPSK)
%   params   - Structure with:
%       .imageHeight    - Original image height
%       .imageWidth     - Original image width
%       .nChannels      - Number of color channels (default: 1)
%       .waveletType    - Wavelet filter ('haar', 'db2', 'db4', etc.)
%       .waveletLevels  - Number of DWT decomposition levels
%       .bitsPerCoeff   - Bits per wavelet coefficient (default: 8)
%       .quantStep      - Quantization step size (optional)
%
% Outputs:
%   baseImage - Reconstructed base layer image (double, same size as original)
%   baseInfo  - Structure with decoding statistics

    %% Default parameters
    % TODO: Update defaults based on TX implementation
    if ~isfield(params, 'nChannels')
        params.nChannels = 1;
    end
    if ~isfield(params, 'waveletType')
        params.waveletType = 'haar';  % Simple, common choice
    end
    if ~isfield(params, 'waveletLevels')
        params.waveletLevels = 2;     % 2-level decomposition
    end
    if ~isfield(params, 'bitsPerCoeff')
        params.bitsPerCoeff = 8;
    end

    %% Calculate LL subband dimensions
    % After L levels of DWT, LL size = original_size / 2^L
    llHeight = ceil(params.imageHeight / (2^params.waveletLevels));
    llWidth = ceil(params.imageWidth / (2^params.waveletLevels));
    llSize = llHeight * llWidth;

    baseInfo = struct();
    baseInfo.llHeight = llHeight;
    baseInfo.llWidth = llWidth;
    baseInfo.waveletType = params.waveletType;
    baseInfo.waveletLevels = params.waveletLevels;

    %% Convert bits to LL coefficients
    baseBits = baseBits(:);
    expectedBits = llSize * params.nChannels * params.bitsPerCoeff;

    % Pad if necessary
    if length(baseBits) < expectedBits
        baseInfo.bitsPadded = expectedBits - length(baseBits);
        baseBits = [baseBits; zeros(expectedBits - length(baseBits), 1)];
    else
        baseInfo.bitsPadded = 0;
    end
    baseBits = baseBits(1:expectedBits);

    %% Decode coefficients
    llCoeffs = bits_to_coefficients(baseBits, llSize * params.nChannels, params);

    %% Dequantize if quantization was used
    if isfield(params, 'quantStep') && params.quantStep > 0
        llCoeffs = llCoeffs * params.quantStep;
    end

    %% Reconstruct base image for each channel
    baseImage = zeros(params.imageHeight, params.imageWidth, params.nChannels);

    for ch = 1:params.nChannels
        chStart = (ch-1) * llSize + 1;
        chEnd = ch * llSize;
        llChannel = reshape(llCoeffs(chStart:chEnd), [llHeight, llWidth]);

        % Perform inverse DWT to reconstruct base
        baseChannel = inverse_dwt_from_ll(llChannel, params);

        % Resize to original dimensions if needed
        if size(baseChannel, 1) ~= params.imageHeight || ...
           size(baseChannel, 2) ~= params.imageWidth
            baseChannel = imresize_simple(baseChannel, ...
                [params.imageHeight, params.imageWidth]);
        end

        baseImage(:,:,ch) = baseChannel;
    end

    %% Calculate base layer statistics
    baseInfo.meanValue = mean(baseImage(:));
    baseInfo.stdValue = std(baseImage(:));
    baseInfo.minValue = min(baseImage(:));
    baseInfo.maxValue = max(baseImage(:));
    baseInfo.totalCoeffs = llSize * params.nChannels;
    baseInfo.totalBits = expectedBits;
end

%% Core Functions

function coeffs = bits_to_coefficients(bits, nCoeffs, params)
% BITS_TO_COEFFICIENTS - Convert bit vector to coefficient values
%
% TODO: Confirm encoding format with TX team

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

        % Convert from unsigned to signed representation
        % TODO: Confirm range with TX team
        maxVal = 2^bitsPerCoeff;
        if val >= maxVal/2
            val = val - maxVal;  % Two's complement
        end

        coeffs(i) = val;
    end
end

function reconstructed = inverse_dwt_from_ll(llSubband, params)
% INVERSE_DWT_FROM_LL - Reconstruct image from LL subband only
%
% Since we only have the LL (low-low) subband, we set the detail
% subbands (LH, HL, HH) to zero and perform inverse DWT.
%
% This gives us a blurry but recognizable base image.

    waveletType = params.waveletType;
    nLevels = params.waveletLevels;

    % Start with LL subband
    currentApprox = llSubband;

    % Perform inverse DWT for each level
    for level = nLevels:-1:1
        [h, w] = size(currentApprox);

        % Create zero detail subbands (since we don't have them)
        lh = zeros(h, w);  % Horizontal detail
        hl = zeros(h, w);  % Vertical detail
        hh = zeros(h, w);  % Diagonal detail

        % Perform single-level inverse DWT
        currentApprox = idwt2_simple(currentApprox, lh, hl, hh, waveletType);
    end

    reconstructed = currentApprox;
end

function img = idwt2_simple(ll, lh, hl, hh, waveletType)
% IDWT2_SIMPLE - Simple 2D inverse discrete wavelet transform
%
% Reconstructs image from wavelet subbands using specified wavelet.

    % Get wavelet filters
    [loR, hiR] = get_wavelet_filters(waveletType, 'reconstruction');

    [h, w] = size(ll);

    % Upsample and filter columns
    ll_up = upsample_cols(ll);
    lh_up = upsample_cols(lh);
    hl_up = upsample_cols(hl);
    hh_up = upsample_cols(hh);

    % Filter and combine
    l = filter_cols(ll_up, loR) + filter_cols(lh_up, hiR);
    h_band = filter_cols(hl_up, loR) + filter_cols(hh_up, hiR);

    % Upsample and filter rows
    l_up = upsample_rows(l);
    h_up = upsample_rows(h_band);

    img = filter_rows(l_up, loR) + filter_rows(h_up, hiR);

    % Trim to expected size
    img = img(1:2*h, 1:2*w);
end

function [loR, hiR] = get_wavelet_filters(waveletType, mode)
% GET_WAVELET_FILTERS - Return wavelet filter coefficients
%
% TODO: Add more wavelet types as needed

    switch lower(waveletType)
        case 'haar'
            % Haar wavelet (simplest)
            loD = [1 1] / sqrt(2);
            hiD = [-1 1] / sqrt(2);
            loR = loD;
            hiR = [-hiD(2) hiD(1)];

        case 'db2'
            % Daubechies-2
            loD = [0.4829629131445341, 0.8365163037378079, ...
                   0.2241438680420134, -0.1294095225512604];
            hiD = [-loD(4), loD(3), -loD(2), loD(1)];
            loR = fliplr(loD);
            hiR = fliplr(hiD);

        case 'db4'
            % Daubechies-4
            loD = [0.2304, 0.7148, 0.6309, -0.0280, ...
                   -0.1870, 0.0308, 0.0329, -0.0106];
            hiD = fliplr(loD) .* [1 -1 1 -1 1 -1 1 -1];
            loR = fliplr(loD);
            hiR = fliplr(hiD);

        otherwise
            % Default to Haar
            loR = [1 1] / sqrt(2);
            hiR = [1 -1] / sqrt(2);
    end
end

function y = upsample_cols(x)
% UPSAMPLE_COLS - Upsample columns by inserting zeros
    [h, w] = size(x);
    y = zeros(2*h, w);
    y(1:2:end, :) = x;
end

function y = upsample_rows(x)
% UPSAMPLE_ROWS - Upsample rows by inserting zeros
    [h, w] = size(x);
    y = zeros(h, 2*w);
    y(:, 1:2:end) = x;
end

function y = filter_cols(x, filt)
% FILTER_COLS - Filter each column
    [h, w] = size(x);
    filtLen = length(filt);
    y = zeros(h, w);
    for col = 1:w
        y(:, col) = conv(x(:, col), filt(:), 'same');
    end
end

function y = filter_rows(x, filt)
% FILTER_ROWS - Filter each row
    [h, w] = size(x);
    y = zeros(h, w);
    for row = 1:h
        y(row, :) = conv(x(row, :), filt(:)', 'same');
    end
end

function imgOut = imresize_simple(imgIn, newSize)
% IMRESIZE_SIMPLE - Simple bilinear interpolation resize
    [h, w] = size(imgIn);
    newH = newSize(1);
    newW = newSize(2);

    [X, Y] = meshgrid(1:w, 1:h);
    [Xq, Yq] = meshgrid(linspace(1, w, newW), linspace(1, h, newH));

    imgOut = interp2(X, Y, imgIn, Xq, Yq, 'linear', 0);
end
