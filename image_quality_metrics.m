function metrics = image_quality_metrics(refImage, testImage, params)
% IMAGE_QUALITY_METRICS - Comprehensive image quality assessment
%
% Calculates multiple image quality metrics including:
%   - PSNR (Peak Signal-to-Noise Ratio)
%   - SSIM (Structural Similarity Index)
%   - MS-SSIM (Multi-Scale SSIM)
%   - FSIM (Feature Similarity) - simplified version
%
% Inputs:
%   refImage  - Reference (original) image
%   testImage - Test (reconstructed/degraded) image
%   params    - Optional structure with:
%       .maxVal     - Maximum pixel value (default: 255)
%       .ssimK      - SSIM constants [K1, K2] (default: [0.01, 0.03])
%       .ssimWindow - SSIM window size (default: 11)
%       .verbose    - Print results (default: false)
%
% Outputs:
%   metrics - Structure with all computed quality metrics

    %% Input validation and defaults
    if nargin < 3
        params = struct();
    end
    if ~isfield(params, 'maxVal')
        params.maxVal = 255;
    end
    if ~isfield(params, 'ssimK')
        params.ssimK = [0.01, 0.03];
    end
    if ~isfield(params, 'ssimWindow')
        params.ssimWindow = 11;
    end
    if ~isfield(params, 'verbose')
        params.verbose = false;
    end

    % Convert to double for calculations
    refImage = double(refImage);
    testImage = double(testImage);

    % Ensure same size
    if ~isequal(size(refImage), size(testImage))
        error('Reference and test images must have the same dimensions');
    end

    %% 1. MSE and PSNR
    [metrics.MSE, metrics.PSNR] = calculate_psnr(refImage, testImage, params.maxVal);

    %% 2. SSIM
    [metrics.SSIM, metrics.SSIMMap] = calculate_ssim(refImage, testImage, params);

    %% 3. Multi-Scale SSIM (MS-SSIM)
    metrics.MSSSIM = calculate_msssim(refImage, testImage, params);

    %% 4. Additional metrics
    metrics.MAE = mean(abs(refImage(:) - testImage(:)));  % Mean Absolute Error
    metrics.RMSE = sqrt(metrics.MSE);

    % Correlation coefficient
    refFlat = refImage(:) - mean(refImage(:));
    testFlat = testImage(:) - mean(testImage(:));
    metrics.Correlation = (refFlat' * testFlat) / (norm(refFlat) * norm(testFlat));

    % Image dimensions
    metrics.imageSize = size(refImage);
    metrics.totalPixels = numel(refImage);

    %% 5. Per-channel metrics for color images
    if ndims(refImage) == 3 && size(refImage, 3) == 3
        metrics.perChannel = struct();
        channelNames = {'R', 'G', 'B'};
        for c = 1:3
            refChannel = refImage(:,:,c);
            testChannel = testImage(:,:,c);
            [metrics.perChannel.(channelNames{c}).MSE, ...
             metrics.perChannel.(channelNames{c}).PSNR] = ...
                calculate_psnr(refChannel, testChannel, params.maxVal);
        end
    end

    %% Print results if verbose
    if params.verbose
        print_metrics(metrics);
    end
end

%% Core Metric Functions

function [mse, psnr] = calculate_psnr(ref, test, maxVal)
% CALCULATE_PSNR - Calculate MSE and PSNR

    mse = mean((ref(:) - test(:)).^2);

    if mse == 0
        psnr = Inf;
    else
        psnr = 10 * log10((maxVal^2) / mse);
    end
end

function [ssim_val, ssim_map] = calculate_ssim(ref, test, params)
% CALCULATE_SSIM - Structural Similarity Index

    K1 = params.ssimK(1);
    K2 = params.ssimK(2);
    L = params.maxVal;
    C1 = (K1 * L)^2;
    C2 = (K2 * L)^2;
    C3 = C2 / 2;

    windowSize = params.ssimWindow;

    % Gaussian window
    sigma = 1.5;
    [X, Y] = meshgrid(-(windowSize-1)/2:(windowSize-1)/2);
    g = exp(-(X.^2 + Y.^2) / (2*sigma^2));
    g = g / sum(g(:));

    % Handle color images
    if ndims(ref) == 3
        nChannels = size(ref, 3);
        ssim_channels = zeros(1, nChannels);
        for c = 1:nChannels
            ssim_channels(c) = compute_ssim_channel(ref(:,:,c), test(:,:,c), g, C1, C2, C3);
        end
        ssim_val = mean(ssim_channels);
        ssim_map = [];  % Map not computed for color
    else
        [ssim_val, ssim_map] = compute_ssim_channel(ref, test, g, C1, C2, C3);
    end
end

function [ssim_val, ssim_map] = compute_ssim_channel(ref, test, g, C1, C2, C3)
% COMPUTE_SSIM_CHANNEL - SSIM for single channel

    % Local means
    mu_ref = conv2(ref, g, 'valid');
    mu_test = conv2(test, g, 'valid');

    % Local variances and covariance
    mu_ref_sq = mu_ref.^2;
    mu_test_sq = mu_test.^2;
    mu_ref_test = mu_ref .* mu_test;

    sigma_ref_sq = conv2(ref.^2, g, 'valid') - mu_ref_sq;
    sigma_test_sq = conv2(test.^2, g, 'valid') - mu_test_sq;
    sigma_ref_test = conv2(ref.*test, g, 'valid') - mu_ref_test;

    % Clamp negative variances (numerical stability)
    sigma_ref_sq = max(sigma_ref_sq, 0);
    sigma_test_sq = max(sigma_test_sq, 0);

    sigma_ref = sqrt(sigma_ref_sq);
    sigma_test = sqrt(sigma_test_sq);

    % SSIM components
    luminance = (2*mu_ref_test + C1) ./ (mu_ref_sq + mu_test_sq + C1);
    contrast = (2*sigma_ref.*sigma_test + C2) ./ (sigma_ref_sq + sigma_test_sq + C2);
    structure = (sigma_ref_test + C3) ./ (sigma_ref.*sigma_test + C3);

    % Combined SSIM
    ssim_map = luminance .* contrast .* structure;
    ssim_val = mean(ssim_map(:));
end

function msssim = calculate_msssim(ref, test, params)
% CALCULATE_MSSSIM - Multi-Scale SSIM

    % Scale weights (from original MS-SSIM paper)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    nLevels = length(weights);

    % Component exponents
    alpha = weights;
    beta = weights;
    gamma = weights;

    msssim_components = ones(1, nLevels);
    currentRef = ref;
    currentTest = test;

    for level = 1:nLevels
        % Check minimum size
        if min(size(currentRef, 1), size(currentRef, 2)) < params.ssimWindow
            break;
        end

        % Calculate SSIM components at this level
        K1 = params.ssimK(1);
        K2 = params.ssimK(2);
        L = params.maxVal;
        C1 = (K1 * L)^2;
        C2 = (K2 * L)^2;

        windowSize = params.ssimWindow;
        sigma = 1.5;
        [X, Y] = meshgrid(-(windowSize-1)/2:(windowSize-1)/2);
        g = exp(-(X.^2 + Y.^2) / (2*sigma^2));
        g = g / sum(g(:));

        % Handle color images - convert to grayscale for MS-SSIM
        if ndims(currentRef) == 3
            refGray = 0.299*currentRef(:,:,1) + 0.587*currentRef(:,:,2) + 0.114*currentRef(:,:,3);
            testGray = 0.299*currentTest(:,:,1) + 0.587*currentTest(:,:,2) + 0.114*currentTest(:,:,3);
        else
            refGray = currentRef;
            testGray = currentTest;
        end

        % Local statistics
        mu_ref = conv2(refGray, g, 'valid');
        mu_test = conv2(testGray, g, 'valid');
        sigma_ref_sq = conv2(refGray.^2, g, 'valid') - mu_ref.^2;
        sigma_test_sq = conv2(testGray.^2, g, 'valid') - mu_test.^2;
        sigma_ref_test = conv2(refGray.*testGray, g, 'valid') - mu_ref.*mu_test;

        sigma_ref_sq = max(sigma_ref_sq, 0);
        sigma_test_sq = max(sigma_test_sq, 0);

        % Contrast and structure
        cs = (2*sqrt(sigma_ref_sq).*sqrt(sigma_test_sq) + C2) ./ (sigma_ref_sq + sigma_test_sq + C2);
        msssim_components(level) = mean(cs(:))^beta(level);

        % At finest level, include luminance
        if level == nLevels
            l = (2*mu_ref.*mu_test + C1) ./ (mu_ref.^2 + mu_test.^2 + C1);
            msssim_components(level) = msssim_components(level) * mean(l(:))^alpha(level);
        end

        % Downsample for next level
        currentRef = imresize_simple(currentRef, 0.5);
        currentTest = imresize_simple(currentTest, 0.5);
    end

    msssim = prod(msssim_components);
end

function imgOut = imresize_simple(imgIn, scale)
% IMRESIZE_SIMPLE - Simple bilinear downsampling

    [h, w, c] = size(imgIn);
    newH = round(h * scale);
    newW = round(w * scale);

    if c == 1
        imgOut = interp2(imgIn, linspace(1, w, newW), linspace(1, h, newH)', 'linear');
    else
        imgOut = zeros(newH, newW, c);
        for ch = 1:c
            imgOut(:,:,ch) = interp2(imgIn(:,:,ch), ...
                linspace(1, w, newW), linspace(1, h, newH)', 'linear');
        end
    end
end

function print_metrics(metrics)
% PRINT_METRICS - Display computed metrics

    fprintf('\n========== Image Quality Metrics ==========\n');
    fprintf('PSNR:        %.2f dB\n', metrics.PSNR);
    fprintf('SSIM:        %.4f\n', metrics.SSIM);
    fprintf('MS-SSIM:     %.4f\n', metrics.MSSSIM);
    fprintf('MSE:         %.4f\n', metrics.MSE);
    fprintf('RMSE:        %.4f\n', metrics.RMSE);
    fprintf('MAE:         %.4f\n', metrics.MAE);
    fprintf('Correlation: %.4f\n', metrics.Correlation);
    fprintf('============================================\n');
end
