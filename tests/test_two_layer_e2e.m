%% End-to-End Test: Two-Layer Residual Coding System
%
% Complete demonstration of the system:
%   TX: Image → Wavelet(base) + DCT(residual) → OFDM
%   Channel: AWGN / Rayleigh / Multipath
%   RX: OFDM → Decode layers → Fuse → Quality assessment
%
% This script tests:
%   1. Normal operation (both layers OK)
%   2. Graceful degradation (enhancement lost)
%   3. SNR performance sweep
%   4. Different channel conditions
%
% Author: RX Processing Team
% Date: 2026

clear all; close all; clc;

% Add paths
addpath('..');
addpath('../tx');
addpath('../channel');

fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║     Two-Layer Residual Coding System - End-to-End Test          ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

%% ==================== SYSTEM PARAMETERS ====================

params = struct();

% OFDM
params.nFFT = 64;
params.nCP = 16;

% Image
params.imageHeight = 64;
params.imageWidth = 64;
params.nChannels = 1;  % Grayscale

% Base layer (Wavelet)
params.waveletType = 'haar';
params.waveletLevels = 2;
params.baseMod = 4;  % QPSK
params.baseBitsPerCoeff = 8;

% Enhancement layer (DCT)
params.dctBlockSize = 8;
params.coeffOrdering = 'zigzag';
params.enhancementMod = 64;  % 64-QAM
params.enhancementBitsPerCoeff = 8;
params.maxDctCoeffs = 64;
params.residualScale = 1.0;

% Subcarrier allocation
guardBand = 4;
dcNull = params.nFFT/2 + 1;
allData = setdiff((guardBand+1):(params.nFFT-guardBand), dcNull);
params.pilotIndices = allData(1:8:end);
dataIndices = setdiff(allData, params.pilotIndices);
nBase = round(length(dataIndices) * 0.25);
params.baseSubcarriers = dataIndices(1:nBase);
params.enhancementSubcarriers = dataIndices(nBase+1:end);
params.pilotValues = ones(length(params.pilotIndices), 1);

% FEC (disabled for now)
params.useFEC = false;
params.baseFEC = 'none';
params.enhancementFEC = 'none';
params.baseFECRate = 1/2;
params.enhancementFECRate = 3/4;

% Quality targets
params.targetSSIM = 0.9;
params.targetPSNR = 30;

% Display
params.verbose = true;

%% ==================== GENERATE TEST IMAGE ====================

fprintf('[Setup] Generating test image...\n');

% Create a test image with various features
testImage = create_test_image(params.imageHeight, params.imageWidth);
params.refImage = testImage;  % Store for quality calculation

figure('Name', 'Test Image', 'Position', [100, 100, 300, 300]);
imshow(testImage, []);
title('Original Test Image');

fprintf('  Image size: %dx%d\n', params.imageHeight, params.imageWidth);

%% ==================== TEST 1: NORMAL OPERATION (AWGN) ====================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('TEST 1: Normal Operation (AWGN, SNR = 25 dB)\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

channelParams = struct();
channelParams.channelType = 'AWGN';
channelParams.SNR_dB = 25;

[rxImage1, ssim1, psnr1] = run_e2e_test(testImage, params, channelParams);

fprintf('\nResults:\n');
fprintf('  SSIM: %.4f\n', ssim1);
fprintf('  PSNR: %.2f dB\n', psnr1);

%% ==================== TEST 2: LOW SNR (GRACEFUL DEGRADATION) ====================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('TEST 2: Low SNR - Testing Graceful Degradation (SNR = 10 dB)\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

channelParams.SNR_dB = 10;

[rxImage2, ssim2, psnr2] = run_e2e_test(testImage, params, channelParams);

fprintf('\nResults:\n');
fprintf('  SSIM: %.4f\n', ssim2);
fprintf('  PSNR: %.2f dB\n', psnr2);

%% ==================== TEST 3: SNR SWEEP ====================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('TEST 3: SNR Performance Sweep\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

params.verbose = false;  % Quiet mode for sweep

SNR_range = 5:5:35;
nTests = length(SNR_range);

ssim_results = zeros(nTests, 1);
psnr_results = zeros(nTests, 1);
base_ber = zeros(nTests, 1);
enh_ber = zeros(nTests, 1);

fprintf('Running SNR sweep...\n');
for i = 1:nTests
    channelParams.SNR_dB = SNR_range(i);
    [~, ssim_results(i), psnr_results(i), results] = run_e2e_test(testImage, params, channelParams);

    % Get BER if available
    if isfield(results, 'baseBER')
        base_ber(i) = results.baseBER;
        enh_ber(i) = results.enhBER;
    end

    fprintf('  SNR %2d dB: SSIM=%.4f, PSNR=%.2f dB\n', ...
        SNR_range(i), ssim_results(i), psnr_results(i));
end

params.verbose = true;

% Plot results
figure('Name', 'SNR Performance', 'Position', [100, 400, 1000, 400]);

subplot(1, 2, 1);
plot(SNR_range, ssim_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(params.targetSSIM, 'r--', 'Target SSIM', 'LineWidth', 1.5);
grid on;
xlabel('SNR (dB)');
ylabel('SSIM');
title('SSIM vs SNR');
ylim([0, 1]);
legend('Measured', 'Target', 'Location', 'southeast');

subplot(1, 2, 2);
plot(SNR_range, psnr_results, 'g-s', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(params.targetPSNR, 'r--', 'Target PSNR', 'LineWidth', 1.5);
grid on;
xlabel('SNR (dB)');
ylabel('PSNR (dB)');
title('PSNR vs SNR');
legend('Measured', 'Target', 'Location', 'southeast');

%% ==================== TEST 4: DIFFERENT CHANNEL TYPES ====================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('TEST 4: Different Channel Types (SNR = 20 dB)\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

channelTypes = {'AWGN', 'Rayleigh', 'Multipath', 'Rician'};
nChannels = length(channelTypes);

channel_ssim = zeros(nChannels, 1);
channel_psnr = zeros(nChannels, 1);
channel_images = cell(nChannels, 1);

for i = 1:nChannels
    fprintf('\nChannel: %s\n', channelTypes{i});

    channelParams = struct();
    channelParams.channelType = channelTypes{i};
    channelParams.SNR_dB = 20;
    channelParams.dopplerHz = 10;
    channelParams.sampleRate = 1e6;
    channelParams.nFFT = params.nFFT;
    channelParams.nCP = params.nCP;
    channelParams.ricianK = 6;

    [channel_images{i}, channel_ssim(i), channel_psnr(i)] = ...
        run_e2e_test(testImage, params, channelParams);

    fprintf('  SSIM: %.4f, PSNR: %.2f dB\n', channel_ssim(i), channel_psnr(i));
end

% Display channel comparison
figure('Name', 'Channel Comparison', 'Position', [200, 200, 1200, 400]);

subplot(1, 5, 1);
imshow(testImage, []);
title('Original');

for i = 1:nChannels
    subplot(1, 5, i+1);
    imshow(channel_images{i}, []);
    title(sprintf('%s\nSSIM=%.3f', channelTypes{i}, channel_ssim(i)));
end

sgtitle('Channel Type Comparison (SNR = 20 dB)');

%% ==================== TEST 5: GRACEFUL DEGRADATION DEMO ====================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('TEST 5: Graceful Degradation Demonstration\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

% Show what happens at different SNR levels
demo_snrs = [30, 20, 15, 10, 5];
nDemo = length(demo_snrs);

figure('Name', 'Graceful Degradation', 'Position', [100, 100, 1400, 350]);

subplot(1, nDemo+1, 1);
imshow(testImage, []);
title('Original');

for i = 1:nDemo
    channelParams.channelType = 'AWGN';
    channelParams.SNR_dB = demo_snrs(i);

    [demoImage, demoSsim, demoPsnr] = run_e2e_test(testImage, params, channelParams);

    subplot(1, nDemo+1, i+1);
    imshow(demoImage, []);
    title(sprintf('SNR=%ddB\nSSIM=%.3f', demo_snrs(i), demoSsim));
end

sgtitle('System Performance at Different SNR Levels');

%% ==================== SUMMARY ====================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TEST SUMMARY                                  ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Two-Layer Residual Coding System                                 ║\n');
fprintf('║   Base: %s wavelet L%d → %d-QAM                                ║\n', ...
    params.waveletType, params.waveletLevels, params.baseMod);
fprintf('║   Enhancement: %dx%d DCT → %d-QAM                               ║\n', ...
    params.dctBlockSize, params.dctBlockSize, params.enhancementMod);
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║ AWGN Performance:                                                ║\n');
fprintf('║   SNR=25dB: SSIM=%.4f, PSNR=%.2fdB                              ║\n', ssim1, psnr1);
fprintf('║   SNR=10dB: SSIM=%.4f, PSNR=%.2fdB (graceful degradation)       ║\n', ssim2, psnr2);
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║ Channel Robustness (SNR=20dB):                                   ║\n');
for i = 1:nChannels
    fprintf('║   %-10s: SSIM=%.4f                                          ║\n', ...
        channelTypes{i}, channel_ssim(i));
end
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

fprintf('\nAll tests complete!\n');

%% ==================== HELPER FUNCTIONS ====================

function testImage = create_test_image(height, width)
% CREATE_TEST_IMAGE - Generate a test image with various features

    % Base gradient
    [X, Y] = meshgrid(1:width, 1:height);
    testImage = 50 + 0.5*X + 0.5*Y;

    % Add circle
    cx = width/2; cy = height/2;
    r = min(width, height) / 4;
    circle = sqrt((X-cx).^2 + (Y-cy).^2) < r;
    testImage(circle) = 200;

    % Add rectangle
    rect = (X > width*0.1) & (X < width*0.3) & (Y > height*0.6) & (Y < height*0.9);
    testImage(rect) = 150;

    % Add some edge detail
    testImage(height/2-2:height/2+2, :) = testImage(height/2-2:height/2+2, :) + 30;
    testImage(:, width/2-2:width/2+2) = testImage(:, width/2-2:width/2+2) + 30;

    % Add noise for texture
    testImage = testImage + 5*randn(height, width);

    % Clip and convert
    testImage = uint8(max(0, min(255, testImage)));
end

function [rxImage, ssim_val, psnr_val, results] = run_e2e_test(testImage, params, channelParams)
% RUN_E2E_TEST - Run complete TX -> Channel -> RX test

    % Transmit
    [txSignal, txInfo] = two_layer_tx_pipeline(testImage, params);

    % Channel
    [rxSignal, channelInfo] = channel_model(txSignal, channelParams);

    % Receive
    [rxImage, feedback, results] = two_layer_rx_pipeline(rxSignal, params);

    % Calculate quality
    [ssim_val, psnr_val] = calculate_quality(testImage, rxImage);

    % BER calculation (if original bits available)
    if isfield(txInfo, 'baseBits')
        results.baseBER = calculate_ber(txInfo.baseBits, results);
        results.enhBER = calculate_ber(txInfo.enhancementBits, results);
    end
end

function [ssim_val, psnr_val] = calculate_quality(ref, test)
% CALCULATE_QUALITY - Compute SSIM and PSNR

    ref = double(ref);
    test = double(test);

    % PSNR
    mse = mean((ref(:) - test(:)).^2);
    if mse == 0
        psnr_val = 100;
    else
        psnr_val = 10 * log10(255^2 / mse);
    end

    % SSIM (simplified)
    K1 = 0.01; K2 = 0.03; L = 255;
    C1 = (K1*L)^2; C2 = (K2*L)^2;

    mu_ref = mean(ref(:));
    mu_test = mean(test(:));
    sigma_ref = std(ref(:));
    sigma_test = std(test(:));
    sigma_rt = mean((ref(:)-mu_ref).*(test(:)-mu_test));

    ssim_val = ((2*mu_ref*mu_test + C1)*(2*sigma_rt + C2)) / ...
               ((mu_ref^2 + mu_test^2 + C1)*(sigma_ref^2 + sigma_test^2 + C2));
end

function ber = calculate_ber(txBits, results)
% CALCULATE_BER - Placeholder BER calculation
    ber = 0;  % TODO: Implement actual BER calculation
end
