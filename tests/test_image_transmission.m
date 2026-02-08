%% Complete Image Transmission Test Script
% Demonstrates the full TX-RX pipeline for image transmission over OFDM
%
% This script:
%   1. Loads or generates a test image
%   2. Simulates a simple transmitter (packing, modulation, OFDM)
%   3. Passes through a simulated channel (AWGN, multipath)
%   4. Processes with your RX pipeline
%   5. Evaluates and visualizes results
%
% Author: RX Processing Team
% Date: 2026

clear all; close all; clc;

% Add parent directory to path for module access
addpath(fileparts(pwd));

fprintf('========================================\n');
fprintf('  Image Transmission Test Suite\n');
fprintf('========================================\n\n');

%% System Parameters
params = struct();

% OFDM parameters
params.nFFT = 64;           % FFT size
params.nCP = 16;            % Cyclic prefix length
params.modOrder = 16;       % 16-QAM
params.pilotSpacing = 8;    % Pilot every 8 subcarriers

% Channel parameters
params.SNR_dB = 20;         % Signal-to-noise ratio
params.channelType = 'AWGN'; % 'AWGN', 'Rayleigh', or 'Multipath'
params.maxDopplerHz = 10;   % For Rayleigh/Multipath

% Image parameters
params.imageHeight = 64;
params.imageWidth = 64;
params.nChannels = 1;       % Grayscale
params.bitDepth = 8;

fprintf('Configuration:\n');
fprintf('  FFT Size: %d, CP Length: %d\n', params.nFFT, params.nCP);
fprintf('  Modulation: %d-QAM\n', params.modOrder);
fprintf('  SNR: %d dB\n', params.SNR_dB);
fprintf('  Image: %dx%dx%d\n', params.imageHeight, params.imageWidth, params.nChannels);

%% Generate Test Image
fprintf('\n[1] Generating test image...\n');

% Create a synthetic test image (gradient + pattern)
[X, Y] = meshgrid(1:params.imageWidth, 1:params.imageHeight);
testImage = uint8(mod(X + Y, 256));  % Diagonal gradient

% Add a circle pattern
centerX = params.imageWidth / 2;
centerY = params.imageHeight / 2;
radius = min(params.imageWidth, params.imageHeight) / 4;
circle = sqrt((X - centerX).^2 + (Y - centerY).^2) < radius;
testImage(circle) = 200;

% Add some texture
testImage = testImage + uint8(10 * randn(params.imageHeight, params.imageWidth));

fprintf('  Generated %dx%d test image\n', params.imageHeight, params.imageWidth);

%% ========== TRANSMITTER (Simplified) ==========
fprintf('\n[2] Transmitter processing...\n');

% 2.1 Image to bitstream with header
txBits = image_to_bitstream(testImage, params);
fprintf('  Packed %d bits (%.1f KB)\n', length(txBits), length(txBits)/8/1024);

% 2.2 Calculate required OFDM symbols
bitsPerSymbol = log2(params.modOrder);
bitsPerOFDMSymbol = params.nFFT * bitsPerSymbol;
nOFDMSymbols = ceil(length(txBits) / bitsPerOFDMSymbol);

% Pad bits to fill complete symbols
txBitsPadded = [txBits; zeros(nOFDMSymbols * bitsPerOFDMSymbol - length(txBits), 1)];

fprintf('  Requires %d OFDM symbols\n', nOFDMSymbols);

% 2.3 QAM Modulation
txSymbols = qammod(txBitsPadded, params.modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
txSymbols = reshape(txSymbols, params.nFFT, nOFDMSymbols);

% 2.4 Add pilots (for channel estimation)
pilotIndices = 1:params.pilotSpacing:params.nFFT;
pilotValues = ones(length(pilotIndices), 1);  % Known pilot symbols
for sym = 1:nOFDMSymbols
    txSymbols(pilotIndices, sym) = pilotValues;
end

% 2.5 IFFT (time domain)
txTimeDomain = ifft(txSymbols, params.nFFT);

% 2.6 Add cyclic prefix
txWithCP = [txTimeDomain(end-params.nCP+1:end, :); txTimeDomain];

% 2.7 Serialize
txSignal = txWithCP(:);
fprintf('  Transmitted signal: %d samples\n', length(txSignal));

%% ========== CHANNEL ==========
fprintf('\n[3] Channel simulation (%s)...\n', params.channelType);

switch params.channelType
    case 'AWGN'
        % Simple AWGN channel
        rxSignal = awgn(txSignal, params.SNR_dB, 'measured');

    case 'Rayleigh'
        % Flat Rayleigh fading + AWGN
        h = (randn(1) + 1j*randn(1)) / sqrt(2);
        rxSignal = h * txSignal;
        rxSignal = awgn(rxSignal, params.SNR_dB, 'measured');

    case 'Multipath'
        % Multipath channel (2-tap)
        delays = [0, 3];  % Sample delays
        gains = [1, 0.5]; % Relative gains
        h = zeros(max(delays)+1, 1);
        h(delays+1) = gains .* exp(1j*2*pi*rand(1,2));
        rxSignal = conv(txSignal, h);
        rxSignal = rxSignal(1:length(txSignal));  % Truncate
        rxSignal = awgn(rxSignal, params.SNR_dB, 'measured');

    otherwise
        rxSignal = awgn(txSignal, params.SNR_dB, 'measured');
end

fprintf('  Channel applied successfully\n');

%% ========== RECEIVER (Your Pipeline) ==========
fprintf('\n[4] Receiver processing...\n');

% Configure RX pipeline
rxParams = struct();
rxParams.nFFT = params.nFFT;
rxParams.nCP = params.nCP;
rxParams.modOrder = params.modOrder;
rxParams.equalizer = 'ZF';
rxParams.noiseVar = 10^(-params.SNR_dB/10);

% Pilot info for channel estimation
rxParams.pilotIndices = pilotIndices;
rxParams.pilotValues = pilotValues;

% Image parameters
rxParams.hasHeader = true;  % Header included in bitstream
rxParams.bitDepth = params.bitDepth;

% Reference for quality assessment
rxParams.refImage = testImage;
rxParams.txBits = txBits;

% Enable verbose output and plots
rxParams.verbose = true;
rxParams.plotResults = true;
rxParams.generateFeedback = true;

% Run RX pipeline
[rxImage, results] = rx_pipeline(rxSignal, rxParams);

%% ========== RESULTS SUMMARY ==========
fprintf('\n========================================\n');
fprintf('  RESULTS SUMMARY\n');
fprintf('========================================\n');

if results.success
    fprintf('Pipeline Status: SUCCESS\n\n');

    % BER
    if isfield(results, 'BER')
        fprintf('Bit Error Rate: %.2e (%d errors / %d bits)\n', ...
            results.BER, results.numBitErrors, length(txBits));
    end

    % Image quality
    if ~isempty(results.qualityMetrics)
        fprintf('\nImage Quality:\n');
        fprintf('  PSNR:    %.2f dB\n', results.qualityMetrics.PSNR);
        fprintf('  SSIM:    %.4f\n', results.qualityMetrics.SSIM);
        fprintf('  MS-SSIM: %.4f\n', results.qualityMetrics.MSSSIM);
    end

    % Feedback
    if isfield(results, 'feedback') && ~isempty(results.feedback)
        fprintf('\nAdaptive Feedback:\n');
        fprintf('  CQI: %d/15 (%s)\n', results.feedback.CQI, ...
            results.feedback.diagnostics.channelState);
        fprintf('  Quality Indicator: %.2f (%s)\n', results.feedback.QI, ...
            results.feedback.diagnostics.qualityState);
        fprintf('  Recommendation: %s\n', results.feedback.diagnostics.recommendation);
    end
else
    fprintf('Pipeline Status: FAILED\n');
    fprintf('Errors:\n');
    for i = 1:length(results.errors)
        fprintf('  - %s\n', results.errors{i});
    end
end

% Warnings
if ~isempty(results.warnings)
    fprintf('\nWarnings:\n');
    for i = 1:length(results.warnings)
        fprintf('  - %s\n', results.warnings{i});
    end
end

fprintf('\nProcessing time: %.3f seconds\n', results.processingTime);
fprintf('========================================\n');

%% ========== ADDITIONAL VISUALIZATIONS ==========
figure('Position', [100, 100, 1000, 400], 'Name', 'Image Comparison');

subplot(1, 3, 1);
imshow(testImage, []);
title('Original Image');

subplot(1, 3, 2);
imshow(rxImage, []);
title(sprintf('Received Image (PSNR: %.1f dB)', results.qualityMetrics.PSNR));

subplot(1, 3, 3);
diffImage = abs(double(testImage) - double(rxImage));
imshow(diffImage, []);
colorbar;
title('Absolute Difference');

%% ========== SNR SWEEP TEST ==========
fprintf('\n[5] Running SNR sweep test...\n');

SNR_range = 0:5:30;
BER_results = zeros(size(SNR_range));
PSNR_results = zeros(size(SNR_range));
SSIM_results = zeros(size(SNR_range));

for idx = 1:length(SNR_range)
    % Apply channel at this SNR
    testRxSignal = awgn(txSignal, SNR_range(idx), 'measured');

    % Configure RX (silent mode)
    testRxParams = rxParams;
    testRxParams.verbose = false;
    testRxParams.plotResults = false;
    testRxParams.noiseVar = 10^(-SNR_range(idx)/10);

    % Run pipeline
    [testRxImage, testResults] = rx_pipeline(testRxSignal, testRxParams);

    % Store results
    if isfield(testResults, 'BER')
        BER_results(idx) = testResults.BER;
    end
    if ~isempty(testResults.qualityMetrics)
        PSNR_results(idx) = testResults.qualityMetrics.PSNR;
        SSIM_results(idx) = testResults.qualityMetrics.SSIM;
    end

    fprintf('  SNR %2d dB: BER=%.2e, PSNR=%.1f dB, SSIM=%.3f\n', ...
        SNR_range(idx), BER_results(idx), PSNR_results(idx), SSIM_results(idx));
end

% Plot SNR sweep results
figure('Position', [200, 200, 1200, 400], 'Name', 'SNR Performance');

subplot(1, 3, 1);
semilogy(SNR_range, BER_results + 1e-10, 'b-o', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR');
ylim([1e-6, 1]);

subplot(1, 3, 2);
plot(SNR_range, PSNR_results, 'r-s', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('PSNR (dB)');
title('Image PSNR vs SNR');

subplot(1, 3, 3);
plot(SNR_range, SSIM_results, 'g-^', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('SSIM');
title('SSIM vs SNR');
ylim([0, 1]);

fprintf('\nTest complete!\n');

%% ========== HELPER FUNCTIONS ==========

function bitstream = image_to_bitstream(image, params)
% IMAGE_TO_BITSTREAM - Convert image to bitstream with header
%
% Header format (64 bits):
%   Bits 1-16:  Height
%   Bits 17-32: Width
%   Bits 33-36: Channels
%   Bits 37-40: Bit depth
%   Bits 41-48: Compression (0=none)
%   Bits 49-64: Checksum

    [h, w, c] = size(image);

    % Create header
    header = zeros(64, 1);
    header(1:16) = uint_to_bits(h, 16);
    header(17:32) = uint_to_bits(w, 16);
    header(33:36) = uint_to_bits(c, 4);
    header(37:40) = uint_to_bits(params.bitDepth, 4);
    header(41:48) = uint_to_bits(0, 8);  % No compression

    % Checksum (XOR of fields)
    checksum = bitxor(h, w);
    checksum = bitxor(checksum, c);
    checksum = bitxor(checksum, params.bitDepth);
    header(49:64) = uint_to_bits(mod(checksum, 65536), 16);

    % Convert image to bits
    imageBytes = image(:);
    imageBits = zeros(length(imageBytes) * params.bitDepth, 1);
    for i = 1:length(imageBytes)
        startIdx = (i-1) * params.bitDepth + 1;
        imageBits(startIdx:startIdx+params.bitDepth-1) = uint_to_bits(imageBytes(i), params.bitDepth);
    end

    bitstream = [header; imageBits];
end

function bits = uint_to_bits(val, nBits)
% UINT_TO_BITS - Convert unsigned integer to bit vector (MSB first)
    bits = zeros(nBits, 1);
    for i = nBits:-1:1
        bits(i) = mod(val, 2);
        val = floor(val / 2);
    end
end
