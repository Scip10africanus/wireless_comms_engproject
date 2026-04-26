%% Test Script for Full AI-Driven Pipeline (TX -> Channel -> RX)
% test_loop_AE.m
clear; clc; close all;

% Resolve paths relative to this script location.
% Supports running from either tx/src or tx/src/AE_Extension.
scriptDir = fileparts(mfilename('fullpath'));
[~, scriptFolderName] = fileparts(scriptDir);
if strcmpi(scriptFolderName, 'AE_Extension')
    txRootDir = fileparts(fileparts(scriptDir)); % tx/src/AE_Extension -> tx
else
    txRootDir = fileparts(scriptDir);            % tx/src -> tx
end

targetPath = fullfile(txRootDir, 'test_functions');
img_path = fullfile(txRootDir, 'images', 'Testing examples.jpg');

if exist(targetPath, 'dir')
    addpath(targetPath);
    fprintf('✅ Successfully added path: %s\n', targetPath);
else
    warning('❌ Path not found: %s', targetPath);
end

fprintf('==================================================\n');
fprintf('  Testing Full AI Pipeline (TX -> AWGN -> RX)\n');
fprintf('==================================================\n\n');

%% 1. Python Environment Check
pe = pyenv;
if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

%% 2. Prepare Original Image
fprintf('[1/4] Preparing Image...\n');

try
    imgOriginal = imread(img_path);
    
    % If the image is RGB (3 channels), convert to grayscale for the AI model.
    if size(imgOriginal, 3) == 3
        imgOriginal = rgb2gray(imgOriginal);
    end
    
    % Resize to 64x64.
    imgOriginal = imresize(imgOriginal, [64 64]);
catch ME
    error('❌ Image not found. Please check img_path and filename spelling.\nError details: %s', ME.message);
end
%% 3. Setup Policy & Execute TX
fprintf('[2/4] Executing AI-TX...\n');
policy = struct();
policy.mcs_base = 2;   % QPSK
policy.mcs_enh  = 6;   % 64-QAM
policy.mux_gap_us = 4;

[txWave, txInfo] = two_layer_tx_AE(imgOriginal, policy, 1);

%% 4. Simulate Wireless Channel (AWGN)
% We test two scenarios: Good SNR and Bad SNR
snr_db = 15; % Set to 15dB (Enhancement layer might fail, triggering Graceful Degradation!)
fprintf('[3/4] Passing through AWGN Channel (SNR = %d dB)...\n', snr_db);

% Add noise
rxWave = awgn(txWave, snr_db, 'measured');

%% 5. Execute RX Pipeline
fprintf('[4/4] Executing AI-RX & Decoding...\n');
try
    tic;
    [rxImage, rxInfo] = two_layer_rx_AE(rxWave, txInfo, policy);
    elapsedTime = toc;
    
    fprintf('\n✅ SUCCESS! Decoding completed in %.3f seconds.\n', elapsedTime);
    if isfield(rxInfo, 'enhStatus')
        fprintf('   - Enhancement Layer Status: %s\n', rxInfo.enhStatus);
    end
    
    % Calculate simple MSE/PSNR
    mse = mean((double(imgOriginal(:)) - rxImage(:)).^2);
    psnr_val = 10*log10(255^2 / mse);
    fprintf('   - Reconstructed PSNR: %.2f dB\n', psnr_val);
    
    % 6. Plot the Result
    figure('Name', 'AI Communication Loop Result', 'Position', [200, 200, 800, 400]);
    
    subplot(1,2,1);
    imshow(imgOriginal);
    title('Original TX Image');
    
    subplot(1,2,2);
    imshow(uint8(rxImage));
    title(sprintf('Reconstructed RX (SNR=%ddB, PSNR=%.1fdB)', snr_db, psnr_val));
    
catch ME
    fprintf('\n❌ RX FAILED! An error occurred:\n');
    disp(ME.message);
end
