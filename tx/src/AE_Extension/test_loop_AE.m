%% Test Script for Full AI-Driven Pipeline (TX -> Channel -> RX)
% test_loop_AE.m
targetPath = 'C:\Users\renka\Desktop\wireless_comms_engproject-main (1)\wireless_comms_engproject-main\tx\test_functions';

if exist(targetPath, 'dir')
    addpath(targetPath);
    fprintf('✅ succesfully: %s\n', targetPath);
else
    warning('❌ error: %s', targetPath);
end
clear; clc; close all;

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

% 填入你图片文件夹中的任意一张具体图片名称，比如 cameraman.tif 或 lena.png
img_path = 'C:\Users\renka\Desktop\wireless_comms_engproject-main (1)\wireless_comms_engproject-main\tx\images\Testing examples.jpg';

try
    imgOriginal = imread(img_path);
    
    % 如果是彩色图片(3通道)，AI 模型只吃灰度图(1通道)，必须转换一下：
    if size(imgOriginal, 3) == 3
        imgOriginal = rgb2gray(imgOriginal);
    end
    
    % 统一缩放到 64x64
    imgOriginal = imresize(imgOriginal, [64 64]);
catch ME
    error('❌ 找不到图片，请检查 img_path 路径和文件名是否拼写正确！\n错误详情: %s', ME.message);
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