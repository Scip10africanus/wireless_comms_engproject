clear; clc; close all;
currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

imagePath = fullfile(rootDir, 'images', 'Testing examples.jpg');
rawImg = imread(imagePath);
img = rgb2gray(rawImg);
img = double(img); 
fprintf('Loaded image: %s (%dx%d)\n', 'Testing examples.jpg', size(img,1), size(img,2));

policy = struct();
policy.Qb_idx = uint8(3);   % -> 0.01
policy.Qe_idx = uint8(3);   % -> 0.01
policy.mcs_base = uint8(0);
policy.mcs_enh  = uint8(2);
policy.tx_mode = 'split';
policy.mux_gap_us = 0;

rx_Qb = 1; % using fixed one first
rx_Qe = 1;

param = struct();
param.SNR_dB = 5;
param.channelType = 'RICIAN';
param.ricianK = 3;

frameId = 0;

fprintf('Policy: Qb=%d, Qe=%d, MCS_B=%d, MCS_E=%d\n', ...
    rx_Qb, rx_Qe, policy.mcs_base, policy.mcs_enh);

[txWaves, txInfo] = two_layer_tx_wavelet(img, policy, frameId);
fprintf('TX done.\n');

baseBlocks = cell(txInfo.numBaseBlocks, 1);
rxWaves = struct('base', [], 'enh', {{}});
if txInfo.numBaseBlocks > 0
    for i = 1:txInfo.numBaseBlocks
        [rxSignal, channelInfo] = channel_model_v1(txWaves.base{i}, param);
        rxWaves.base{i} = rxSignal;
    end
end
enhIdx = 0;
for i = 1:txInfo.numEnhBlocks
    [rxSignal, channelInfo] = channel_model_v1(txWaves.enh{i}, param);
    rxWaves.enh{i} = rxSignal;
end
[res_hat,rxinfo] = two_layer_rx_testing_dwt(rxWaves, txInfo, rx_Qb, rx_Qe);
fprintf('RX done.\n');
fprintf('Base fail count: %d\n', numel(rxinfo.baseFailReasons));
fprintf('Enh fail count : %d\n', numel(rxinfo.enhFailReasons));

if ~isempty(rxinfo.baseFailReasons)
    disp(unique(string(rxinfo.baseFailReasons)));
end
if ~isempty(rxinfo.enhFailReasons)
    disp(unique(string(rxinfo.enhFailReasons)));
end

fprintf('Running IDWT...\n');
reconImg = IDWT(res_hat, txInfo.level);

if ~isempty(rxinfo.baseFailReasons)
    disp('Base fail reasons:');
    disp(unique(string(rxinfo.baseFailReasons)));
end

if ~isempty(rxinfo.enhFailReasons)
    disp('Enh fail reasons:');
    disp(unique(string(rxinfo.enhFailReasons)));
end

err = img - reconImg;
mse = mean(err(:).^2);
psnr_val = psnr(reconImg, img, 255);
ssim_val = ssim(reconImg, img);
    
fprintf('--------------------------------------------------\n');
fprintf('Results:\n');
fprintf('  MSE:  %.4e\n', mse);
fprintf('  PSNR: %.2f dB\n', psnr_val);
fprintf('  SSIM: %.4f\n', ssim_val);
fprintf('--------------------------------------------------\n');
    
figure('Name', 'Wavelet Tx/Rx Loopback Test');
subplot(1,3,1); imshow(uint8(img)); title('Original');
subplot(1,3,2); imshow(uint8(reconImg)); title(sprintf('Reconstructed (PSNR=%.2f dB)', psnr_val));
subplot(1,3,3); imagesc(abs(err)); colorbar; title('Error Residual');