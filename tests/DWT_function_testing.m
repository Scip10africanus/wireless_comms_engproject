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
policy.mcs_base = uint8(4);
policy.mcs_enh  = uint8(4);
policy.tx_mode = 'split';
policy.mux_gap_us = 0;

rx_Qb = 1; % using fixed one first
rx_Qe = 1;

frameId = uint8(1);

fprintf('Policy: Qb=%d, Qe=%d, MCS_B=%d, MCS_E=%d\n', ...
    rx_Qb, rx_Qe, policy.mcs_base, policy.mcs_enh);

[txWaves, txInfo] = two_layer_tx_wavelet(img, policy, frameId);
fprintf('TX done.\n');

rxWaves = txWaves;

fprintf('Running RX loop...\n');

baseBlocks = cell(txInfo.numBaseBlocks, 1);
if txInfo.numBaseBlocks > 0
    for i = 1:txInfo.numBaseBlocks
        % Demodulate
        rxBaseBytes = demodem(rxWaves.base{i}, txInfo.cfgBase);
        
        % Unpack
        rxBaseInt16 = typecast(rxBaseBytes, 'int16');
        
        % Dequantize
        [blk_hat, ~] = dequant(rxBaseInt16, [], rx_Qb, rx_Qe);
        
        % Reshape to block size
        baseBlocks{i} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
    end
    
    base_hat_coeffs = untile_coeffs(baseBlocks, txInfo.basePadInfo, txInfo.blk_size);
else
    base_hat_coeffs = zeros(txInfo.basePadInfo);
end

finalLL_hat = base_hat_coeffs;

res_hat = struct();
res_hat.finalLL = finalLL_hat;
res_hat.subbands = struct('LH', {}, 'HL', {}, 'HH', {}, 'inputSize', {});

% Pre-allocate subbands
for k = 1:txInfo.level
    res_hat.subbands(k).LH = [];
    res_hat.subbands(k).HL = [];
    res_hat.subbands(k).HH = [];
    res_hat.subbands(k).inputSize = []; 
end

enhIdx = 1;
for i = 1:numel(txInfo.enhSubbandMeta)
    meta = txInfo.enhSubbandMeta(i);
    sbBlocks = cell(meta.numBlocks, 1);
    
    for b = 1:meta.numBlocks
        waveEnh = rxWaves.enh{enhIdx};
        % Assume same config for all enh packets
        cfgEnh  = txInfo.cfgEnh; 
        
        % Demodulate
        rxEnhBytes = demodem(waveEnh, cfgEnh);
        
        % Unpack
        rxEnhInt8 = typecast(rxEnhBytes, 'int8');
        
        % Dequantize
        [~, blk_hat] = dequant([], rxEnhInt8, rx_Qb, rx_Qe);
        
        % Reshape
        sbBlocks{b} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
        
        enhIdx = enhIdx + 1;
    end
    
    % Reconstruct Subband
    sbMat = untile_coeffs(sbBlocks, meta.origSize, txInfo.blk_size);
    
    % Assign
    res_hat.subbands(meta.level).(meta.type) = sbMat;
    
    % Infer inputSize
    if isempty(res_hat.subbands(meta.level).inputSize)
        res_hat.subbands(meta.level).inputSize = meta.origSize * 2;
    end
end

fprintf('Running IDWT...\n');
reconImg = IDWT(res_hat, txInfo.level);
    
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

