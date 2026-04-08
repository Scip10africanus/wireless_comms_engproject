clear; clc; close all;
currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

imagePath = fullfile(rootDir, 'images', 'Testing examples.jpg');
rawImg = imread(imagePath);
img = double(rgb2gray(rawImg));
fprintf('Loaded image: %s (%dx%d)\n', 'Testing examples.jpg', size(img,1), size(img,2));

policy = struct();
policy.K = 20; % Kb
policy.Ke = 64; % Ke
policy.Qb_idx = 2; % Q = 1
policy.Qe_idx = 2; % Q = 1
policy.mcs_base = 0;
policy.mcs_enh = 2;

param = struct();
param.channelType = 'RICIAN';
param.ricianK = 0;
param.SNR_dB = 7; % not actually used since we mock channel drops

frameId = 1;

fprintf('====================================================\n');
fprintf('        DWT-fine-oracle Validation Branch           \n');
fprintf('====================================================\n');
fprintf('Testing whether the main value comes from finer     \n');
fprintf('layering itself, or from adding a subband-DCT stage.\n');
fprintf('Note: HL/LH selected indices are shared via metadata\n');
fprintf('      and are NOT counted in the payload bits.\n');
fprintf('====================================================\n\n');

fprintf('Policy: Kb=%d, Ke=%d, Qb_idx=%d, Qe_idx=%d\n', policy.K, policy.Ke, policy.Qb_idx, policy.Qe_idx);

fprintf('\n--- Running DWT-fine-oracle TX ---\n');
[txWaves_dwt, txInfo_dwt] = two_layer_tx_dwt_fine_oracle(img, policy, frameId);
basePayloadBits_dwt = sum(txInfo_dwt.basePktLens) * 8;
enhPayloadBits_dwt = sum(txInfo_dwt.enhPktLens) * 8;

fprintf('\n--- Running DWT-fine-oracle RX (Base + Enh) ---\n');
rxWaves_dwt_full = txWaves_dwt;
[reconImgFull_dwt, ~] = two_layer_rx_dwt_fine_oracle(rxWaves_dwt_full, txInfo_dwt);
psnr_full_dwt = psnr(min(max(reconImgFull_dwt, 0), 255), img, 255);
ssim_full_dwt = ssim(min(max(reconImgFull_dwt, 0), 255), img);

fprintf('\n--- Running DWT-fine-oracle RX (Base Only) ---\n');
rxWaves_dwt_base = txWaves_dwt;
rxWaves_dwt_base.enh = {};
[reconImgBase_dwt, ~] = two_layer_rx_dwt_fine_oracle(rxWaves_dwt_base, txInfo_dwt);
psnr_base_dwt = psnr(min(max(reconImgBase_dwt, 0), 255), img, 255);
ssim_base_dwt = ssim(min(max(reconImgBase_dwt, 0), 255), img);

hybrid_target_base = 1048576 + 2 * policy.K * 1024 * 16;

fprintf('\n====================================================\n');
fprintf('                    RESULTS                         \n');
fprintf('====================================================\n');

fprintf('Payload Comparison (bits):\n');
fprintf('  Target Hybrid Base: %d\n', hybrid_target_base);
fprintf('  DWT-fine Base     : %d\n', basePayloadBits_dwt);
fprintf('  DWT-fine Enh      : %d\n', enhPayloadBits_dwt);
if basePayloadBits_dwt == hybrid_target_base
    fprintf('  Base Match Status : PERFECT MATCH\n');
else
    fprintf('  Base Match Status : MISMATCH (Diff: %d bits)\n', basePayloadBits_dwt - hybrid_target_base);
end

fprintf('\nQuality Comparison (Best-case: Base + Enh):\n');
fprintf('  DWT-fine Oracle - PSNR: %6.2f dB | SSIM: %6.4f\n', psnr_full_dwt, ssim_full_dwt);

fprintf('\nQuality Comparison (Degraded: Base Only):\n');
fprintf('  DWT-fine Oracle - PSNR: %6.2f dB | SSIM: %6.4f\n', psnr_base_dwt, ssim_base_dwt);
fprintf('====================================================\n');

figure('Name', 'DWT-fine Oracle Layered Reconstruction', 'Position', [100, 100, 1200, 400]);
subplot(1,3,1);
imshow(uint8(img));
title('Original Image');

subplot(1,3,2);
imshow(uint8(reconImgBase_dwt));
title(sprintf('Base Only\nPSNR: %.2f dB, SSIM: %.4f', psnr_base_dwt, ssim_base_dwt));

subplot(1,3,3);
imshow(uint8(reconImgFull_dwt));
title(sprintf('Base + Enh\nPSNR: %.2f dB, SSIM: %.4f', psnr_full_dwt, ssim_full_dwt));
