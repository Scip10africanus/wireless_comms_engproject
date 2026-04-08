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
policy.Qb_idx = 0; % Q = 1
policy.Qe_idx = 0; % Q = 1
policy.mcs_base = 0;
policy.mcs_enh = 0;

param = struct();
param.channelType = 'RICIAN';
param.ricianK = 0;
param.SNR_dB = 7; % not actually used since we mock channel drops

frameId = 1;

fprintf('====================================================\n');
fprintf('  Shared-Parameter Ranking-Rule Validation Branch   \n');
fprintf('====================================================\n');
fprintf('Testing whether HL/LH magnitude-based base selection\n');
fprintf('improves quality relative to zigzag-based selection.\n');
fprintf('Note: HL/LH selected indices are shared via metadata\n');
fprintf('      and are NOT counted in the payload bits.\n');
fprintf('====================================================\n\n');

fprintf('Policy: Kb=%d, Ke=%d, Qb_idx=%d, Qe_idx=%d\n', policy.K, policy.Ke, policy.Qb_idx, policy.Qe_idx);

fprintf('\n--- Running MAGNITUDE-BASED Hybrid TX ---\n');
[txWaves_mag, txInfo_mag] = two_layer_tx_hybrid_magbase(img, policy, frameId);
basePayloadBits_mag = sum(txInfo_mag.basePktLens) * 8;
enhPayloadBits_mag = sum(txInfo_mag.enhPktLens) * 8;

fprintf('\n--- Running MAGNITUDE-BASED Hybrid RX (Base + Enh) ---\n');
rxWaves_mag_full = txWaves_mag;
[reconImgFull_mag, ~] = two_layer_rx_hybrid_magbase(rxWaves_mag_full, txInfo_mag);
psnr_full_mag = psnr(min(max(reconImgFull_mag, 0), 255), img, 255);
ssim_full_mag = ssim(min(max(reconImgFull_mag, 0), 255), img);

fprintf('\n--- Running MAGNITUDE-BASED Hybrid RX (Base Only) ---\n');
rxWaves_mag_base = txWaves_mag;
rxWaves_mag_base.enh = {};
[reconImgBase_mag, ~] = two_layer_rx_hybrid_magbase(rxWaves_mag_base, txInfo_mag);
psnr_base_mag = psnr(min(max(reconImgBase_mag, 0), 255), img, 255);
ssim_base_mag = ssim(min(max(reconImgBase_mag, 0), 255), img);

fprintf('\n====================================================\n');
fprintf('                    RESULTS                         \n');
fprintf('====================================================\n');

fprintf('Payload Comparison (bits):\n');
fprintf('  MagBase  Base: %d, Enh: %d, Total: %d\n', basePayloadBits_mag, enhPayloadBits_mag, basePayloadBits_mag + enhPayloadBits_mag);

fprintf('\nQuality Comparison (Best-case: Base + Enh):\n');
fprintf('  MagBase  Hybrid - PSNR: %6.2f dB | SSIM: %6.4f\n', psnr_full_mag, ssim_full_mag);

fprintf('\nQuality Comparison (Degraded: Base Only):\n');
fprintf('  MagBase  Hybrid - PSNR: %6.2f dB | SSIM: %6.4f\n', psnr_base_mag, ssim_base_mag);
fprintf('====================================================\n');

figure('Name', 'Magnitude-Based Hybrid Layered Reconstruction', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1);
imshow(uint8(img));
title('Original Image');

subplot(1,3,2);
imshow(uint8(min(max(reconImgBase_mag, 0), 255)));
title(sprintf('Base Only\nPSNR: %.2f dB, SSIM: %.4f', psnr_base_mag, ssim_base_mag));

subplot(1,3,3);
imshow(uint8(min(max(reconImgFull_mag, 0), 255)));
title(sprintf('Base + Enh\nPSNR: %.2f dB, SSIM: %.4f', psnr_full_mag, ssim_full_mag));
