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
param.SNR_dB = 7;

frameId = 1;

fprintf('Policy: K=%d, Qb_idx=%d, Qe_idx=%d\n', policy.K, policy.Qb_idx, policy.Qe_idx);

fprintf('Running Hybrid TX...\n');
[txWaves, txInfo] = two_layer_tx_hybrid(img, policy, frameId);

basePayloadBits = sum(txInfo.basePktLens) * 8;
enhPayloadBits = sum(txInfo.enhPktLens) * 8;
fprintf('  Base packets: %d, Base Payload: %d Bits\n', txInfo.numBasePkts, basePayloadBits);
fprintf('  Enh  packets: %d, Enh  Payload: %d Bits\n', txInfo.numEnhPkts, enhPayloadBits);

fprintf('\nRunning RX (Base + Enh)...\n');
rxWaves = txWaves;



% rxWaves = struct('base', {{}}, 'enh', {{}});
% if txInfo.numBasePkts > 0
%     for i = 1:txInfo.numBasePkts
%         [rxSignalB, ~] = channel_model_v1(txWaves.base{i}, param);
%         rxWaves.base{i} = rxSignalB;
%     end
% end

% % Enh packets
% if txInfo.numEnhPkts > 0
%     for i = 1:txInfo.numEnhPkts
%         [rxSignalE, ~] = channel_model_v1(txWaves.enh{i}, param);
%         rxWaves.enh{i} = rxSignalE;
%     end
% end

[reconImgFull, rxInfoFull] = two_layer_rx_hybrid(rxWaves, txInfo);
reconEvalFull = min(max(reconImgFull, 0), 255);

psnr_full = psnr(reconImgFull, img, 255);
ssim_full = ssim(reconImgFull, img);
fprintf('  PSNR: %.2f dB\n', psnr_full);
fprintf('  SSIM: %.4f\n', ssim_full);

fprintf('\nRunning RX (Base Only)...\n');
rxWavesBaseOnly = rxWaves;
rxWavesBaseOnly.enh = {};
[reconImgBase, rxInfoBase] = two_layer_rx_hybrid(rxWavesBaseOnly, txInfo);

reconEvalBase = min(max(reconImgBase, 0), 255);
psnr_base = psnr(reconImgBase, img, 255);
ssim_base = ssim(reconImgBase, img);
fprintf('  PSNR: %.2f dB\n', psnr_base);
fprintf('  SSIM: %.4f\n', ssim_base);

figure('Name', 'Hybrid Layered Reconstruction', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1);
imshow(uint8(img));
title('Original Image');

subplot(1,3,2);
imshow(uint8(reconImgBase));
title(sprintf('Base Only\nPSNR: %.2f dB, SSIM: %.4f', psnr_base, ssim_base));

subplot(1,3,3);
imshow(uint8(reconImgFull));
title(sprintf('Base + Enh\nPSNR: %.2f dB, SSIM: %.4f', psnr_full, ssim_full));

% Extract Subbands for Visualization (from TX)
figure('Name', 'DWT Subbands', 'Position', [100, 100, 800, 800]);
[dwt_res, ~] = DWT(img, struct(), 1);

subplot(2,2,1); imagesc(dwt_res.finalLL); colormap gray; title('LL (Base)'); axis off;
subplot(2,2,2); imagesc(dwt_res.subbands(1).HL); colormap gray; title('HL (Split)'); axis off;
subplot(2,2,3); imagesc(dwt_res.subbands(1).LH); colormap gray; title('LH (Split)'); axis off;
subplot(2,2,4); imagesc(dwt_res.subbands(1).HH); colormap gray; title('HH (Enh)'); axis off;
