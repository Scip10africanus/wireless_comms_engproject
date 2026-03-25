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
policy.Qb_idx = uint8(0);   % -> 0.01
policy.Qe_idx = uint8(0);   % -> 0.01
policy.mcs_base = uint8(0);
policy.mcs_enh  = uint8(2);
policy.tx_mode = 'split'; % 'split' or 'combined'
policy.mux_gap_us = 0;
policy.K = 20;

param = struct();
param.SNR_dB = 5;
param.channelType = 'RICIAN';
param.ricianK = 1;

frameId = 0;

Qb_list = [0.01 0.02 0.05 0.1];
Qe_list = [0.01 0.02 0.05 0.1];

fprintf('Policy: Qb=%f, Qe=%f, MCS_B=%d, MCS_E=%d\n', ...
    Qb_list(policy.Qb_idx + 1), Qe_list(policy.Qe_idx + 1), policy.mcs_base, policy.mcs_enh);   

N = 8;
[dctCoeffs, dctInfo] = DCT_Block(double(img), N);
% DCT matrix for recon
dctReco = zeros(dctInfo.Hp, dctInfo.Wp);
blkOk = 0;
blkFail = 0;
nBlocksW = dctInfo.Wp / N;
nBlocksH = dctInfo.Hp / N;
baseFailReasons = {};
enhFailReasons = {};
combinedFailReasons = {};
for blockId = 0:(nBlocksH * nBlocksW) -1 
        [rStart, cStart] = block_pos(blockId, nBlocksW, N);
        block = dctCoeffs(rStart:rStart + N - 1,cStart:cStart + N - 1);
        % TX:
        if policy.tx_mode == "combined" % combined
            [txWave, txInfo] = two_layer_tx_DCT(block ,policy, frameId,blockId,0);
            [rxWave, channelInfo] = channel_model_v1(txWave,param);
            
            [dctBlockHat, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo);
            if rxinfo.blockOK
                blkOk = blkOk+1;
            else
                blkFail = blkFail+1;
                combinedFailReasons{end+1} = rxinfo.combinedFailReason; %#ok<SAGROW>
            end

            dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
            blkOk = blkOk+1;
        else % "split"
            % base packet
            [txWave, txInfo] = two_layer_tx_DCT(block, policy, frameId, blockId, 1);
            [rxWave, channelInfo] = channel_model_v1(txWave,param);
            
            [dctBaseHat, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo);
            if rxinfo.blockOK
                blkOk = blkOk+1;
            else
                blkFail = blkFail+1;
                baseFailReasons{end+1} = rxinfo.baseFailReason; %#ok<SAGROW>
            end

            % enh packet
            [txWave, txInfo] = two_layer_tx_DCT(block, policy, frameId, blockId, 2);
            
            [rxWave, channelInfo] = channel_model_v1(txWave,param);
            
            [dctEnhHat, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo);
            if rxinfo.blockOK
                blkOk = blkOk+1;
            else
                blkFail = blkFail+1;
                enhFailReasons{end+1} = rxinfo.enhFailReason; %#ok<SAGROW>
            end

            % merge (non-overlapping supports -> safe)
            dctBlockHat = dctBaseHat + dctEnhHat;
            dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
            blkOk = blkOk + 1;
        end
end

fprintf('Running IDCT...\n');
% recon
reconImg = IDCT(dctReco,dctInfo);% IDCT

if ~isempty(baseFailReasons)
    disp('Base fail reasons:');
    disp(unique(string(rxinfo.baseFailReasons)));
end

if ~isempty(enhFailReasons)
    disp('Enh fail reasons:');
    disp(unique(string(rxinfo.enhFailReasons)));
end

if ~isempty(combinedFailReasons)
    disp('Combined fail reasons:');
    disp(unique(string(combinedFailReasons)));
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

function [rStart, cStart] = block_pos(blockId, nBlocksW, N)
% blockId: 0-based
    bRow = floor(double(blockId) / double(nBlocksW)); % 0-based
    bCol = mod(double(blockId), double(nBlocksW));    % 0-based
    rStart = bRow*N + 1;  % 1-based index
    cStart = bCol*N + 1;
end