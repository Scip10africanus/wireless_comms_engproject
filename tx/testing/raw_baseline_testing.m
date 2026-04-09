clear; clc; close all;
currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));
imagePath = fullfile(rootDir, 'images', 'Testing examples.jpg');
if ~isfile(imagePath)
    error('Test image not found at: %s', imagePath);
end

rawImg = imread(imagePath);
if size(rawImg, 3) == 3
    img = rgb2gray(rawImg);
else
    img = rawImg;
end
img = double(img); % Keep as double 0-255 for metric calculation

fprintf('Loaded image: %s (%dx%d)\n', 'Testing examples.jpg', size(img,1), size(img,2));
[~, imgName, ~] = fileparts(imagePath);

policy = struct();
frameId = uint8(0);

% Parameter sweeps
MCS_list = [0, 2, 4, 6]; 
SNR_list = [5, 10, 15, 20, 25];
K_list = [0, 1, 2, 3];

for mcs_idx = 1:length(MCS_list)
    mcs_val = MCS_list(mcs_idx);
    policy.mcs_base = uint8(mcs_val);
    policy.mcs_enh  = uint8(mcs_val); % For naming consistency

    fprintf('\n==================================================\n');
    fprintf('Starting MCS Sweep: MCSB=%d, MCSE=%d\n', mcs_val, mcs_val);
    fprintf('==================================================\n');

    % Create MCS directory
    mcsDir = fullfile(rootDir, 'results', 'raw_baseline', sprintf('MCSB_%d_MCSE_%d', mcs_val, mcs_val));
    if ~exist(mcsDir, 'dir'), mkdir(mcsDir); end
    
    % Pre-allocate results storage for this MCS
    psnr_results = zeros(length(K_list), length(SNR_list));
    ssim_results = zeros(length(K_list), length(SNR_list));
    mse_results  = zeros(length(K_list), length(SNR_list));
    
    for k_idx = 1:length(K_list)
        for snr_idx = 1:length(SNR_list)
            param = struct();
            param.channelType = 'RICIAN';
            param.ricianK = K_list(k_idx);
            param.SNR_dB = SNR_list(snr_idx);
            
            % Create SNR directory
            snrDir = fullfile(mcsDir, sprintf('SNR_%d', param.SNR_dB));
            if ~exist(snrDir, 'dir'), mkdir(snrDir); end
            
            fprintf('\n--- Testing: Channel=%s, K=%d, SNR=%d dB, MCS=%d ---\n', ...
                param.channelType, param.ricianK, param.SNR_dB, mcs_val);
            
            BLK_SIZE = 32; % 32x32 = 1024 bytes per block
            [H, W] = size(img);

            % Dimensions of padded image
            padH = ceil(H/BLK_SIZE)*BLK_SIZE;
            padW = ceil(W/BLK_SIZE)*BLK_SIZE;

            % Pad image symmetrically to fit block size
            img_padded = padarray(img, [padH-H padW-W], 'symmetric', 'post');

            nBlocksH = padH / BLK_SIZE;
            nBlocksW = padW / BLK_SIZE;
            totalBlocks = nBlocksH * nBlocksW;

            recon_padded = zeros(padH, padW);

            blkOk = 0;
            blkFail = 0;
            totalPayloadBits = 0;

            for blockId = 0:(totalBlocks - 1)
                bRow = floor(double(blockId) / double(nBlocksW));
                bCol = mod(double(blockId), double(nBlocksW));
                
                rStart = bRow * BLK_SIZE + 1;
                cStart = bCol * BLK_SIZE + 1;
                
                block = img_padded(rStart:rStart + BLK_SIZE - 1, cStart:cStart + BLK_SIZE - 1);
                
                [txWave, txInfo] = raw_tx(block, policy, frameId, blockId);
                
                % Accumulate payload bits
                % Raw baseline uses full block as bytes, so 1024 bytes = 8192 bits
                totalPayloadBits = totalPayloadBits + (txInfo.blk_size^2 * 8);

                [rxWave, channelInfo] = channel_model_v1(txWave, param);
                
                [reconBlock, rxInfo] = raw_rx(rxWave, txInfo);
                
                if rxInfo.blockOK
                    blkOk = blkOk + 1;
                else
                    blkFail = blkFail + 1;
                end
                
                recon_padded(rStart:rStart+BLK_SIZE-1, cStart:cStart+BLK_SIZE-1) = reconBlock;
            end

            % Crop padding to get original size
            reconImg = recon_padded(1:H, 1:W);

            err = img - reconImg;
            mse_val = mean(err(:).^2);
            psnr_val = psnr(reconImg, img, 255);
            ssim_val = ssim(reconImg, img);
            
            % Store results
            psnr_results(k_idx, snr_idx) = psnr_val;
            ssim_results(k_idx, snr_idx) = ssim_val;
            mse_results(k_idx, snr_idx)  = mse_val;
                
            fprintf('--------------------------------------------------\n');
            fprintf('Raw Baseline Results (MCS=%d, K=%d, SNR=%d dB):\n', mcs_val, param.ricianK, param.SNR_dB);
            fprintf('  MSE:  %.4e\n', mse_val);
            fprintf('  PSNR: %.2f dB\n', psnr_val);
            fprintf('  SSIM: %.4f\n', ssim_val);
            fprintf('  Total Payload Bits: %d\n', totalPayloadBits);
            fprintf('  Success: %d, Fail: %d\n', blkOk, blkFail);
            fprintf('--------------------------------------------------\n');
                
            fig_name = sprintf('Raw Baseline - %s (MCS=%d, K=%d, SNR=%d dB)', imgName, mcs_val, param.ricianK, param.SNR_dB);
            fig = figure('Name', fig_name, 'Position', [100, 100, 1200, 400], 'Visible', 'off');
            
            subplot(1,3,1); imshow(uint8(img)); title('Original');
            subplot(1,3,2); imshow(uint8(reconImg)); title(sprintf('Reconstructed\n(PSNR=%.2f dB)', psnr_val));
            subplot(1,3,3); imagesc(abs(err)); colorbar; title('Error Residual');

            % Save images to SNR dir
            reconPath = fullfile(snrDir, sprintf('recon_K%d.png', param.ricianK));
            imwrite(uint8(reconImg), reconPath);
            
            figPath = fullfile(snrDir, sprintf('comparison_K%d.png', param.ricianK));
            saveas(fig, figPath);
            close(fig);
        end
    end

    % Plot curves for this MCS
    fig_psnr = figure('Name', sprintf('Raw Baseline PSNR vs SNR (MCS=%d)', mcs_val), 'Position', [100, 100, 600, 500], 'Visible', 'off');
    hold on; grid on;
    markers = {'-o', '-s', '-^', '-d'};
    for k_idx = 1:length(K_list)
        plot(SNR_list, psnr_results(k_idx, :), markers{k_idx}, 'LineWidth', 2, 'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end
    xlabel('SNR (dB)');
    ylabel('PSNR (dB)');
    title(sprintf('Raw Baseline: PSNR vs SNR (MCS=%d)', mcs_val));
    legend('Location', 'southeast');
    psnrCurvePath = fullfile(mcsDir, 'PSNR_curve.png');
    saveas(fig_psnr, psnrCurvePath);
    close(fig_psnr);

    fig_ssim = figure('Name', sprintf('Raw Baseline SSIM vs SNR (MCS=%d)', mcs_val), 'Position', [750, 100, 600, 500], 'Visible', 'off');
    hold on; grid on;
    for k_idx = 1:length(K_list)
        plot(SNR_list, ssim_results(k_idx, :), markers{k_idx}, 'LineWidth', 2, 'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end
    xlabel('SNR (dB)');
    ylabel('SSIM');
    title(sprintf('Raw Baseline: SSIM vs SNR (MCS=%d)', mcs_val));
    legend('Location', 'southeast');
    ssimCurvePath = fullfile(mcsDir, 'SSIM_curve.png');
    saveas(fig_ssim, ssimCurvePath);
    close(fig_ssim);

    matPath = fullfile(mcsDir, 'metrics_results.mat');
    save(matPath, 'SNR_list', 'K_list', 'psnr_results', 'ssim_results', 'mse_results', 'mcs_val');
end
