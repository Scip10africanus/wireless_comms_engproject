clear; clc; close all;
currentDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(currentDir);
addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'test_functions'));

imagePath = fullfile(rootDir, 'images', 'Testing examples.jpg');
rawImg = imread(imagePath);
if size(rawImg, 3) == 3
    img = rgb2gray(rawImg);
else
    img = rawImg;
end
img = double(img); 
fprintf('Loaded image: %s (%dx%d)\n', 'Testing examples.jpg', size(img,1), size(img,2));
[~, imgName, ~] = fileparts(imagePath);
policy = struct();
policy.Qb_idx = uint8(1);   % -> 0.05
policy.Qe_idx = uint8(1);   % -> 0.05
policy.tx_mode = 'split'; % 'split' or 'combined'
policy.mux_gap_us = 0;
policy.K = 20;
frameId = 0;

Qb_list = [0.02 0.05 0.1 0.2];
Qe_list = [0.02 0.05 0.1 0.2];

fprintf('Policy: Qb=%f, Qe=%f\n', Qb_list(policy.Qb_idx + 1), Qe_list(policy.Qe_idx + 1));   

% Parameter sweeps
% Each row is [MCS_B, MCS_E]
MCS_pairs = [2, 0;
             4, 0;
             4, 2; 
             4, 4; 
             4, 6;
             6, 0;
             6, 2;
             6, 4]; 

SNR_list = [5, 10, 15, 20, 25];
K_list = [1, 2, 3];

for mcs_idx = 1:size(MCS_pairs, 1)
    mcs_base_val = MCS_pairs(mcs_idx, 1);
    mcs_enh_val  = MCS_pairs(mcs_idx, 2);
    
    policy.mcs_base = uint8(mcs_base_val);
    policy.mcs_enh  = uint8(mcs_enh_val);

    fprintf('\n==================================================\n');
    fprintf('Starting MCS Sweep: MCSB=%d, MCSE=%d\n', mcs_base_val, mcs_enh_val);
    fprintf('==================================================\n');

    % Create MCS directory for DCT
    mcsDir = fullfile(rootDir, 'results', 'dct_baseline', sprintf('MCSB_%d_MCSE_%d', mcs_base_val, mcs_enh_val));
    if ~exist(mcsDir, 'dir'), mkdir(mcsDir); end
    
    % Pre-allocate results storage
    psnr_results = zeros(length(K_list), length(SNR_list));
    ssim_results = zeros(length(K_list), length(SNR_list));
    mse_results  = zeros(length(K_list), length(SNR_list));

    for k_idx = 1:length(K_list)
        for snr_idx = 1:length(SNR_list)
            
            param = struct();
            param.channelType = 'RICIAN';
            param.ricianK = K_list(k_idx);
            param.SNR_dB = SNR_list(snr_idx);
            
            snrDir = fullfile(mcsDir, sprintf('SNR_%d', param.SNR_dB));
            if ~exist(snrDir, 'dir'), mkdir(snrDir); end
            
            fprintf('\n--- Testing: Channel=%s, K=%d, SNR=%d dB, MCSB=%d, MCSE=%d ---\n', ...
                param.channelType, param.ricianK, param.SNR_dB, mcs_base_val, mcs_enh_val);
            
            N = 8;
            [dctCoeffs, dctInfo] = DCT_Block(double(img), N);
            
            % DCT matrix for recon
            dctReco = zeros(dctInfo.Hp, dctInfo.Wp);
            
            blkOk = 0;
            blkFail = 0;
            nBlocksW = dctInfo.Wp / N;
            nBlocksH = dctInfo.Hp / N;
            totalBlocks = nBlocksH * nBlocksW;
            
            totalBasePayloadBits = 0;
            totalEnhPayloadBits = 0;
            
            for blockId = 0:(totalBlocks - 1)
                bRow = floor(double(blockId) / double(nBlocksW)); 
                bCol = mod(double(blockId), double(nBlocksW));    
                rStart = bRow*N + 1;  
                cStart = bCol*N + 1;
                
                block = dctCoeffs(rStart:rStart + N - 1, cStart:cStart + N - 1);
                
                % TX:
                if policy.tx_mode == "combined"
                    [txWave, txInfo] = two_layer_tx_DCT(block ,policy, frameId, blockId, 0);
                    [rxWave, channelInfo] = channel_model_v1(txWave, param);
                    
                    [dctBlockHat, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo);
                    if rxinfo.blockOK
                        blkOk = blkOk + 1;
                    else
                        blkFail = blkFail + 1;
                    end
                    
                    totalBasePayloadBits = totalBasePayloadBits + (txInfo.basePktLen * 8);
                    totalEnhPayloadBits = totalEnhPayloadBits + (txInfo.enhPktLen * 8);

                    dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
                else % "split"
                    % base packet
                    [txWaveB, txInfoB] = two_layer_tx_DCT(block, policy, frameId, blockId, 1);
                    [rxWaveB, channelInfoB] = channel_model_v1(txWaveB, param);
                    
                    [dctBaseHat, rxinfoB] = two_layer_rx_testing_dct(rxWaveB, txInfoB);
                    if rxinfoB.blockOK
                        blkOk = blkOk + 1;
                    else
                        blkFail = blkFail + 1;
                    end
                    
                    totalBasePayloadBits = totalBasePayloadBits + (txInfoB.basePktLen * 8);

                    % enh packet
                    [txWaveE, txInfoE] = two_layer_tx_DCT(block, policy, frameId, blockId, 2);
                    [rxWaveE, channelInfoE] = channel_model_v1(txWaveE, param);
                    
                    [dctEnhHat, rxinfoE] = two_layer_rx_testing_dct(rxWaveE, txInfoE);
                    if rxinfoE.blockOK
                        blkOk = blkOk + 1;
                    else
                        blkFail = blkFail + 1;
                    end
                    
                    totalEnhPayloadBits = totalEnhPayloadBits + (txInfoE.enhPktLen * 8);

                    % merge (non-overlapping supports -> safe)
                    dctBlockHat = dctBaseHat + dctEnhHat;
                    dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
                end
            end

            % recon
            reconImg = IDCT(dctReco, dctInfo); % IDCT

            err = img - reconImg;
            mse_val = mean(err(:).^2);
            psnr_val = psnr(reconImg, img, 255);
            ssim_val = ssim(reconImg, img);
            
            % Store results
            psnr_results(k_idx, snr_idx) = psnr_val;
            ssim_results(k_idx, snr_idx) = ssim_val;
            mse_results(k_idx, snr_idx)  = mse_val;
                
            fprintf('--------------------------------------------------\n');
            fprintf('DCT Results (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB):\n', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
            fprintf('  MSE:  %.4e\n', mse_val);
            fprintf('  PSNR: %.2f dB\n', psnr_val);
            fprintf('  SSIM: %.4f\n', ssim_val);
            fprintf('  Total Payload Bits: %d\n', totalBasePayloadBits + totalEnhPayloadBits);
            if policy.tx_mode == "split"
                fprintf('    Base Payload Bits: %d\n', totalBasePayloadBits);
                fprintf('    Enh  Payload Bits: %d\n', totalEnhPayloadBits);
            end
            fprintf('  Success: %d, Fail: %d\n', blkOk, blkFail);
            fprintf('--------------------------------------------------\n');
                
            fig_name = sprintf('DCT - %s (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', imgName, mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
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

    % Plot Performance Curves for this MCS pair
    fig_psnr = figure('Name', sprintf('DCT PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), 'Position', [100, 100, 600, 500], 'Visible', 'off');
    hold on; grid on;
    markers = {'-o', '-s', '-^'};
    for k_idx = 1:length(K_list)
        plot(SNR_list, psnr_results(k_idx, :), markers{k_idx}, 'LineWidth', 2, 'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end
    xlabel('SNR (dB)');
    ylabel('PSNR (dB)');
    title(sprintf('DCT: PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val));
    legend('Location', 'southeast');
    psnrCurvePath = fullfile(mcsDir, 'PSNR_curve.png');
    saveas(fig_psnr, psnrCurvePath);
    close(fig_psnr);

    fig_ssim = figure('Name', sprintf('DCT SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), 'Position', [750, 100, 600, 500], 'Visible', 'off');
    hold on; grid on;
    for k_idx = 1:length(K_list)
        plot(SNR_list, ssim_results(k_idx, :), markers{k_idx}, 'LineWidth', 2, 'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end
    xlabel('SNR (dB)');
    ylabel('SSIM');
    title(sprintf('DCT: SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val));
    legend('Location', 'southeast');
    ssimCurvePath = fullfile(mcsDir, 'SSIM_curve.png');
    saveas(fig_ssim, ssimCurvePath);
    close(fig_ssim);

    matPath = fullfile(mcsDir, 'metrics_results.mat');
    save(matPath, 'SNR_list', 'K_list', 'psnr_results', 'ssim_results', 'mse_results', 'mcs_base_val', 'mcs_enh_val');
end
