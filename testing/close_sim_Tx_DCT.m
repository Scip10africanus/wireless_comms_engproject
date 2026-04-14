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
MCS_pairs = [0, 0;
             0, 2; 
             2, 0;
             2, 2;
             2, 4; 
             4, 2;
             4, 4;
             4, 6]; 

SNR_list = [5, 10, 15, 20, 25];
K_list = [0, 1, 2, 3];

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
            param.seed = 100 + k_idx;
            
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
            
            batchSize = 16;
            numBatches = ceil(totalBlocks / batchSize);
            
            for batchId = 0:(numBatches - 1)
                startBlock = batchId * batchSize;
                endBlock = min(startBlock + batchSize - 1, totalBlocks - 1);
                numBlocksInBatch = endBlock - startBlock + 1;
                
                blocks = zeros(N, N, numBlocksInBatch);
                for i = 1:numBlocksInBatch
                    bId = startBlock + i - 1;
                    bRow = floor(double(bId) / double(nBlocksW)); 
                    bCol = mod(double(bId), double(nBlocksW));    
                    rStart = bRow*N + 1;  
                    cStart = bCol*N + 1;
                    blocks(:,:,i) = dctCoeffs(rStart:rStart + N - 1, cStart:cStart + N - 1);
                end
                
                % TX:
                if policy.tx_mode == "combined"
                    [txWave, txInfo] = two_layer_tx_DCT(blocks, policy, frameId, batchId, 0);
                    [rxWave, channelInfo] = channel_model_v1(txWave, param);
                    
                    [dctBlockHat, rxinfo] = two_layer_rx_testing_dct(rxWave, txInfo);
                    if rxinfo.blockOK
                        blkOk = blkOk + numBlocksInBatch;
                    else
                        blkFail = blkFail + numBlocksInBatch;
                    end
                    
                    totalBasePayloadBits = totalBasePayloadBits + (txInfo.basePktLen * 8);
                    totalEnhPayloadBits = totalEnhPayloadBits + (txInfo.enhPktLen * 8);

                    for i = 1:numBlocksInBatch
                        bId = startBlock + i - 1;
                        bRow = floor(double(bId) / double(nBlocksW)); 
                        bCol = mod(double(bId), double(nBlocksW));    
                        rStart = bRow*N + 1;  
                        cStart = bCol*N + 1;
                        dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat(:,:,i);
                    end
                else % "split"
                    % base packet
                    [txWaveB, txInfoB] = two_layer_tx_DCT(blocks, policy, frameId, batchId, 1);
                    [rxWaveB, channelInfoB] = channel_model_v1(txWaveB, param);
                    
                    [dctBaseHat, rxinfoB] = two_layer_rx_testing_dct(rxWaveB, txInfoB);
                    if rxinfoB.blockOK
                        blkOk = blkOk + numBlocksInBatch;
                    else
                        blkFail = blkFail + numBlocksInBatch;
                    end
                    
                    totalBasePayloadBits = totalBasePayloadBits + (txInfoB.basePktLen * 8);

                    % enh packet
                    [txWaveE, txInfoE] = two_layer_tx_DCT(blocks, policy, frameId, batchId, 2);
                    [rxWaveE, channelInfoE] = channel_model_v1(txWaveE, param);
                    
                    [dctEnhHat, rxinfoE] = two_layer_rx_testing_dct(rxWaveE, txInfoE);
                    if rxinfoE.blockOK
                        blkOk = blkOk + numBlocksInBatch; % count enh separately or assume success? Just keep counting. Wait, blkOk shouldn't be counted twice. Let's just track overall ok.
                    end
                    
                    totalEnhPayloadBits = totalEnhPayloadBits + (txInfoE.enhPktLen * 8);

                    % merge (non-overlapping supports -> safe)
                    for i = 1:numBlocksInBatch
                        bId = startBlock + i - 1;
                        bRow = floor(double(bId) / double(nBlocksW)); 
                        bCol = mod(double(bId), double(nBlocksW));    
                        rStart = bRow*N + 1;  
                        cStart = bCol*N + 1;
                        dctBlockHat = dctBaseHat(:,:,i) + dctEnhHat(:,:,i);
                        dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
                    end
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
            
            figPath = fullfile(snrDir, sprintf('comparison_K%d', param.ricianK));
            save_image_figure_multi(fig, figPath);
            close(fig); 
            
            % Add spectral comparison
            plot_spectral_comparison(img, reconImg, fullfile(snrDir, sprintf('spectrum_comparison_K%d', param.ricianK)), sprintf('(MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB));
            
        end
    end

    % Plot Performance Curves for this MCS pair
    curveColors = [
        0.20, 0.20, 0.22;   % charcoal
        0.35, 0.47, 0.58;   % muted blue-gray
        0.48, 0.43, 0.38;   % muted brown
        0.42, 0.50, 0.46    % muted sage
    ];

    markerList = {'o', 's', '^', 'd'};

    fig_psnr = figure( ...
        'Name', sprintf('DCT PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
        'Position', [100, 100, 600, 500], ...
        'Color', 'w', ...
        'InvertHardcopy', 'off', ...
        'Visible', 'off');

    ax = axes(fig_psnr);
    hold(ax, 'on');
    grid(ax, 'on');

    for k_idx = 1:length(K_list)
        color_idx = 1 + mod(k_idx-1, size(curveColors, 1));
        marker_idx = 1 + mod(k_idx-1, numel(markerList));

        plot(ax, SNR_list, psnr_results(k_idx, :), ...
            'LineStyle', '-', ...
            'Marker', markerList{marker_idx}, ...
            'Color', curveColors(color_idx, :), ...
            'LineWidth', 2.0, ...
            'MarkerSize', 6.5, ...
            'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end

    ax.XTick = SNR_list;
    ax.XTickLabel = string(SNR_list);
    ax.XTickMode = 'manual';
    ax.XTickLabelMode = 'manual';
    ax.XMinorTick = 'off';
    ax.XLim = [min(SNR_list), max(SNR_list)];

    % Style settings
    ax.Color = [1, 1, 1];                 % white background
    ax.FontName = 'Times New Roman';
    ax.FontSize = 11;
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.0;
    ax.Box = 'on';
    ax.XColor = [0.10, 0.10, 0.10];       % dark gray text/axes
    ax.YColor = [0.10, 0.10, 0.10];
    ax.GridColor = [0.88, 0.88, 0.88];    % light gray grid
    ax.GridAlpha = 0.7;
    ax.Layer = 'top';

    xlabel(ax, 'SNR', ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    ylabel(ax, 'PSNR (dB)', ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    title(ax, sprintf('DCT: PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    lgd = legend(ax, 'Location', 'southeast');
    lgd.Box = 'off';
    lgd.FontSize = 10;
    lgd.TextColor = [0.12, 0.12, 0.12];
    lgd.Color = 'none';

    psnrCurvePath = fullfile(mcsDir, 'PSNR_curve');
    save_plot_multi(fig_psnr, psnrCurvePath);
    close(fig_psnr);

    fig_ssim = figure( ...
        'Name', sprintf('DCT SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
        'Position', [750, 100, 600, 500], ...
        'Color', 'w', ...
        'InvertHardcopy', 'off', ...
        'Visible', 'off');

    ax = axes(fig_ssim);
    hold(ax, 'on');
    grid(ax, 'on');

    for k_idx = 1:length(K_list)
        color_idx = 1 + mod(k_idx-1, size(curveColors, 1));
        marker_idx = 1 + mod(k_idx-1, numel(markerList));

        plot(ax, SNR_list, ssim_results(k_idx, :), ...
            'LineStyle', '-', ...
            'Marker', markerList{marker_idx}, ...
            'Color', curveColors(color_idx, :), ...
            'LineWidth', 2.0, ...
            'MarkerSize', 6.5, ...
            'DisplayName', sprintf('Rician K=%d', K_list(k_idx)));
    end

    ax.XTick = SNR_list;
    ax.XTickLabel = string(SNR_list);
    ax.XTickMode = 'manual';
    ax.XTickLabelMode = 'manual';
    ax.XMinorTick = 'off';
    ax.XLim = [min(SNR_list), max(SNR_list)];

    % Style settings
    ax.Color = [1, 1, 1];
    ax.FontName = 'Times New Roman';
    ax.FontSize = 11;
    ax.FontWeight = 'bold';
    ax.LineWidth = 1.0;
    ax.Box = 'on';
    ax.XColor = [0.10, 0.10, 0.10];
    ax.YColor = [0.10, 0.10, 0.10];
    ax.GridColor = [0.88, 0.88, 0.88];
    ax.GridAlpha = 0.7;
    ax.Layer = 'top';

    xlabel(ax, 'SNR', ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    ylabel(ax, 'SSIM', ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    title(ax, sprintf('DCT: SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
        'Color', [0.08, 0.08, 0.08], ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    lgd = legend(ax, 'Location', 'southeast');
    lgd.Box = 'off';
    lgd.FontSize = 10;
    lgd.TextColor = [0.12, 0.12, 0.12];
    lgd.Color = 'none';

    ssimCurvePath = fullfile(mcsDir, 'SSIM_curve');
    save_plot_multi(fig_ssim, ssimCurvePath);
    close(fig_ssim);

    matPath = fullfile(mcsDir, 'metrics_results.mat');
    save(matPath, 'SNR_list', 'K_list', 'psnr_results', 'ssim_results', 'mse_results', 'mcs_base_val', 'mcs_enh_val');
end
