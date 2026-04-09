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
img = double(img); 
fprintf('Loaded image: %s (%dx%d)\n', 'Testing examples.jpg', size(img,1), size(img,2));
[~, imgName, ~] = fileparts(imagePath);

policy = struct();
policy.Qb_idx = uint8(0);  
policy.Qe_idx = uint8(0);  
policy.tx_mode = 'split';
policy.mux_gap_us = 0;

rx_Qb = 1; % using fixed one first
rx_Qe = 1;

frameId = 0;

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

    % Create MCS directory for DWT
    mcsDir = fullfile(rootDir, 'results', 'dwt_baseline', sprintf('MCSB_%d_MCSE_%d', mcs_base_val, mcs_enh_val));
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
            
            [txWaves, txInfo] = two_layer_tx_wavelet(img, policy, frameId);
            
            totalBasePayloadBits = 0;
            totalEnhPayloadBits = 0;

            % Pass through channel
            rxWaves = struct('base', [], 'enh', {{}});
            if txInfo.numBaseBlocks > 0
                for i = 1:txInfo.numBaseBlocks
                    [rxSignal, ~] = channel_model_v1(txWaves.base{i}, param);
                    rxWaves.base{i} = rxSignal;
                    
                    % Payload acc: Each base block is blk_size^2 coeffs * 2 bytes (int16)
                    pktBits = txInfo.blk_size^2 * 2 * 8;
                    totalBasePayloadBits = totalBasePayloadBits + pktBits;
                end
            end
            
            if isfield(txInfo, 'numEnhBlocks')
                for i = 1:txInfo.numEnhBlocks
                    [rxSignal, ~] = channel_model_v1(txWaves.enh{i}, param);
                    rxWaves.enh{i} = rxSignal;
                    
                    % Payload acc: Each enh block is blk_size^2 coeffs * 1 byte (int8)
                    pktBits = txInfo.blk_size^2 * 1 * 8;
                    totalEnhPayloadBits = totalEnhPayloadBits + pktBits;
                end
            end

            % Decode
            [res_hat, rxinfo] = two_layer_rx_testing_dwt(rxWaves, txInfo, rx_Qb, rx_Qe);
            reconImg = IDWT(res_hat, txInfo.level);

            % Statistics
            baseFailCount = numel(rxinfo.baseFailReasons);
            enhFailCount  = numel(rxinfo.enhFailReasons);
            
            baseOkCount = txInfo.numBaseBlocks - baseFailCount;
            if isfield(txInfo, 'numEnhBlocks')
                enhOkCount = txInfo.numEnhBlocks - enhFailCount;
            else
                enhOkCount = 0;
            end

            err = img - reconImg;
            mse_val = mean(err(:).^2);
            psnr_val = psnr(reconImg, img, 255);
            ssim_val = ssim(reconImg, img);
            
            % Store results
            psnr_results(k_idx, snr_idx) = psnr_val;
            ssim_results(k_idx, snr_idx) = ssim_val;
            mse_results(k_idx, snr_idx)  = mse_val;
                
            fprintf('--------------------------------------------------\n');
            fprintf('DWT Results (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB):\n', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
            fprintf('  MSE:  %.4e\n', mse_val);
            fprintf('  PSNR: %.2f dB\n', psnr_val);
            fprintf('  SSIM: %.4f\n', ssim_val);
            fprintf('  Total Payload Bits: %d\n', totalBasePayloadBits + totalEnhPayloadBits);
            fprintf('    Base Payload Bits: %d\n', totalBasePayloadBits);
            fprintf('    Enh  Payload Bits: %d\n', totalEnhPayloadBits);
            fprintf('  Base Packets - Success: %d, Fail: %d (PER: %.2f%%)\n', baseOkCount, baseFailCount, (baseFailCount/(baseOkCount+baseFailCount+eps))*100);
            fprintf('  Enh  Packets - Success: %d, Fail: %d (PER: %.2f%%)\n', enhOkCount, enhFailCount, (enhFailCount/(enhOkCount+enhFailCount+eps))*100);
            fprintf('--------------------------------------------------\n');
                
            fig_name = sprintf('DWT - %s (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', imgName, mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
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
        'Name', sprintf('DWT PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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

    title(ax, sprintf('DWT: PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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
        'Name', sprintf('DWT SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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

    title(ax, sprintf('DWT: SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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