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
policy.K = 20; % Kb
policy.Ke = 32; % Ke

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

SNR_list = [5, 10,15,20,25];
K_list = [0, 1, 2, 3];

for mcs_idx = 1:size(MCS_pairs, 1)
    mcs_base_val = MCS_pairs(mcs_idx, 1);
    mcs_enh_val  = MCS_pairs(mcs_idx, 2);
    
    policy.mcs_base = uint8(mcs_base_val);
    policy.mcs_enh  = uint8(mcs_enh_val);

    fprintf('\n==================================================\n');
    fprintf('Starting MCS Sweep: MCSB=%d, MCSE=%d\n', mcs_base_val, mcs_enh_val);
    fprintf('==================================================\n');

    % Create MCS directory for Hybrid
    mcsDir = fullfile(rootDir, 'results', 'hybrid_baseline', sprintf('MCSB_%d_MCSE_%d', mcs_base_val, mcs_enh_val));
    if ~exist(mcsDir, 'dir'), mkdir(mcsDir); end
    
    % Pre-allocate results storage
    psnr_results = zeros(length(K_list), length(SNR_list));
    ssim_results = zeros(length(K_list), length(SNR_list));
    mse_results  = zeros(length(K_list), length(SNR_list));
    
    psnr_base_results = zeros(length(K_list), length(SNR_list));
    ssim_base_results = zeros(length(K_list), length(SNR_list));
    mse_base_results  = zeros(length(K_list), length(SNR_list));

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
            
            % TX: Hybrid Encode
            [txWaves, txInfo] = two_layer_tx_hybrid(img, policy, frameId);
            
            totalBasePayloadBits = sum(txInfo.basePktLens) * 8;
            totalEnhPayloadBits = sum(txInfo.enhPktLens) * 8;

            % Pass through channel
            rxWaves = struct('base', {{}}, 'enh', {{}});
            
            % Base packets
            if txInfo.numBasePkts > 0
                for i = 1:txInfo.numBasePkts
                    [rxSignalB, ~] = channel_model_v1(txWaves.base{i}, param);
                    rxWaves.base{i} = rxSignalB;
                end
            end
            
            % Enh packets
            if txInfo.numEnhPkts > 0
                for i = 1:txInfo.numEnhPkts
                    [rxSignalE, ~] = channel_model_v1(txWaves.enh{i}, param);
                    rxWaves.enh{i} = rxSignalE;
                end
            end

            % RX: Hybrid Decode (Base + Enh)
            [img_hat_full, rxInfo_full] = two_layer_rx_hybrid(rxWaves, txInfo);
            reconImgFull = min(max(img_hat_full, 0), 255);
            
            % RX: Hybrid Decode (Base Only)
            rxWavesBaseOnly = rxWaves;
            rxWavesBaseOnly.enh = {};
            [img_hat_base, rxInfo_base] = two_layer_rx_hybrid(rxWavesBaseOnly, txInfo);
            reconImgBase = min(max(img_hat_base, 0), 255);

            % Statistics
            baseOkCount = txInfo.numBasePkts - rxInfo_full.baseFailCount;
            enhOkCount = txInfo.numEnhPkts - rxInfo_full.enhFailCount;

            % Base + Enh Evaluation
            err_full = img - reconImgFull;
            mse_val_full = mean(err_full(:).^2);
            psnr_val_full = psnr(reconImgFull, img, 255);
            ssim_val_full = ssim(reconImgFull, img);
            
            % Base Only Evaluation
            err_base = img - reconImgBase;
            mse_val_base = mean(err_base(:).^2);
            psnr_val_base = psnr(reconImgBase, img, 255);
            ssim_val_base = ssim(reconImgBase, img);
            
            % Store results
            psnr_results(k_idx, snr_idx) = psnr_val_full;
            ssim_results(k_idx, snr_idx) = ssim_val_full;
            mse_results(k_idx, snr_idx)  = mse_val_full;
            
            psnr_base_results(k_idx, snr_idx) = psnr_val_base;
            ssim_base_results(k_idx, snr_idx) = ssim_val_base;
            mse_base_results(k_idx, snr_idx)  = mse_val_base;
                
            fprintf('--------------------------------------------------\n');
            fprintf('Hybrid Results (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB):\n', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
            fprintf('  [Base+Enh] MSE: %.4e | PSNR: %.2f dB | SSIM: %.4f\n', mse_val_full, psnr_val_full, ssim_val_full);
            fprintf('  [BaseOnly] MSE: %.4e | PSNR: %.2f dB | SSIM: %.4f\n', mse_val_base, psnr_val_base, ssim_val_base);
            fprintf('  Total Payload Bits: %d\n', totalBasePayloadBits + totalEnhPayloadBits);
            fprintf('    Base Payload Bits: %d\n', totalBasePayloadBits);
            fprintf('    Enh  Payload Bits: %d\n', totalEnhPayloadBits);
            fprintf('  Base Packets - Success: %d, Fail: %d (PER: %.2f%%)\n', baseOkCount, rxInfo_full.baseFailCount, (rxInfo_full.baseFailCount/(txInfo.numBasePkts+eps))*100);
            if ~rxInfo_full.baseOk, fprintf('    First Base Fail Reason: %s\n', rxInfo_full.baseFailReason); end
            fprintf('  Enh  Packets - Success: %d, Fail: %d (PER: %.2f%%)\n', enhOkCount, rxInfo_full.enhFailCount, (rxInfo_full.enhFailCount/(txInfo.numEnhPkts+eps))*100);
            if ~rxInfo_full.enhOk, fprintf('    First Enh Fail Reason: %s\n', rxInfo_full.enhFailReason); end
            fprintf('--------------------------------------------------\n');
                
            fig_name = sprintf('Hybrid - %s (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', imgName, mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB);
            fig = figure('Name', fig_name, 'Position', [100, 100, 1200, 400], 'Visible', 'off');
            
            subplot(1,3,1); imshow(uint8(img)); title('Original');
            subplot(1,3,2); imshow(uint8(reconImgBase)); title(sprintf('Base Only\n(PSNR=%.2f dB)', psnr_val_base));
            subplot(1,3,3); imshow(uint8(reconImgFull)); title(sprintf('Base + Enh\n(PSNR=%.2f dB)', psnr_val_full));

            % Save images to SNR dir
            reconPath = fullfile(snrDir, sprintf('recon_K%d_full.png', param.ricianK));
            imwrite(uint8(reconImgFull), reconPath);
            reconBasePath = fullfile(snrDir, sprintf('recon_K%d_base.png', param.ricianK));
            imwrite(uint8(reconImgBase), reconBasePath);
            
            figPath = fullfile(snrDir, sprintf('comparison_K%d', param.ricianK));
            save_image_figure_multi(fig, figPath);
            close(fig); 
            
            % Add spectral comparison
            plot_spectral_comparison(img, reconImgFull, fullfile(snrDir, sprintf('spectrum_comparison_K%d_full', param.ricianK)), sprintf('Full (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB));
            plot_spectral_comparison(img, reconImgBase, fullfile(snrDir, sprintf('spectrum_comparison_K%d_base', param.ricianK)), sprintf('Base Only (MCSB=%d, MCSE=%d, K=%d, SNR=%d dB)', mcs_base_val, mcs_enh_val, param.ricianK, param.SNR_dB));
            
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
        'Name', sprintf('Hybrid PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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

    title(ax, sprintf('Hybrid: PSNR vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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
        'Name', sprintf('Hybrid SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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

    title(ax, sprintf('Hybrid: SSIM vs SNR (MCSB=%d, MCSE=%d)', mcs_base_val, mcs_enh_val), ...
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
    save(matPath, 'SNR_list', 'K_list', ...
        'psnr_results', 'ssim_results', 'mse_results', ...
        'psnr_base_results', 'ssim_base_results', 'mse_base_results', ...
        'mcs_base_val', 'mcs_enh_val');
end