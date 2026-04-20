function bold_compare_baseline_vs_ml()
% BOLD_COMPARE_BASELINE_VS_ML  Run baseline vs ML-enhanced and compare
% This is the main evaluation script. It runs the same simulation twice:
%   1. Baseline: rule-based + GA (no neural networks)
%   2. ML-enhanced: with MLP quality predictor + LSTM channel predictor
% Then plots side-by-side comparison for the report.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   bold_compare_baseline_vs_ml()
%
% Produces:
%   bold_comparison.png  - side-by-side performance plot
%   bold_comparison.mat  - raw results for further analysis

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
    fprintf('║     BOLD: Baseline vs ML-Enhanced Comparison                     ║\n');
    fprintf('║     EEME30003 Major Project - Unubold Tuguldur                   ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

    % Shared config
    sharedOpts = struct();
    sharedOpts.nFrames = 50;
    sharedOpts.channelType = 'RICIAN';
    sharedOpts.verbose = false;
    sharedOpts.snrProfile = [ ...
        31 29 27 24 21 18 15 12 10  8 ...
        10 12 15 18 22 26 30 33 28 24 ...
        20 16 13 11  9  8  9  9  9  9 ...
         9  9  9 34 38 41 36 30 24 24 ...
        24 24 24 10 10 10 10 10 10 10];

    %% Run baseline
    fprintf('━━━ Running BASELINE (no ML) ━━━\n');
    baseOpts = sharedOpts;
    baseOpts.useML = false;
    baseOpts.useGaOptimizer = false;
    baseOpts.policyCsvPath = fullfile(tempdir, 'policy_baseline.csv');
    rBase = bold_closed_loop_ml(baseOpts);

    %% Run ML-enhanced
    fprintf('\n━━━ Running ML-ENHANCED ━━━\n');
    mlOpts = sharedOpts;
    mlOpts.useML = true;
    mlOpts.useGaOptimizer = false;
    mlOpts.policyCsvPath = fullfile(tempdir, 'policy_ml.csv');
    rML = bold_closed_loop_ml(mlOpts);

    %% Comparison metrics
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
    fprintf('║  COMPARISON RESULTS                                              ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Metric          │  Baseline  │  ML-Enhanced  │  Improvement    ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');

    avgSsimBase = mean(rBase.ssim);
    avgSsimML   = mean(rML.ssim);
    avgPsnrBase = mean(rBase.psnr);
    avgPsnrML   = mean(rML.psnr);

    fprintf('║  Avg SSIM        │  %.4f    │  %.4f       │  %+.4f         ║\n', ...
        avgSsimBase, avgSsimML, avgSsimML - avgSsimBase);
    fprintf('║  Avg PSNR (dB)   │  %.2f    │  %.2f       │  %+.2f dB       ║\n', ...
        avgPsnrBase, avgPsnrML, avgPsnrML - avgPsnrBase);
    fprintf('║  Min SSIM        │  %.4f    │  %.4f       │  %+.4f         ║\n', ...
        min(rBase.ssim), min(rML.ssim), min(rML.ssim) - min(rBase.ssim));
    fprintf('║  Frames > 0.9    │  %d/%d     │  %d/%d       │                 ║\n', ...
        sum(rBase.ssim > 0.9), rBase.nFrames, sum(rML.ssim > 0.9), rML.nFrames);
    fprintf('║  K range         │  %d–%d     │  %d–%d       │                 ║\n', ...
        min(rBase.K), max(rBase.K), min(rML.K), max(rML.K));
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

    %% Plot comparison
    figure('Position',[50 50 1200 800],'Name','Bold: Baseline vs ML Comparison');
    frames = 1:sharedOpts.nFrames;

    % SNR profile
    subplot(3,2,1);
    plot(frames, sharedOpts.snrProfile, 'k-', 'LineWidth', 2);
    if any(rML.predicted_snr > 0)
        hold on;
        plot(frames, rML.predicted_snr, 'r--', 'LineWidth', 1.5);
        legend('Actual SNR', 'LSTM Predicted');
    end
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel Condition'); grid on;

    % SSIM comparison
    subplot(3,2,2);
    plot(frames, rBase.ssim, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.ssim, 'r-s','MarkerSize',3,'LineWidth',1.5);
    yline(0.9, 'k--', 'Target');
    legend('Baseline','ML-Enhanced','Target');
    xlabel('Frame'); ylabel('SSIM'); title('Image Quality (SSIM)');
    ylim([0 1]); grid on;

    % PSNR comparison
    subplot(3,2,3);
    plot(frames, rBase.psnr, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.psnr, 'r-s','MarkerSize',3,'LineWidth',1.5);
    yline(30, 'k--', 'Target');
    legend('Baseline','ML-Enhanced','Target');
    xlabel('Frame'); ylabel('PSNR (dB)'); title('Image Quality (PSNR)'); grid on;

    % MCS adaptation
    subplot(3,2,4);
    plot(frames, rBase.mcs_enh, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.mcs_enh, 'r-s','MarkerSize',3,'LineWidth',1.5);
    legend('Baseline MCS_{enh}','ML MCS_{enh}');
    xlabel('Frame'); ylabel('MCS'); title('Enhancement MCS Adaptation'); grid on;

    % K adaptation
    subplot(3,2,5);
    plot(frames, rBase.K, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.K, 'r-s','MarkerSize',3,'LineWidth',1.5);
    legend('Baseline K','ML K');
    xlabel('Frame'); ylabel('K'); title('Layer Split Adaptation'); ylim([0 64]); grid on;

    % ML quality prediction accuracy
    subplot(3,2,6);
    if any(rML.ml_ssim > 0)
        scatter(rML.ssim, rML.ml_ssim, 20, 'filled');
        hold on; plot([0 1],[0 1],'k--');
        xlabel('Actual SSIM'); ylabel('ML Predicted SSIM');
        title('Quality Predictor Accuracy');
        xlim([0 1]); ylim([0 1]); grid on;
        % Compute correlation
        valid = rML.ml_ssim > 0;
        if sum(valid) > 2
            R = corrcoef(rML.ssim(valid), rML.ml_ssim(valid));
            text(0.1, 0.85, sprintf('R = %.3f', R(1,2)), 'FontSize', 12);
        end
    else
        text(0.5, 0.5, 'No ML quality predictions', 'HorizontalAlignment','center');
        title('Quality Predictor (not active)');
    end

    sgtitle(sprintf('Bold: Baseline vs ML-Enhanced | ΔSSIM = %+.4f | ΔPSNR = %+.2f dB', ...
        avgSsimML - avgSsimBase, avgPsnrML - avgPsnrBase), 'FontWeight', 'bold');

    saveas(gcf, 'bold_comparison.png');
    fprintf('\n  ✓ Plot saved: bold_comparison.png\n');

    % Save results
    comparison = struct('baseline', rBase, 'ml', rML, 'snrProfile', sharedOpts.snrProfile);
    save('bold_comparison.mat', '-struct', 'comparison');
    fprintf('  ✓ Data saved: bold_comparison.mat\n');

    fprintf('\n  Done! Use these plots in your report.\n\n');
end
