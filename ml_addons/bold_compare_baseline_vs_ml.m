function bold_compare_baseline_vs_ml(opts)
% BOLD_COMPARE_BASELINE_VS_ML  Run baseline vs ML-enhanced and compare
% Runs:
%   1. Baseline: rules only, no ML, GA on by default (same K/Q GA as ML path)
%   2. ML-enhanced: MLP + LSTM + GA (symbol denoiser off by default in closed_loop)
%   3. Optional: ML + GA off when runMlGaOffAblation=true (GA contribution under ML)
% Then plots and saves metrics for the report.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   bold_compare_baseline_vs_ml()  % default: GA report varying SNR + rx v2 (~0.95 baseline SSIM)
%   bold_compare_baseline_vs_ml(struct('useGaReportVarying', false, 'nFrames', 60))  % stress profile
%   bold_compare_baseline_vs_ml(struct('useGaOptimizer', false))
%   bold_smoke_ml_ga_ablation  % 32 frames on GA varying trace
%
% Produces:
%   bold_comparison.png  - side-by-side performance plot
%   bold_comparison.mat  - raw results + tail_metrics, snrProfileStress
%
% opts (optional struct):
%   .nFrames          — default 50 with GA report profile (match generate_ga_report_results_dct)
%   .snrProfile       — if non-empty, overrides all built-in profiles
%   .psnrFloorDb      — tail metric: count frames with PSNR below this (default 30)
%   .useGaReportVarying — default TRUE: same 50-pt SNR as GA report "Varying (realistic)"
%   .useLegacyProfile — if true, raw legacy 50-pt trace (ignores useGaReportVarying)
%   .useRxV2          — default follows useGaReportVarying (TRUE=v2 like run_dct benchmark)
%   .channelSnrPenaltyDb — AWGN harsher than nominal SNR (default 3.5); 0 = clean match
%   .ricianK — LOS K in dB for channel_model_v1 (default 3; keep team Rician)
%   .useGaOptimizer — default TRUE for baseline and ML runs (set false to disable GA everywhere)
%   .runMlGaOffAblation — default FALSE (saves ~1/3 runtime); set true for GA ablation figure
%   .doPlot            — passed to bold_closed_loop_ml (default false here for speed)
%   .gaSettings        — optional struct for GA popSize/generations (default: lighter GA in compare)
%   .useSymbolDenoiser — optional; passed through if you add it to sharedOpts (see bold_closed_loop_ml)
%   .baselineGaBlendWeight — GA vs rules weight for BASELINE run only (rx_policy_update Step 2).
%         Default 0.40 (global default is 0.38); use 0.38 for strict parity with older baselines.

    if nargin < 1 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'useGaReportVarying'), opts.useGaReportVarying = true; end
    if ~isfield(opts,'nFrames'), opts.nFrames = 50; end
    if ~isfield(opts,'psnrFloorDb'), opts.psnrFloorDb = 30; end
    if ~isfield(opts,'useLegacyProfile'), opts.useLegacyProfile = false; end
    if ~isfield(opts,'snrProfile'), opts.snrProfile = []; end
    if ~isfield(opts,'channelSnrPenaltyDb'), opts.channelSnrPenaltyDb = 3.5; end
    if ~isfield(opts,'ricianK'), opts.ricianK = 3; end
    if ~isfield(opts,'useGaOptimizer'), opts.useGaOptimizer = true; end
    if ~isfield(opts,'runMlGaOffAblation'), opts.runMlGaOffAblation = false; end
    if ~isfield(opts,'doPlot'), opts.doPlot = false; end
    if ~isfield(opts,'baselineGaBlendWeight')
        opts.baselineGaBlendWeight = 0.40;
    elseif isempty(opts.baselineGaBlendWeight)
        opts.baselineGaBlendWeight = 0.38;
    end

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
    fprintf('║     BOLD: Baseline vs ML-Enhanced Comparison                     ║\n');
    fprintf('║     EEME30003 Major Project - Unubold Tuguldur                   ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

    % Shared config — default matches generate_ga_report_results_dct "Varying (realistic)"
    sharedOpts = struct();
    sharedOpts.nFrames = opts.nFrames;
    sharedOpts.channelType = 'RICIAN';
    sharedOpts.verbose = false;
    sharedOpts.doPlot = opts.doPlot;
    sharedOpts.channelSnrPenaltyDb = double(opts.channelSnrPenaltyDb);
    sharedOpts.ricianK = double(opts.ricianK);
    % GA: full pop/gen for GA-like profiles; lighter GA only for built-in stress ramp
    useBuiltInStress = isempty(opts.snrProfile) && ~opts.useLegacyProfile && ~opts.useGaReportVarying;
    if ~isfield(opts,'gaSettings') || isempty(opts.gaSettings)
        if useBuiltInStress
            sharedOpts.gaSettings = struct('popSize', 20, 'generations', 5, 'verbose', false);
        else
            sharedOpts.gaSettings = struct('popSize', 30, 'generations', 10, 'verbose', false);
        end
    else
        sharedOpts.gaSettings = opts.gaSettings;
    end
    if isfield(opts,'useSymbolDenoiser')
        sharedOpts.useSymbolDenoiser = logical(opts.useSymbolDenoiser);
    end
    if ~isfield(opts,'useRxV2')
        opts.useRxV2 = logical(opts.useGaReportVarying);
    end
    sharedOpts.useRxV2 = logical(opts.useRxV2);

    if ~isempty(opts.snrProfile)
        sharedOpts.snrProfile = opts.snrProfile(:).';
    elseif opts.useLegacyProfile
        leg = [ ...
            31 29 27 24 21 18 15 12 10  8 ...
            10 12 15 18 22 26 30 33 28 24 ...
            20 16 13 11  9  8  9  9  9  9 ...
             9  9  9 34 38 41 36 30 24 24 ...
            24 24 24 10 10 10 10 10 10 10];
        nf = max(1, round(double(sharedOpts.nFrames)));
        if numel(leg) < nf
            sharedOpts.snrProfile = [leg, leg(end) * ones(1, nf - numel(leg))];
        else
            sharedOpts.snrProfile = leg(1:nf);
        end
        sharedOpts.nFrames = numel(sharedOpts.snrProfile);
    elseif opts.useGaReportVarying
        trace = bold_ga_report_varying_profile();
        nf = max(1, round(double(sharedOpts.nFrames)));
        if numel(trace) < nf
            sharedOpts.snrProfile = [trace, trace(end) * ones(1, nf - numel(trace))];
        else
            sharedOpts.snrProfile = trace(1:nf);
        end
        sharedOpts.nFrames = numel(sharedOpts.snrProfile);
    else
        sharedOpts.snrProfile = bold_build_stress_snr_profile(sharedOpts.nFrames);
    end
    fprintf('  Config: nFrames=%d | SNR min/max = %.0f / %.0f dB | frames in [8,15] dB: %d | useRxV2=%d\n', ...
        sharedOpts.nFrames, min(sharedOpts.snrProfile), max(sharedOpts.snrProfile), ...
        sum(sharedOpts.snrProfile >= 8 & sharedOpts.snrProfile <= 15), sharedOpts.useRxV2);
    fprintf('  Channel: Rician K=%g dB | channelSnrPenaltyDb=%g | GAvary=%d | useGaOptimizer=%d | runMlGaOffAblation=%d\n', ...
        sharedOpts.ricianK, sharedOpts.channelSnrPenaltyDb, logical(opts.useGaReportVarying), ...
        logical(opts.useGaOptimizer), logical(opts.runMlGaOffAblation));
    fprintf('  Baseline GA blend weight (baseline only): %.2f  (ML uses default 0.38)\n', ...
        double(opts.baselineGaBlendWeight));

    %% Run baseline
    fprintf('━━━ Running BASELINE (no ML, GA=%s) ━━━\n', bold_onoff(opts.useGaOptimizer));
    baseOpts = sharedOpts;
    baseOpts.gaBlendWeight = double(opts.baselineGaBlendWeight);
    baseOpts.useML = false;
    baseOpts.useGaOptimizer = opts.useGaOptimizer;
    baseOpts.policyCsvPath = fullfile(tempdir, 'policy_baseline.csv');
    rBase = bold_closed_loop_ml(baseOpts);

    %% Run ML-enhanced (GA follows opts.useGaOptimizer)
    fprintf('\n━━━ Running ML-ENHANCED (GA=%s) ━━━\n', bold_onoff(opts.useGaOptimizer));
    mlOpts = sharedOpts;
    mlOpts.useML = true;
    mlOpts.useGaOptimizer = opts.useGaOptimizer;
    mlOpts.policyCsvPath = fullfile(tempdir, 'policy_ml.csv');
    rML = bold_closed_loop_ml(mlOpts);

    %% Ablation: same ML stack with GA off (only meaningful when opts.useGaOptimizer is true)
    rMLGaOff = [];
    tailMGO = [];
    hasAblation = logical(opts.runMlGaOffAblation) && logical(opts.useGaOptimizer);
    if hasAblation
        fprintf('\n━━━ Running ML + GA OFF (ablation / comparison) ━━━\n');
        mlGaOffOpts = sharedOpts;
        mlGaOffOpts.useML = true;
        mlGaOffOpts.useGaOptimizer = false;
        mlGaOffOpts.policyCsvPath = fullfile(tempdir, 'policy_ml_ga_off.csv');
        rMLGaOff = bold_closed_loop_ml(mlGaOffOpts);
        tailMGO = bold_tail_metrics(rMLGaOff, opts.psnrFloorDb);
    end

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
    fprintf('║  Frames > 0.9    │  %d/%d     │  %d/%d       │                 ║\n', ...
        sum(rBase.ssim > 0.9), rBase.nFrames, sum(rML.ssim > 0.9), rML.nFrames);
    fprintf('║  K range         │  %d–%d     │  %d–%d       │                 ║\n', ...
        min(rBase.K), max(rBase.K), min(rML.K), max(rML.K));
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  TAIL / STRESS METRICS (mean SSIM saturated — use these too)     ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');

    tailB = bold_tail_metrics(rBase, opts.psnrFloorDb);
    tailM = bold_tail_metrics(rML, opts.psnrFloorDb);

    fprintf('║  Min SSIM        │  %.4f    │  %.4f       │  %+.4f         ║\n', ...
        tailB.minSsim, tailM.minSsim, tailM.minSsim - tailB.minSsim);
    fprintf('║  P5 SSIM (%%ile)  │  %.4f    │  %.4f       │  %+.4f         ║\n', ...
        tailB.p5Ssim, tailM.p5Ssim, tailM.p5Ssim - tailB.p5Ssim);
    fprintf('║  Min PSNR (dB)   │  %.2f    │  %.2f       │  %+.2f         ║\n', ...
        tailB.minPsnr, tailM.minPsnr, tailM.minPsnr - tailB.minPsnr);
    fprintf('║  P5 PSNR (dB)    │  %.2f    │  %.2f       │  %+.2f         ║\n', ...
        tailB.p5Psnr, tailM.p5Psnr, tailM.p5Psnr - tailB.p5Psnr);
    fprintf('║  Enh dropped     │  %d/%d     │  %d/%d       │  %+d frames      ║\n', ...
        tailB.enhDropped, rBase.nFrames, tailM.enhDropped, rML.nFrames, ...
        tailM.enhDropped - tailB.enhDropped);
    fprintf('║  PSNR < %.0f dB   │  %d/%d     │  %d/%d       │  %+d frames      ║\n', ...
        opts.psnrFloorDb, tailB.framesPsnrBelow, rBase.nFrames, ...
        tailM.framesPsnrBelow, rML.nFrames, tailM.framesPsnrBelow - tailB.framesPsnrBelow);
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

    if hasAblation
        fprintf('\n');
        fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
        fprintf('║  ML ABLATION: GA on vs GA off (same ML path)                     ║\n');
        fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
        fprintf('║  Δ mean SSIM (ML+GA minus ML GA-off) = %+.4f                    ║\n', mean(rML.ssim) - mean(rMLGaOff.ssim));
        fprintf('║  Δ mean PSNR (ML+GA minus ML GA-off) = %+.2f dB                 ║\n', mean(rML.psnr) - mean(rMLGaOff.psnr));
        fprintf('║  Δ min SSIM                          = %+.4f                    ║\n', tailM.minSsim - tailMGO.minSsim);
        fprintf('║  Δ P5 SSIM                           = %+.4f                    ║\n', tailM.p5Ssim - tailMGO.p5Ssim);
        fprintf('╚══════════════════════════════════════════════════════════════════╝\n');
    end

    %% Plot comparison (capture figure handle — gcf can be 0 in some batch/headless setups)
    figCmp = figure('Position',[50 50 1200 960],'Name','Bold: Baseline vs ML Comparison');
    nF = sharedOpts.nFrames;
    frames = 1:nF;

    % SNR profile
    subplot(4,2,1);
    plot(frames, sharedOpts.snrProfile, 'k-', 'LineWidth', 2);
    if any(rML.predicted_snr > 0)
        hold on;
        plot(frames, rML.predicted_snr, 'r--', 'LineWidth', 1.5);
        legend('Actual SNR', 'LSTM Predicted');
    end
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel Condition'); grid on;

    % SSIM comparison
    subplot(4,2,2);
    plot(frames, rBase.ssim, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.ssim, 'r-s','MarkerSize',3,'LineWidth',1.5);
    if hasAblation
        plot(frames, rMLGaOff.ssim, '--','Color',[0.15 0.65 0.25],'LineWidth',1.5);
        legend('Baseline','ML (GA on)','ML (GA off)', 'Location', 'best');
    else
        legend('Baseline','ML-Enhanced', 'Location', 'best');
    end
    yline(0.9, 'k--', 'Target');
    xlabel('Frame'); ylabel('SSIM'); title('Image Quality (SSIM)');
    ylim([0 1]); grid on;

    % PSNR comparison
    subplot(4,2,3);
    plot(frames, rBase.psnr, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.psnr, 'r-s','MarkerSize',3,'LineWidth',1.5);
    if hasAblation
        plot(frames, rMLGaOff.psnr, '--','Color',[0.15 0.65 0.25],'LineWidth',1.5);
        legend('Baseline','ML (GA on)','ML (GA off)', 'Location', 'best');
    else
        legend('Baseline','ML-Enhanced', 'Location', 'best');
    end
    yline(30, 'k--', 'Target');
    xlabel('Frame'); ylabel('PSNR (dB)'); title('Image Quality (PSNR)'); grid on;

    % MCS adaptation
    subplot(4,2,4);
    plot(frames, rBase.mcs_enh, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.mcs_enh, 'r-s','MarkerSize',3,'LineWidth',1.5);
    legend('Baseline MCS_{enh}','ML MCS_{enh}');
    xlabel('Frame'); ylabel('MCS'); title('Enhancement MCS Adaptation'); grid on;

    % K adaptation
    subplot(4,2,5);
    plot(frames, rBase.K, 'b-o','MarkerSize',3,'LineWidth',1.5); hold on;
    plot(frames, rML.K, 'r-s','MarkerSize',3,'LineWidth',1.5);
    legend('Baseline K','ML K');
    xlabel('Frame'); ylabel('K'); title('Layer Split Adaptation'); ylim([0 64]); grid on;

    % ML quality prediction accuracy
    subplot(4,2,6);
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

    % Tail metrics — SSIM distribution (where ML often separates)
    subplot(4,2,7);
    if hasAblation
        bar([tailB.minSsim, tailM.minSsim, tailMGO.minSsim; tailB.p5Ssim, tailM.p5Ssim, tailMGO.p5Ssim]);
        legend('Baseline', 'ML (GA on)', 'ML (GA off)', 'Location', 'northeast');
    else
        bar([tailB.minSsim, tailM.minSsim; tailB.p5Ssim, tailM.p5Ssim]);
        legend('Baseline', 'ML', 'Location', 'northeast');
    end
    set(gca, 'XTickLabel', {'Min SSIM', 'P5 SSIM'});
    ylabel('SSIM'); ylim([0 1]); title('SSIM tail: min and 5th percentile'); grid on;

    subplot(4,2,8);
    if hasAblation
        bar([tailB.minPsnr, tailM.minPsnr, tailMGO.minPsnr; tailB.p5Psnr, tailM.p5Psnr, tailMGO.p5Psnr; ...
             tailB.enhDropped, tailM.enhDropped, tailMGO.enhDropped; ...
             tailB.framesPsnrBelow, tailM.framesPsnrBelow, tailMGO.framesPsnrBelow]);
        legend('Baseline', 'ML (GA on)', 'ML (GA off)', 'Location', 'northeast');
    else
        bar([tailB.minPsnr, tailM.minPsnr; tailB.p5Psnr, tailM.p5Psnr; ...
             tailB.enhDropped, tailM.enhDropped; tailB.framesPsnrBelow, tailM.framesPsnrBelow]);
        legend('Baseline', 'ML', 'Location', 'northeast');
    end
    set(gca, 'XTickLabel', {'Min PSNR', 'P5 PSNR', 'Enh dropped', sprintf('PSNR<%d', opts.psnrFloorDb)});
    ylabel('dB or frame count'); title('PSNR tail & failure counts'); grid on;

    st2 = sprintf('Stress profile %d frames | Δmin SSIM = %+.4f | ΔP5 SSIM = %+.4f | nFrames enh lost: %d vs %d', ...
        nF, tailM.minSsim - tailB.minSsim, tailM.p5Ssim - tailB.p5Ssim, tailB.enhDropped, tailM.enhDropped);
    if hasAblation
        st2 = sprintf('%s | ML ablation Δmean SSIM = %+.4f', st2, mean(rML.ssim) - mean(rMLGaOff.ssim));
    end
    sgtitle({sprintf('Bold: Baseline vs ML | Δmean SSIM = %+.4f | Δmean PSNR = %+.2f dB', ...
        avgSsimML - avgSsimBase, avgPsnrML - avgPsnrBase), st2}, 'FontWeight', 'bold');

    outPng = fullfile(pwd, 'bold_comparison.png');
    drawnow;
    try
        if ishghandle(figCmp)
            try
                exportgraphics(figCmp, outPng, 'Resolution', 150);
            catch
                saveas(figCmp, outPng);
            end
            fprintf('\n  ✓ Plot saved: %s\n', outPng);
        else
            warning('BoldCompare:NoFigure', 'Comparison figure handle invalid; skip saving PNG.');
        end
    catch ME
        warning('BoldCompare:SaveFigureFailed', 'Could not save comparison plot: %s', ME.message);
    end

    % Save results
    comparison = struct();
    comparison.baseline = rBase;
    comparison.ml = rML;
    comparison.snrProfile = sharedOpts.snrProfile;
    comparison.opts = opts;
    comparison.tail_baseline = tailB;
    comparison.tail_ml = tailM;
    comparison.delta_tail = bold_tail_delta(tailB, tailM);
    if hasAblation
        comparison.ml_ga_off = rMLGaOff;
        comparison.tail_ml_ga_off = tailMGO;
        % Positive deltas => ML+GA better than ML with GA off
        comparison.delta_ga_under_ml = bold_tail_delta(tailMGO, tailM);
    else
        comparison.ml_ga_off = [];
        comparison.tail_ml_ga_off = [];
        comparison.delta_ga_under_ml = [];
    end
    save('bold_comparison.mat', '-struct', 'comparison');
    if hasAblation
        fprintf('  ✓ Data saved: bold_comparison.mat (baseline, ml, ml_ga_off, tail_*, delta_tail, delta_ga_under_ml)\n');
    else
        fprintf('  ✓ Data saved: bold_comparison.mat (includes tail_baseline, tail_ml, delta_tail)\n');
    end

    fprintf('\n  Done! Use these plots in your report.\n\n');
end


function p = bold_ga_report_varying_profile()
% Same nominal SNR vector as generate_ga_report_results_dct scenario "Varying (realistic)".
    p = max(5, [31 29 27 24 21 18 15 12 10  8 ...
        10 12 15 18 22 26 30 33 28 24 ...
        20 16 13 11  9  8  9  9  9  9 ...
         9  9  9 34 38 41 36 30 24 24 ...
        24 24 24 10 10 10 10 10 10 10] - 1);
end


function s = bold_onoff(tf)
    if nargin < 1 || isempty(tf) || ~logical(tf)
        s = 'off';
    else
        s = 'on';
    end
end


function p = bold_build_stress_snr_profile(nFrames)
% Longer profile: fast ramps into 8–15 dB, extended dwell, oscillations, cliffs.
    nFrames = max(32, round(double(nFrames)));
    p = zeros(1, nFrames);

    n1 = max(6, round(0.06 * nFrames));   % short high-SNR lead-in
    n2 = max(8, round(0.12 * nFrames));  % fast ramp down
    n3 = max(round(0.45 * nFrames), 20); % long stress band 8–15
    n4 = max(8, round(0.12 * nFrames));  % recovery ramp
    n5 = nFrames - n1 - n2 - n3 - n4;
    if n5 < 0
        scale = nFrames / (n1 + n2 + n3 + n4);
        n1 = max(4, round(n1 * scale));
        n2 = max(6, round(n2 * scale));
        n3 = max(15, round(n3 * scale));
        n4 = max(6, round(n4 * scale));
        n5 = nFrames - n1 - n2 - n3 - n4;
    end
    while n5 < 0 && n3 > 8
        n3 = n3 - 1;
        n5 = nFrames - n1 - n2 - n3 - n4;
    end
    while n5 < 0 && n2 > 4
        n2 = n2 - 1;
        n5 = nFrames - n1 - n2 - n3 - n4;
    end
    while n5 < 0 && n4 > 0
        n4 = n4 - 1;
        n5 = nFrames - n1 - n2 - n3 - n4;
    end
    while n5 < 0 && n1 > 2
        n1 = n1 - 1;
        n5 = nFrames - n1 - n2 - n3 - n4;
    end
    if n5 < 0
        n1 = 1; n2 = 1; n3 = max(1, nFrames - 4); n4 = 1; n5 = nFrames - n1 - n2 - n3 - n4;
    end

    i0 = 0;
    % A: mild high SNR
    p(i0 + (1:n1)) = linspace(34, 27, n1);
    i0 = i0 + n1;
    % B: steep descent into stress band (~1.2 dB / frame)
    p(i0 + (1:n2)) = linspace(27, 10, n2);
    i0 = i0 + n2;
    % C: dwell — fast wiggles mostly 8–15 dB
    stressPat = [15 12 9 8 10 14 11 8 13 9 8 12 15 10 8 11 14 9 8 10];
    for k = 1:n3
        p(i0 + k) = stressPat(mod(k - 1, numel(stressPat)) + 1);
    end
    i0 = i0 + n3;
    % D: climb then cliff
    if n4 > 0
        p(i0 + (1:n4)) = linspace(11, 30, n4);
        i0 = i0 + n4;
    end
    % E: remainder — low plateau + spikes (MCS / enh stress)
    if n5 > 0
        tailPat = [9 8 9 8 7 10 28 12 9 8 15 32 10 9 8 11];
        for k = 1:n5
            p(i0 + k) = tailPat(mod(k - 1, numel(tailPat)) + 1);
        end
    end

    p = max(5, min(45, round(p)));
end


function v = bold_percentile(x, p)
% Linear interpolation on sorted sample (p in [0, 100]); no Statistics TBX.
    x = sort(double(x(:)));
    n = numel(x);
    if n < 1, v = NaN; return; end
    if n == 1, v = x(1); return; end
    r = (p / 100) * (n - 1) + 1;
    lo = floor(r);
    hi = min(n, ceil(r));
    if lo == hi
        v = x(lo);
    else
        v = x(lo) + (r - lo) * (x(hi) - x(lo));
    end
end


function t = bold_tail_metrics(r, psnrFloorDb)
    t = struct();
    t.minSsim = min(r.ssim(:));
    t.p5Ssim = bold_percentile(r.ssim(:), 5);
    t.minPsnr = min(r.psnr(:));
    t.p5Psnr = bold_percentile(r.psnr(:), 5);
    if isfield(r, 'enh_valid')
        t.enhDropped = sum(~r.enh_valid(:));
    else
        t.enhDropped = NaN;
    end
    t.framesPsnrBelow = sum(r.psnr(:) < psnrFloorDb);
    t.nFrames = r.nFrames;
end


function d = bold_tail_delta(tailB, tailM)
    d = struct();
    d.deltaMinSsim = tailM.minSsim - tailB.minSsim;
    d.deltaP5Ssim = tailM.p5Ssim - tailB.p5Ssim;
    d.deltaMinPsnr = tailM.minPsnr - tailB.minPsnr;
    d.deltaP5Psnr = tailM.p5Psnr - tailB.p5Psnr;
    d.deltaEnhDropped = tailM.enhDropped - tailB.enhDropped;
    d.deltaFramesPsnrBelow = tailM.framesPsnrBelow - tailB.framesPsnrBelow;
end
