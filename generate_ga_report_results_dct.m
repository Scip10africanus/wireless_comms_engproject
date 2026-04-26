function generate_ga_report_results_dct()
% GENERATE_GA_REPORT_RESULTS_DCT  One-click script to produce GA results (DCT) for report
%
% HOW TO USE (in MATLAB):
%   1. cd to this repository root (folder containing this file)
%   2. >> generate_ga_report_results_dct
%   3. Wait ~3-5 minutes. Figures saved to results/ga_report/

    %% ---- Setup ----
    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end

    addpath(fullfile(thisDir, 'common'));
    addpath(fullfile(thisDir, 'tx', 'src'));
    addpath(fullfile(thisDir, 'rx'));
    addpath(fullfile(thisDir, 'channel'));
    addpath(fullfile(thisDir, 'ml_addons'));

    assert(~isempty(which('ga_parameter_optimizer')), ...
        'ga_parameter_optimizer not found. Current folder must be the WCE repo root.');
    assert(~isempty(which('channel_predictor')), ...
        'channel_predictor not found under ml_addons.');

    outDir = fullfile(thisDir, 'results', 'ga_report');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    %% ---- SNR scenarios ----
    nFrames = 50;
    scenarios = struct();

    scenarios(1).name = 'Varying (realistic)';
    scenarios(1).tag  = 'varying';
    scenarios(1).snr  = [31 29 27 24 21 18 15 12 10  8 ...
                         10 12 15 18 22 26 30 33 28 24 ...
                         20 16 13 11  9  8  9  9  9  9 ...
                          9  9  9 34 38 41 36 30 24 24 ...
                         24 24 24 10 10 10 10 10 10 10];

    scenarios(2).name = 'Fixed High SNR (30 dB)';
    scenarios(2).tag  = 'high_snr';
    scenarios(2).snr  = 30 * ones(1, nFrames);

    scenarios(3).name = 'Fixed Low SNR (10 dB)';
    scenarios(3).tag  = 'low_snr';
    scenarios(3).snr  = 10 * ones(1, nFrames);

    scenarios(4).name = 'Rapid Swings';
    scenarios(4).tag  = 'rapid_swing';
    scenarios(4).snr  = repmat([35 10 35 10 25], 1, 10);

    scenarios(5).name = 'Slow Degradation (30->8 dB)';
    scenarios(5).tag  = 'slow_degrade';
    scenarios(5).snr  = linspace(30, 8, nFrames);

    %% ---- Configurations ----
    configs = struct();

    configs(1).name  = 'Rules Only';
    configs(1).tag   = 'rules_only';
    configs(1).useGA = false;

    configs(2).name  = 'Rules + GA';
    configs(2).tag   = 'rules_ga';
    configs(2).useGA = true;

    %% ---- Run all combinations ----
    nScen = numel(scenarios);
    nConf = numel(configs);
    allResults = cell(nConf, nScen);
    totalRuns = nScen * nConf;
    runIdx = 0;

    for c = 1:nConf
        for s = 1:nScen
            runIdx = runIdx + 1;
            fprintf('\n>>> [%d/%d] %s | %s\n', ...
                runIdx, totalRuns, configs(c).name, scenarios(s).name);

            opts = struct();
            opts.nFrames        = nFrames;
            opts.snrProfile     = scenarios(s).snr;
            opts.useGaOptimizer = configs(c).useGA;
            opts.verbose        = false;
            opts.doPlot         = false;
            opts.policyCsvPath  = fullfile(outDir, ...
                sprintf('policy_%s_%s.csv', configs(c).tag, scenarios(s).tag));

            try
                r = run_dct_closed_loop_benchmark(opts);
                allResults{c, s} = r;
            catch ME
                fprintf('  *** ERROR: %s\n', ME.message);
                allResults{c, s} = struct( ...
                    'ssim',    zeros(1,nFrames), ...
                    'psnr',    zeros(1,nFrames), ...
                    'K',       zeros(1,nFrames), ...
                    'mcs_base',zeros(1,nFrames), ...
                    'mcs_enh', zeros(1,nFrames), ...
                    'success', false(1,nFrames), ...
                    'nFrames', nFrames, ...
                    'blkOK',   zeros(1,nFrames), ...
                    'nBlocks', ones(1,nFrames),  ...
                    'snr',     scenarios(s).snr(1:nFrames), ...
                    'error',   ME.message);
            end
        end
    end

    %% ---- GA convergence analysis ----
    fprintf('\n>>> Running GA convergence analysis...\n');
    snrTestPoints = [8 15 20 25 30 38];
    gaConv.snrPoints  = snrTestPoints;
    gaConv.fitHistory = zeros(numel(snrTestPoints), 15);
    gaConv.bestK  = zeros(1, numel(snrTestPoints));
    gaConv.bestQb = zeros(1, numel(snrTestPoints));
    gaConv.bestQe = zeros(1, numel(snrTestPoints));

    for i = 1:numel(snrTestPoints)
        chCond    = struct('snr_dB', snrTestPoints(i), 'prevSSIM', 0.95);
        curParams = struct('K',20,'Qb_idx',0,'Qe_idx',0,'mcs_base',2,'mcs_enh',2);
        gaSettings = struct('popSize',40,'generations',15,'verbose',false,'eliteCount',4);

        [optP, gaM] = ga_parameter_optimizer(chCond, curParams, gaSettings);
        gaConv.fitHistory(i,:) = gaM.fitnessHistory(:)';
        gaConv.bestK(i)  = double(optP.K);
        gaConv.bestQb(i) = double(optP.Qb_idx);
        gaConv.bestQe(i) = double(optP.Qe_idx);

        fprintf('  SNR=%2d dB -> K=%2d Qb=%d Qe=%d fitness=%.3f\n', ...
            snrTestPoints(i), gaConv.bestK(i), gaConv.bestQb(i), ...
            gaConv.bestQe(i), gaM.finalFitness);
    end

    %% ---- Save + plot ----
    save(fullfile(outDir,'ga_report_data.mat'), ...
         'allResults','configs','scenarios','gaConv','nFrames');
    fprintf('\nRaw data saved.\n');

    write_summary_csv(fullfile(outDir,'ga_summary_table.csv'), allResults, configs, scenarios);
    plot_ssim_comparison(allResults, configs, scenarios, outDir);
    plot_ga_convergence(gaConv, outDir);
    plot_parameter_adaptation(allResults, configs, scenarios, outDir);
    plot_summary_bars(allResults, configs, scenarios, outDir);

    fprintf('\n=== ALL DONE! Results in: %s ===\n', outDir);
end


%% ======================================================================
%  PLOTTING FUNCTIONS
%% ======================================================================

function plot_ssim_comparison(allResults, configs, scenarios, outDir)
    nScen = numel(scenarios);
    nConf = numel(configs);
    colors = {[0.2 0.4 0.8], [0.9 0.3 0.2]};

    fig = new_report_figure('SSIM Comparison', [50 50 1400 950]);
    for s = 1:nScen
        subplot(2,3,s); hold on;
        for c = 1:nConf
            r = allResults{c,s};
            if isempty(r), continue; end
            plot(1:r.nFrames, r.ssim, '-','Color',colors{c},'LineWidth',1.5);
        end
        yline(0.9,'k--','LineWidth',0.8);
        xlabel('Frame'); ylabel('SSIM');
        title(scenarios(s).name,'FontSize',10);
        ylim([0 1.08]); grid on;
        if s == 1
            legend([{configs.name},{'Target (0.9)'}], 'Location', 'northeast', ...
                'FontSize', 8, 'Box', 'off');
        end
        hold off;
    end
    subplot(2,3,6); hold on;
    profileColors = lines(nScen);
    for s = 1:nScen
        plot(1:numel(scenarios(s).snr), scenarios(s).snr, '-', ...
            'Color',profileColors(s,:),'LineWidth',1.2);
    end
    xlabel('Frame'); ylabel('SNR (dB)');
    title('SNR Profiles Used','FontSize',10);
    legend({scenarios.name}, 'Location', 'northeast', 'FontSize', 7, 'Box', 'off');
    grid on; hold off;
    sgtitle('SSIM: Rules Only vs Rules+GA','FontSize',13,'FontWeight','bold');
    apply_report_style(fig);
    save_report_pdf(fig, fullfile(outDir, 'fig1_ssim_comparison'));
    fprintf('Saved: fig1_ssim_comparison.pdf\n');
end


function plot_ga_convergence(gaConv, outDir)
    fig = new_report_figure('GA Convergence', [100 100 1050 550]);
    subplot(1,2,1); hold on;
    colors = parula(numel(gaConv.snrPoints));
    legendEntries = cell(1,numel(gaConv.snrPoints));
    for i = 1:numel(gaConv.snrPoints)
        plot(1:size(gaConv.fitHistory,2), gaConv.fitHistory(i,:), ...
            '-o','Color',colors(i,:),'LineWidth',1.5,'MarkerSize',4);
        legendEntries{i} = sprintf('SNR = %d dB', gaConv.snrPoints(i));
    end
    xlabel('Generation'); ylabel('Best Fitness');
    title('GA Convergence by SNR','FontSize',11);
    legend(legendEntries, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    grid on; hold off;

    subplot(1,2,2);
    bar([gaConv.bestK; gaConv.bestQb; gaConv.bestQe]');
    set(gca,'XTickLabel',arrayfun(@(x) sprintf('%d dB',x), gaConv.snrPoints,'UniformOutput',false));
    xlabel('Channel SNR'); ylabel('Parameter Value');
    title('GA Optimal Parameters by SNR','FontSize',11);
    legend({'K','Qb idx','Qe idx'}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    grid on;
    sgtitle('GA Convergence & Optimal Parameters','FontSize',13,'FontWeight','bold');
    apply_report_style(fig);
    save_report_pdf(fig, fullfile(outDir, 'fig2_ga_convergence'));
    fprintf('Saved: fig2_ga_convergence.pdf\n');
end


function plot_parameter_adaptation(allResults, configs, scenarios, outDir)
    sIdx = find(strcmp({scenarios.tag},'varying'),1);
    if isempty(sIdx), sIdx = 1; end
    nConf = numel(configs);
    colors = {[0.2 0.4 0.8],[0.9 0.3 0.2]};

    fig = new_report_figure('Parameter Adaptation', [80 80 1300 900]);

    subplot(3,2,1);
    r1 = allResults{1,sIdx};
    plot(1:r1.nFrames, scenarios(sIdx).snr(1:r1.nFrames),'b-','LineWidth',1.5);
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel SNR'); grid on;

    subplot(3,2,2); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.ssim,'-','Color',colors{c},'LineWidth',1.5);
    end
    yline(0.9,'k--');
    legend([{configs.name},{'Target (0.9)'}], 'Location', 'northeast', ...
        'FontSize', 8, 'Box', 'off');
    xlabel('Frame'); ylabel('SSIM'); title('SSIM'); ylim([0 1.08]); grid on; hold off;

    subplot(3,2,3); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.K,'-','Color',colors{c},'LineWidth',1.5);
    end
    legend({configs.name}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    xlabel('Frame'); ylabel('K'); title('Layer Split K'); grid on; hold off;

    subplot(3,2,4); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.mcs_base,'-','Color',colors{c},'LineWidth',1.2);
        plot(1:r.nFrames, r.mcs_enh,'--','Color',colors{c},'LineWidth',1.2);
    end
    legend({configs.name}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    xlabel('Frame'); ylabel('MCS'); title('MCS (solid=base, dashed=enh)'); grid on; hold off;

    subplot(3,2,5); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        if isfield(r,'blkOK') && isfield(r,'nBlocks')
            blkRatio = r.blkOK ./ max(1,r.nBlocks);
        else
            blkRatio = double(r.success);
        end
        plot(1:r.nFrames, blkRatio,'-','Color',colors{c},'LineWidth',1.5);
    end
    legend({configs.name}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    xlabel('Frame'); ylabel('Block OK Ratio'); title('Block Success Rate');
    ylim([0 1.08]); grid on; hold off;

    subplot(3,2,6); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.psnr,'-','Color',colors{c},'LineWidth',1.5);
    end
    yline(30,'k--');
    legend([{configs.name},{'Target (30 dB)'}], 'Location', 'northeast', ...
        'FontSize', 8, 'Box', 'off');
    xlabel('Frame'); ylabel('PSNR (dB)'); title('PSNR'); grid on; hold off;

    sgtitle(sprintf('Parameter Adaptation: %s',scenarios(sIdx).name), ...
        'FontSize',13,'FontWeight','bold');
    apply_report_style(fig);
    save_report_pdf(fig, fullfile(outDir, 'fig3_parameter_adaptation'));
    fprintf('Saved: fig3_parameter_adaptation.pdf\n');
end


function plot_summary_bars(allResults, configs, scenarios, outDir)
    nScen = numel(scenarios);
    nConf = numel(configs);
    avgSSIM = zeros(nConf,nScen);
    avgPSNR = zeros(nConf,nScen);

    for c = 1:nConf
        for s = 1:nScen
            r = allResults{c,s}; if isempty(r), continue; end
            ok = r.ssim > 0;
            if any(ok)
                avgSSIM(c,s) = mean(r.ssim(ok));
                avgPSNR(c,s) = mean(r.psnr(ok));
            end
        end
    end

    scenLabels = strrep({scenarios.tag}, '_', ' ');

    fig = new_report_figure('Summary', [100 100 1300 620]);
    colors = {[0.2 0.4 0.8],[0.9 0.3 0.2]};

    ax1 = subplot(1,2,1);
    b1 = bar(avgSSIM');
    for c = 1:nConf, b1(c).FaceColor = colors{c}; end
    set(ax1,'XTickLabel',scenLabels,'XTickLabelRotation',25);
    xlabel('Scenario'); ylabel('Avg SSIM');
    title('Average SSIM','FontSize',11);
    legend({configs.name}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    ylim([0 1.08]); grid on;
    set(ax1, 'Position', [0.08 0.20 0.38 0.65]);

    ax2 = subplot(1,2,2);
    b2 = bar(avgPSNR');
    for c = 1:nConf, b2(c).FaceColor = colors{c}; end
    set(ax2,'XTickLabel',scenLabels,'XTickLabelRotation',25);
    xlabel('Scenario'); ylabel('Avg PSNR (dB)');
    title('Average PSNR','FontSize',11);
    legend({configs.name}, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    grid on;
    set(ax2, 'Position', [0.54 0.20 0.38 0.65]);

    sgtitle('Summary: GA Impact on Image Quality','FontSize',13,'FontWeight','bold');
    apply_report_style(fig);
    save_report_pdf(fig, fullfile(outDir, 'fig4_summary_bars'));
    fprintf('Saved: fig4_summary_bars.pdf\n');
end


function fig = new_report_figure(name, pos)
%NEW_REPORT_FIGURE  White figure + Times New Roman defaults for report plots.
    fig = figure('Name', name, 'Position', pos, ...
        'Color', 'w', ...
        'InvertHardcopy', 'off', ...
        'DefaultAxesFontName', 'Times New Roman', ...
        'DefaultTextFontName', 'Times New Roman');
end


function apply_report_style(fig)
%APPLY_REPORT_STYLE  White axes, Times labels, black titles/labels, light grid.
    if ~isgraphics(fig), return; end
    set(fig, 'Color', 'w');
    fn = 'Times New Roman';
    axs = findall(fig, 'Type', 'axes');
    for i = 1:numel(axs)
        a = axs(i);
        set(a, 'FontName', fn, 'Color', 'w', ...
            'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', ...
            'GridColor', [0.88 0.88 0.88], 'MinorGridColor', [0.92 0.92 0.92]);
        try, set(a, 'TickLabelColor', 'k'); catch, end %#ok<CTCH>
        try
            set(a.Title, 'FontName', fn, 'Color', 'k');
        catch, end %#ok<CTCH>
        try
            set(a.XLabel, 'FontName', fn, 'Color', 'k');
            set(a.YLabel, 'FontName', fn, 'Color', 'k');
            if isprop(a, 'ZLabel') && ~isempty(a.ZLabel.String)
                set(a.ZLabel, 'FontName', fn, 'Color', 'k');
            end
        catch, end %#ok<CTCH>
        if isprop(a, 'Subtitle')
            try
                set(a.Subtitle, 'FontName', fn, 'Color', 'k');
            catch, end %#ok<CTCH>
        end
    end
    for typ = {'text', 'legend'}
        h = findall(fig, 'Type', typ{1});
        for j = 1:numel(h)
            try
                set(h(j), 'FontName', fn, 'Color', 'k');
            catch, end %#ok<CTCH>
        end
    end
    st = findall(fig, 'Type', 'subplottext');
    for i = 1:numel(st)
        try
            set(st(i), 'FontName', fn, 'Color', 'k');
        catch, end %#ok<CTCH>
    end
end


function save_report_pdf(fig, fileBase)
%SAVE_REPORT_PDF  Vector PDF, white background, margins to reduce clipping.
    fileBase = char(fileBase);
    pdfPath = [fileBase '.pdf'];
    if ~isgraphics(fig)
        warning('save_report_pdf:InvalidFigure', 'Figure invalid; skipped: %s', pdfPath);
        return;
    end
    hide_toolbars_for_export(fig);
    drawnow nocallbacks;
    [outDir, ~, ~] = fileparts(pdfPath);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    oldppm = fig.PaperPositionMode;
    oldpu = fig.PaperUnits;
    oldpp = fig.PaperPosition;
    oldps = [];
    if isprop(fig, 'PaperSize')
        oldps = fig.PaperSize;
    end

    fig.Units = 'inches';
    fp = fig.Position;
    margin = 0.35;
    pw = fp(3) + 2*margin;
    ph = fp(4) + 2*margin;
    fig.PaperUnits = 'inches';
    fig.PaperSize = [pw ph];
    fig.PaperPosition = [margin margin fp(3) fp(4)];
    fig.PaperPositionMode = 'manual';

    lastErr = '';
    wrote = false;
    if exist('exportgraphics', 'file') == 2
        try
            exportgraphics(fig, pdfPath, 'ContentType', 'vector', ...
                'BackgroundColor', 'white', 'Padding', 'tight');
            wrote = true;
        catch ME
            lastErr = ME.message;
        end
    end
    if ~wrote
        try
            print(fig, pdfPath, '-dpdf', '-painters');
            wrote = true;
        catch ME
            lastErr = ME.message;
        end
    end

    fig.PaperPositionMode = oldppm;
    fig.PaperUnits = oldpu;
    fig.PaperPosition = oldpp;
    if ~isempty(oldps)
        fig.PaperSize = oldps;
    end

    if ~wrote
        warning('save_report_pdf:Failed', 'Could not save PDF (%s): %s', pdfPath, lastErr);
    end
end


function hide_toolbars_for_export(fig)
%HIDE_TOOLBARS_FOR_EXPORT  Stops exportgraphics embedding the axes toolbar in PDF/PNG.
    try
        if isprop(fig, 'Toolbar') && ~isempty(fig.Toolbar) && isgraphics(fig.Toolbar) ...
                && isprop(fig.Toolbar, 'Visible')
            fig.Toolbar.Visible = 'off';
        end
    catch, end %#ok<CTCH>
    axs = findall(fig, 'Type', 'axes');
    for k = 1:numel(axs)
        try
            ax = axs(k);
            if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar) && isgraphics(ax.Toolbar) ...
                    && isprop(ax.Toolbar, 'Visible')
                ax.Toolbar.Visible = 'off';
            end
        catch, end %#ok<CTCH>
    end
end


function write_summary_csv(csvPath, allResults, configs, scenarios)
    fid = fopen(csvPath,'w');
    fprintf(fid,'Configuration,Scenario,Avg SSIM,Avg PSNR (dB),Min SSIM,Max SSIM,Success Rate (%%)\n');
    for c = 1:numel(configs)
        for s = 1:numel(scenarios)
            r = allResults{c,s};
            if isempty(r)
                fprintf(fid,'%s,%s,N/A,N/A,N/A,N/A,N/A\n',configs(c).name,scenarios(s).name);
                continue;
            end
            ok = r.ssim > 0;
            avgS = 0; avgP = 0;
            if any(ok), avgS = mean(r.ssim(ok)); avgP = mean(r.psnr(ok)); end
            succRate = 100 * sum(r.success) / r.nFrames;
            fprintf(fid,'%s,%s,%.4f,%.2f,%.4f,%.4f,%.1f\n', ...
                configs(c).name, scenarios(s).name, ...
                avgS, avgP, min(r.ssim), max(r.ssim), succRate);
        end
    end
    fclose(fid);
    fprintf('Saved: %s\n', csvPath);
end