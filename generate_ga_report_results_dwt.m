function generate_ga_report_results()
% GENERATE_GA_REPORT_RESULTS  One-click script to produce GA results for report
%
% HOW TO USE (in MATLAB):
%   1. cd('/Users/angelailieva/Downloads/wireless_comms_project')
%   2. >> generate_ga_report_results
%   3. Wait ~3-5 minutes. Figures saved to results/ga_report_dwt/

    %% ---- Setup ----
    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end

    addpath(fullfile(thisDir, 'common'));
    addpath(fullfile(thisDir, 'tx', 'src'));
    addpath(fullfile(thisDir, 'rx'));
    addpath(fullfile(thisDir, 'channel'));
    addpath(fullfile(thisDir, 'ml_addons'));

    outDir = fullfile(thisDir, 'results', 'ga_report_dwt');
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
                r = run_dwt_closed_loop_benchmark(opts);
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

    fig = figure('Position',[50 50 1400 950],'Name','SSIM Comparison');
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
        ylim([0 1.05]); grid on;
        if s == 1
            legend([{configs.name},{'Target (0.9)'}],'Location','southwest','FontSize',7);
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
    legend({scenarios.name},'Location','best','FontSize',6);
    grid on; hold off;
    sgtitle('SSIM: Rules Only vs Rules+GA','FontSize',13,'FontWeight','bold');
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, '-dpng', '-r150', fullfile(outDir,'fig1_ssim_comparison.png'));
    fprintf('Saved: fig1_ssim_comparison.png\n');
end


function plot_ga_convergence(gaConv, outDir)
    fig = figure('Position',[100 100 1050 550],'Name','GA Convergence');
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
    legend(legendEntries,'Location','southeast','FontSize',8);
    grid on; hold off;

    subplot(1,2,2);
    bar([gaConv.bestK; gaConv.bestQb; gaConv.bestQe]');
    set(gca,'XTickLabel',arrayfun(@(x) sprintf('%d dB',x), gaConv.snrPoints,'UniformOutput',false));
    xlabel('Channel SNR'); ylabel('Parameter Value');
    title('GA Optimal Parameters by SNR','FontSize',11);
    legend({'K','Qb idx','Qe idx'},'Location','northwest');
    grid on;
    sgtitle('GA Convergence & Optimal Parameters','FontSize',13,'FontWeight','bold');
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, '-dpng', '-r150', fullfile(outDir,'fig2_ga_convergence.png'));
    fprintf('Saved: fig2_ga_convergence.png\n');
end


function plot_parameter_adaptation(allResults, configs, scenarios, outDir)
    sIdx = find(strcmp({scenarios.tag},'varying'),1);
    if isempty(sIdx), sIdx = 1; end
    nConf = numel(configs);
    colors = {[0.2 0.4 0.8],[0.9 0.3 0.2]};

    fig = figure('Position',[80 80 1300 900],'Name','Parameter Adaptation');

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
    legend([{configs.name},{'Target (0.9)'}],'Location','southwest','FontSize',7);
    xlabel('Frame'); ylabel('SSIM'); title('SSIM'); ylim([0 1.05]); grid on; hold off;

    subplot(3,2,3); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.K,'-','Color',colors{c},'LineWidth',1.5);
    end
    legend({configs.name},'Location','best','FontSize',7);
    xlabel('Frame'); ylabel('K'); title('Layer Split K'); grid on; hold off;

    subplot(3,2,4); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.mcs_base,'-','Color',colors{c},'LineWidth',1.2);
        plot(1:r.nFrames, r.mcs_enh,'--','Color',colors{c},'LineWidth',1.2);
    end
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
    legend({configs.name},'Location','best','FontSize',7);
    xlabel('Frame'); ylabel('Block OK Ratio'); title('Block Success Rate');
    ylim([0 1.05]); grid on; hold off;

    subplot(3,2,6); hold on;
    for c = 1:nConf
        r = allResults{c,sIdx}; if isempty(r), continue; end
        plot(1:r.nFrames, r.psnr,'-','Color',colors{c},'LineWidth',1.5);
    end
    yline(30,'k--');
    legend([{configs.name},{'Target (30 dB)'}],'Location','southwest','FontSize',7);
    xlabel('Frame'); ylabel('PSNR (dB)'); title('PSNR'); grid on; hold off;

    sgtitle(sprintf('Parameter Adaptation: %s',scenarios(sIdx).name), ...
        'FontSize',13,'FontWeight','bold');
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, '-dpng', '-r150', fullfile(outDir,'fig3_parameter_adaptation.png'));
    fprintf('Saved: fig3_parameter_adaptation.png\n');
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

    fig = figure('Position',[100 100 1300 620],'Name','Summary');
    colors = {[0.2 0.4 0.8],[0.9 0.3 0.2]};

    ax1 = subplot(1,2,1);
    b1 = bar(avgSSIM');
    for c = 1:nConf, b1(c).FaceColor = colors{c}; end
    set(ax1,'XTickLabel',scenLabels,'XTickLabelRotation',25);
    xlabel('Scenario'); ylabel('Avg SSIM');
    title('Average SSIM','FontSize',11);
    legend({configs.name},'Location','best','FontSize',8);
    ylim([0 1]); grid on;
    set(ax1, 'Position', [0.06 0.18 0.40 0.70]);

    ax2 = subplot(1,2,2);
    b2 = bar(avgPSNR');
    for c = 1:nConf, b2(c).FaceColor = colors{c}; end
    set(ax2,'XTickLabel',scenLabels,'XTickLabelRotation',25);
    xlabel('Scenario'); ylabel('Avg PSNR (dB)');
    title('Average PSNR','FontSize',11);
    legend({configs.name},'Location','best','FontSize',8);
    grid on;
    set(ax2, 'Position', [0.56 0.18 0.40 0.70]);

    sgtitle('Summary: GA Impact on Image Quality','FontSize',13,'FontWeight','bold');
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, '-dpng', '-r150', fullfile(outDir,'fig4_summary_bars.png'));
    fprintf('Saved: fig4_summary_bars.png\n');
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