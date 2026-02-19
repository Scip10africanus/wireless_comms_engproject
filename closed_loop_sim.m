function results = closed_loop_sim(opts)
%% CLOSED_LOOP_SIM  Multi-frame TX -> Channel -> RX with adaptive feedback
% Runs the full closed loop: PolicyController -> TX -> Channel -> RX -> Feedback -> CSV -> loop
%
% Usage:
%   closed_loop_sim()
%   closed_loop_sim(struct('nFrames',30, 'snrProfile',[25 20 15 10 15 20 25]))

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),       opts.nFrames = 20; end
    if ~isfield(opts,'useGaOptimizer'),opts.useGaOptimizer = true; end
    if ~isfield(opts,'usePredictor'),  opts.usePredictor = true; end
    if ~isfield(opts,'channelType'),   opts.channelType = 'AWGN'; end
    if ~isfield(opts,'policyCsvPath'), opts.policyCsvPath = fullfile(pwd,'policy.csv'); end
    if ~isfield(opts,'verbose'),       opts.verbose = true; end
    if ~isfield(opts,'useOctaveMode'), opts.useOctaveMode = false; end

    % SNR profile - simulates varying channel
    if ~isfield(opts,'snrProfile')
        opts.snrProfile = [30 28 25 22 18 15 12 10 8 10 12 15 18 22 25 28 30 30 28 25];
    end
    if length(opts.snrProfile) < opts.nFrames
        opts.snrProfile = [opts.snrProfile, opts.snrProfile(end)*ones(1, opts.nFrames-length(opts.snrProfile))];
    end
    opts.snrProfile = opts.snrProfile(1:opts.nFrames);

    %% paths
    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    addpath(fullfile(thisDir,'Tx','src'));
    addpath(fullfile(thisDir,'rx'));
    addpath(fullfile(thisDir,'channel'));
    addpath(fullfile(thisDir,'common'));
    addpath(fullfile(thisDir,'config'));
    addpath(fullfile(thisDir,'ml_addons'));

    %% test image (gradient + circle)
    if isfield(opts,'testImage')
        testImg = opts.testImage;
    else
        [X,Y] = meshgrid(1:64,1:64);
        testImg = uint8(X*255/64 + (sqrt((X-32).^2+(Y-32).^2)<20)*100);
    end

    %% init PolicyController
    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end
    pc = PolicyController(opts.policyCsvPath);
    pc.Verbose = opts.verbose;

    chHistory = struct('snr',[],'fading',[]);

    rxSettings = struct();
    rxSettings.originalImage = testImg;
    rxSettings.policyCsvPath = opts.policyCsvPath;
    rxSettings.useGaOptimizer = opts.useGaOptimizer;
    rxSettings.gaSettings = struct('popSize',30,'generations',10,'verbose',false);
    rxSettings.useChannelPredictor = opts.usePredictor;
    rxSettings.predictorParams = struct('predictionHorizon',3,'windowSize',8);
    rxSettings.verbose = opts.verbose;
    rxSettings.useOctaveMode = opts.useOctaveMode;

    %% header
    fprintf('\n================================================================\n');
    fprintf('  CLOSED-LOOP SIM | %dx%d | %d frames | %s\n', ...
        size(testImg,1), size(testImg,2), opts.nFrames, opts.channelType);
    fprintf('================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %6s\n', 'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR');
    fprintf('%s\n', repmat('-',1,52));

    %% logging
    results = struct();
    results.nFrames = opts.nFrames;
    results.snrProfile = opts.snrProfile;
    results.ssim     = zeros(1,opts.nFrames);
    results.psnr     = zeros(1,opts.nFrames);
    results.K        = zeros(1,opts.nFrames);
    results.mcs_base = zeros(1,opts.nFrames);
    results.mcs_enh  = zeros(1,opts.nFrames);
    results.success  = true(1,opts.nFrames);

    %% main loop
    for f = 1:opts.nFrames
        frameId = mod(f-1, 256);
        snr = opts.snrProfile(f);

        try
            [txWave, txInfo] = two_layer_tx_pipeline(testImg, pc, frameId);
            p = txInfo.policy;
            results.K(f) = double(p.K);
            results.mcs_base(f) = double(p.mcs_base);
            results.mcs_enh(f)  = double(p.mcs_enh);

            [rxWave, chInfo] = channel_model(txWave, struct('channelType',opts.channelType,'SNR_dB',snr));

            chHistory.snr = [chHistory.snr, snr];
            rxSettings.channelHistory = chHistory;

            [~, fb, rxRes] = two_layer_rx_v2(rxWave, txInfo, rxSettings);

            if fb.hasQualityMetrics
                results.ssim(f) = fb.ssim;
                results.psnr(f) = fb.psnr;
            end
            results.success(f) = rxRes.success;

            fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %6.2f\n', ...
                f, snr, double(p.K), double(p.mcs_base), double(p.mcs_enh), ...
                results.ssim(f), results.psnr(f));
        catch ME
            results.success(f) = false;
            fprintf('%5d | %6.1f | ERROR: %s\n', f, snr, ME.message);
        end
    end

    %% summary
    ok = results.success & results.ssim > 0;
    fprintf('\n================================================================\n');
    if any(ok)
        fprintf('  Avg SSIM=%.4f  Avg PSNR=%.2f dB\n', mean(results.ssim(ok)), mean(results.psnr(ok)));
    end
    fprintf('  Success: %d/%d frames | K range: %d-%d | MCS enh: %d-%d\n', ...
        sum(results.success), opts.nFrames, min(results.K), max(results.K), ...
        min(results.mcs_enh), max(results.mcs_enh));
    fprintf('================================================================\n');

    %% plot
    try
        plot_results(results, opts);
    catch
    end
end

function plot_results(r, opts)
    figure('Position',[100 100 900 600],'Name','Closed-Loop Results');
    frames = 1:r.nFrames;

    subplot(2,2,1);
    plot(frames, opts.snrProfile, 'b-o','MarkerSize',3);
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel SNR'); grid on;

    subplot(2,2,2);
    plot(frames, r.ssim, 'g-o','MarkerSize',3); hold on;
    yline(0.9,'r--','Target'); hold off;
    xlabel('Frame'); ylabel('SSIM'); title('Image Quality'); ylim([0 1]); grid on;

    subplot(2,2,3);
    plot(frames, r.mcs_base,'b-s','MarkerSize',3); hold on;
    plot(frames, r.mcs_enh,'r-o','MarkerSize',3); hold off;
    legend('Base','Enh'); xlabel('Frame'); ylabel('MCS'); title('MCS Adaptation'); grid on;

    subplot(2,2,4);
    plot(frames, r.K, 'k-d','MarkerSize',3);
    xlabel('Frame'); ylabel('K'); title('Layer Split'); ylim([0 64]); grid on;

    try saveas(gcf,'closed_loop_results.png'); end
end
