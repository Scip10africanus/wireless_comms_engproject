function results = bold_closed_loop_ml(opts)
%% BOLD_CLOSED_LOOP_ML  Closed-loop simulation with all 3 ML models
% Requires: Image Processing Toolbox, WLAN Toolbox
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),        opts.nFrames = 50; end
    if ~isfield(opts,'useML'),          opts.useML = true; end
    if ~isfield(opts,'useGaOptimizer'), opts.useGaOptimizer = true; end
    if ~isfield(opts,'channelType'),    opts.channelType = 'RICIAN'; end
    if ~isfield(opts,'policyCsvPath'),  opts.policyCsvPath = fullfile(pwd,'policy.csv'); end
    if ~isfield(opts,'verbose'),        opts.verbose = true; end

    if ~isfield(opts,'snrProfile')
        opts.snrProfile = [ ...
          31 29 27 24 21 18 15 12 10  8 ...
          10 12 15 18 22 26 30 33 28 24 ...
          20 16 13 11  9  8  9  9  9  9 ...
           9  9  9 34 38 41 36 30 24 24 ...
          24 24 24 10 10 10 10 10 10 10];
    end
    if length(opts.snrProfile) < opts.nFrames
        opts.snrProfile = [opts.snrProfile, opts.snrProfile(end)*ones(1, opts.nFrames-length(opts.snrProfile))];
    end
    opts.snrProfile = opts.snrProfile(1:opts.nFrames);

    %% Paths
    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    root = fullfile(thisDir, '..');
    addpath(fullfile(root,'tx','src'));
    addpath(fullfile(root,'rx'));
    addpath(fullfile(root,'channel'));
    addpath(fullfile(root,'ml_addons'));

    %% Test image
    testImg = imread(fullfile(root,'tx','images','Testing examples.jpg'));
    testImg = im2double(testImg);
    if size(testImg,3) == 3, testImg = rgb2gray(testImg); end
    testImg = testImg(1:64, 1:64);

    %% Init PolicyController
    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end
    pc = PolicyController(opts.policyCsvPath);
    pc.Verbose = opts.verbose;

    param = struct();
    param.channelType = opts.channelType;

    rxSettings = struct();
    rxSettings.originalImage = testImg;
    rxSettings.policyCsvPath = opts.policyCsvPath;
    rxSettings.useGaOptimizer = opts.useGaOptimizer;
    rxSettings.gaSettings = struct('popSize',30,'generations',10,'verbose',false);
    rxSettings.verbose = opts.verbose;
    rxSettings.useMLQualityPredictor = opts.useML;
    rxSettings.useMLChannelPredictor = opts.useML;
    rxSettings.predictorParams = struct('predictionHorizon',3,'windowSize',15);

    %% Header
    mlTag = 'ML-ON';
    if ~opts.useML, mlTag = 'BASELINE'; end
    fprintf('\n======================================================================\n');
    fprintf('  BOLD CLOSED-LOOP [%s] | %dx%d | %d frames | %s\n', ...
        mlTag, size(testImg,1), size(testImg,2), opts.nFrames, opts.channelType);
    fprintf('======================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %6s | %s\n', ...
        'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR','ML Status');
    fprintf('%s\n', repmat('-',1,72));

    %% Logging
    results = struct();
    results.nFrames     = opts.nFrames;
    results.snrProfile  = opts.snrProfile;
    results.useML       = opts.useML;
    results.ssim        = zeros(1, opts.nFrames);
    results.psnr        = zeros(1, opts.nFrames);
    results.ml_ssim     = zeros(1, opts.nFrames);
    results.ml_psnr     = zeros(1, opts.nFrames);
    results.K           = zeros(1, opts.nFrames);
    results.mcs_base    = zeros(1, opts.nFrames);
    results.mcs_enh     = zeros(1, opts.nFrames);
    results.success     = true(1, opts.nFrames);
    results.predicted_snr = zeros(1, opts.nFrames);
    results.channel_trend = cell(1, opts.nFrames);

    channelHistory = struct('snr', []);

    %% Main loop
    for f = 1:opts.nFrames
        frameId = mod(f-1, 256);
        snr = opts.snrProfile(f);
        channelHistory.snr(end+1) = snr;
        param.SNR_dB = snr;
        param.seed = f;

        rxSettings.channelHistory = channelHistory;

        policy = pc.tick(frameId);
        N = 8;
        [dctCoeffs, dctInfo] = DCT_Block(double(testImg), N);
        dctReco = zeros(dctInfo.Hp, dctInfo.Wp);

        nBlocksW = dctInfo.Wp / N;
        nBlocksH = dctInfo.Hp / N;
        blkOk = 0;

        for blockId = 0:(nBlocksH * nBlocksW - 1)
            bRow = floor(double(blockId) / double(nBlocksW));
            bCol = mod(double(blockId), double(nBlocksW));
            rS = bRow*N + 1;
            cS = bCol*N + 1;
            block = dctCoeffs(rS:rS+N-1, cS:cS+N-1);

            try
            if policy.tx_mode == "combined"
                [txWave, txInfo] = two_layer_tx_pipeline(block, policy, frameId, blockId, 0);
                [rxWave, ~] = channel_model_v1(txWave, param);
                rxWave = [rxWave; zeros(10,1)];  % pad for multipath
                dctBlockHat = two_layer_rx_v2(rxWave, txInfo);
                dctReco(rS:rS+N-1, cS:cS+N-1) = dctBlockHat;
                blkOk = blkOk + 1;
            else
                [txWave, txInfo] = two_layer_tx_pipeline(block, policy, frameId, blockId, 1);
                [rxWave, ~] = channel_model_v1(txWave, param);
                rxWave = [rxWave; zeros(10,1)];
                dctBaseHat = two_layer_rx_v2(rxWave, txInfo);
                [txWave, txInfo] = two_layer_tx_pipeline(block, policy, frameId, blockId, 2);
                [rxWave, ~] = channel_model_v1(txWave, param);
                rxWave = [rxWave; zeros(10,1)];
                dctEnhHat = two_layer_rx_v2(rxWave, txInfo);
                dctReco(rS:rS+N-1, cS:cS+N-1) = dctBaseHat + dctEnhHat;
                blkOk = blkOk + 1;
            end
            catch, end
        end

        recon = IDCT(dctReco, dctInfo);

        ssimVal = ssim(recon, testImg);
        psnrVal = psnr(recon, testImg);

        quality = struct();
        quality.hasMetrics = true;
        quality.ssim = ssimVal;
        quality.psnr = psnrVal;
        quality.mse  = mean((testImg(:) - recon(:)).^2);

        rxSettings.dctCoeffsReceived = dctReco;

        rxRes = struct('success', true, 'errors', {{}}, 'warnings', {{}}, ...
            'blkOK', blkOk, 'nBlocks', nBlocksH*nBlocksW);

        if opts.useML
            feedback = bold_rx_feedback(quality, rxRes, policy, rxSettings);
        else
            feedback = rx_feedback(quality, rxRes, policy, rxSettings);
        end

        pc.Policy = feedback.policy;
        pc.LastVersion = feedback.policy.version;

        results.ssim(f)     = ssimVal;
        results.psnr(f)     = psnrVal;
        results.K(f)        = double(policy.K);
        results.mcs_base(f) = double(policy.mcs_base);
        results.mcs_enh(f)  = double(policy.mcs_enh);
        results.success(f)  = true;

        if isfield(feedback, 'ml_ssim'), results.ml_ssim(f) = feedback.ml_ssim; end
        if isfield(feedback, 'ml_psnr'), results.ml_psnr(f) = feedback.ml_psnr; end
        if isfield(feedback, 'predictedSNR') && ~isnan(feedback.predictedSNR(1))
            results.predicted_snr(f) = feedback.predictedSNR(1);
        end
        if isfield(feedback, 'channelTrend')
            results.channel_trend{f} = feedback.channelTrend;
        end

        mlStatus = '';
        if opts.useML
            if isfield(feedback, 'mlQualityUsed') && feedback.mlQualityUsed, mlStatus = [mlStatus 'Q']; end
            if isfield(feedback, 'mlChannelUsed') && feedback.mlChannelUsed, mlStatus = [mlStatus 'C']; end
            if isempty(mlStatus), mlStatus = 'fallback'; end
        else
            mlStatus = '-';
        end

        fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %6.2f | %s\n', ...
            f, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
            ssimVal, psnrVal, mlStatus);
    end

    %% Summary
    ok = results.success & results.ssim > 0;
    fprintf('\n======================================================================\n');
    if any(ok)
        fprintf('  Avg SSIM = %.4f  |  Avg PSNR = %.2f dB\n', mean(results.ssim(ok)), mean(results.psnr(ok)));
    end
    fprintf('  Mode: %s  |  Success: %d/%d frames\n', mlTag, sum(results.success), opts.nFrames);
    fprintf('  K range: %d-%d  |  MCS enh: %d-%d\n', ...
        min(results.K), max(results.K), min(results.mcs_enh), max(results.mcs_enh));
    fprintf('======================================================================\n');

    try bold_plot_results(results, opts); catch, end
end

function bold_plot_results(r, opts)
    figure('Position',[50 50 1200 700],'Name','Bold ML Closed-Loop Results');
    frames = 1:r.nFrames;

    subplot(2,3,1);
    plot(frames, opts.snrProfile, 'b-o','MarkerSize',3); hold on;
    if any(r.predicted_snr > 0)
        plot(frames, r.predicted_snr, 'r--s','MarkerSize',3);
        legend('Actual','LSTM Predicted');
    end
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel SNR + LSTM Prediction'); grid on;

    subplot(2,3,2);
    plot(frames, r.ssim, 'g-o','MarkerSize',3); hold on;
    if any(r.ml_ssim > 0)
        plot(frames, r.ml_ssim, 'm--s','MarkerSize',3);
        legend('Actual SSIM','ML Predicted');
    end
    yline(0.9,'r--','Target');
    xlabel('Frame'); ylabel('SSIM'); title('Quality (SSIM)'); ylim([0 1]); grid on;

    subplot(2,3,3);
    plot(frames, r.psnr, 'b-o','MarkerSize',3); hold on;
    if any(r.ml_psnr > 0)
        plot(frames, r.ml_psnr, 'm--s','MarkerSize',3);
        legend('Actual PSNR','ML Predicted');
    end
    yline(30,'r--','Target');
    xlabel('Frame'); ylabel('PSNR (dB)'); title('Quality (PSNR)'); grid on;

    subplot(2,3,4);
    plot(frames, r.mcs_base,'b-s','MarkerSize',3); hold on;
    plot(frames, r.mcs_enh,'r-o','MarkerSize',3);
    legend('Base','Enh'); xlabel('Frame'); ylabel('MCS'); title('MCS Adaptation'); grid on;

    subplot(2,3,5);
    plot(frames, r.K, 'k-d','MarkerSize',3);
    xlabel('Frame'); ylabel('K'); title('Layer Split (K)'); ylim([0 64]); grid on;

    subplot(2,3,6);
    mlActive = zeros(1, r.nFrames);
    for i = 1:r.nFrames
        if r.ml_ssim(i) > 0, mlActive(i) = mlActive(i) + 1; end
        if r.predicted_snr(i) > 0, mlActive(i) = mlActive(i) + 1; end
    end
    bar(frames, mlActive, 'FaceColor', [0.3 0.7 0.5]);
    xlabel('Frame'); ylabel('# ML Models Active');
    title('ML Model Activity'); ylim([0 3]); grid on;

    tag = 'ML'; if ~r.useML, tag = 'Baseline'; end
    sgtitle(sprintf('Bold Closed-Loop [%s] | Avg SSIM=%.4f | Avg PSNR=%.2f dB', ...
        tag, mean(r.ssim), mean(r.psnr)));

    try saveas(gcf, sprintf('bold_results_%s.png', lower(tag))); end
end
