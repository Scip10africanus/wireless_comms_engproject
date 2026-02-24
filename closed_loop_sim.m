function results = closed_loop_sim(opts)
%% CLOSED_LOOP_SIM  Multi-frame TX -> Channel -> RX with adaptive feedback
% Runs the full closed loop: PolicyController -> TX -> Channel -> RX -> Feedback -> CSV -> loop
%
% Usage:
%   closed_loop_sim()
%   closed_loop_sim(struct('nFrames',30, 'snrProfile',[25 20 15 10 15 20 25]))

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),       opts.nFrames = 50; end
    if ~isfield(opts,'useGaOptimizer'),opts.useGaOptimizer = true; end
    if ~isfield(opts,'usePredictor'),  opts.usePredictor = true; end
    if ~isfield(opts,'channelType'),   opts.channelType = 'AWGN'; end
    if ~isfield(opts,'policyCsvPath'), opts.policyCsvPath = fullfile(pwd,'policy.csv'); end
    if ~isfield(opts,'verbose'),       opts.verbose = true; end
    if ~isfield(opts,'useOctaveMode'), opts.useOctaveMode = true; end

    % SNR profile - simulates varying channel
    if ~isfield(opts,'snrProfile')
        opts.snrProfile = [ ...
  31 29 27 24 21 18 15 12 10  8 ...
  10 12 15 18 22 26 30 33 28 24 ...
  20 16 13 11  9  8  9 9 9 9 ...
  9 9 9 34 38 41 36 30 24 24 ...
  24 24 24 10 10 10 10 10 10 10];
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
    testImg = imread(fullfile("tx\images", "Testing examples.jpg"));
    testImg = im2double(testImg);
    if size(testImg,3) == 3
        testImg = rgb2gray(testImg);
    else
        testImg = img;
    end
    testImg = testImg(1:64, 1:64); 

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

    %% main loop (per-block PPDU)
    for f = 1:opts.nFrames
        frameId = mod(f-1, 256);
        snr = opts.snrProfile(f);

        try
            % update policy 
            policy = pc.tick(frameId);

            N = 8;
            [dctCoeffs, dctInfo] = DCT_Block(double(testImg), N);

            % DCT matrix for recon
            dctReco = zeros(dctInfo.Hp, dctInfo.Wp);

            okCount = 0;
            nBlocksW = dctInfo.Wp / N;
            nBlocksH = dctInfo.Hp / N;
            for blockId = 0:(nBlocksH * nBlocksW) -1 
                [rStart, cStart] = block_pos(blockId, nBlocksW, N);
                block = dctCoeffs(rStart:rStart + N - 1,cStart:cStart + N - 1);
                % TX: 
                [txWave, txInfo] = two_layer_tx_pipeline(block, pc ,policy, frameId,blockId);

                %  Channel: 
                [rxWave, chInfo] = channel_model(txWave, struct('channelType',opts.channelType,'SNR_dB',snr));

                % --- RX: recover
                [dctBlockHat] = two_layer_rx_v2(rxWave, txInfo, rxSettings);
                
                dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = dctBlockHat;
                okCount = okCount+1;
            end
            % IDCT
            recon = IDCT(dctReco,dctInfo);
            disp(class(recon));
            fprintf("recon range: [%g,%g]\n", min(recon(:)), max(recon(:)));
            ssimVal = ssim(recon, testImg);
            psnrVal = psnr(recon, testImg);
            % FEEDBACK LOOP (frame-level) 
            quality = struct();
            quality.hasMetrics = true;
            quality.ssim = ssimVal;
            quality.psnr = psnrVal;
            quality.mse  = mean((testImg(:) - recon(:)).^2);

            rxRes = struct('success', true, 'errors', {{}}, 'warnings', {{}}, ...
                'blkOK', okCount, 'nBlocks', nBlocksH*nBlocksW);

            feedback = rx_feedback(quality, rxRes, policy, rxSettings);

            chCond = struct('snr_dB', snr);
            newPolicy = rx_policy_update(policy, quality, feedback, chCond);

            pc.Policy = newPolicy;            % update policy
            pc.LastVersion = newPolicy.version;

            %  CSV writing for furtrue
            % write_policy_csv(opts.policyCsvPath, newPolicy);

            results.success(f) = true;

            fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %6.2f | blkOK=%d\n', ...
                f, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
                ssimVal, psnrVal, okCount);
        catch ME
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

function [rStart, cStart] = block_pos(blockId, nBlocksW, N)
% blockId: 0-based
    bRow = floor(double(blockId) / double(nBlocksW)); % 0-based
    bCol = mod(double(blockId), double(nBlocksW));    % 0-based
    rStart = bRow*N + 1;  % 1-based index
    cStart = bCol*N + 1;
end
