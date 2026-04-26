function results = bold_closed_loop_ml(opts)
%% BOLD_CLOSED_LOOP_ML  Closed-loop simulation with all 3 ML models
% Requires: Image Processing Toolbox, WLAN Toolbox
%
% CHANGES v2 (Bold - Apr 2026):
%   - Replaced deleted two_layer_tx_pipeline with two_layer_tx_DCT
%     (processes all blocks as N x N x numBlocks in one call per frame)
%   - Replaced two_layer_rx_v2 with two_layer_rx_testing_dct
%   - Added enhancement layer garbage detection using rxinfo.blockOK
%     and rxinfo.sanityOK -- base-only fallback when enh decode is bad
%   - useGaOptimizer now defaults to FALSE (GA causes catastrophic
%     policy collapse when combined with closed loop; config 11 is best)
%   - DCT block packing moved outside frame loop (image is constant)
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),        opts.nFrames = 50; end
    if ~isfield(opts,'useML'),          opts.useML = true; end
    if ~isfield(opts,'useGaOptimizer'), opts.useGaOptimizer = false; end  % GA disabled: catastrophic interaction confirmed
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
    % Force split mode from frame 1.
    % Default policy has tx_mode='combined' which packs all 64 blocks into
    % one WLAN packet (5376 bytes) -- exceeds the 4095-byte PSDU limit.
    % rx_policy_update always sets split mode anyway, so this only affects frame 1.
    pc.Policy.tx_mode = "split";

    param = struct();
    param.channelType = opts.channelType;

    rxSettings = struct();
    rxSettings.originalImage         = testImg;
    rxSettings.policyCsvPath         = opts.policyCsvPath;
    rxSettings.useGaOptimizer        = opts.useGaOptimizer;
    rxSettings.gaSettings            = struct('popSize',30,'generations',10,'verbose',false);
    rxSettings.verbose               = opts.verbose;
    rxSettings.useMLQualityPredictor = opts.useML;
    rxSettings.useMLChannelPredictor = opts.useML;
    rxSettings.predictorParams       = struct('predictionHorizon',3,'windowSize',15);

    %% Header
    mlTag = 'ML-ON';
    if ~opts.useML, mlTag = 'BASELINE'; end
    fprintf('\n======================================================================\n');
    fprintf('  BOLD CLOSED-LOOP v2 [%s] | %dx%d | %d frames | %s\n', ...
        mlTag, size(testImg,1), size(testImg,2), opts.nFrames, opts.channelType);
    fprintf('======================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %6s | %4s | %s\n', ...
        'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR','Enh?','ML');
    fprintf('%s\n', repmat('-',1,76));

    %% Logging
    results = struct();
    results.nFrames        = opts.nFrames;
    results.snrProfile     = opts.snrProfile;
    results.useML          = opts.useML;
    results.ssim           = zeros(1, opts.nFrames);
    results.psnr           = zeros(1, opts.nFrames);
    results.ml_ssim        = zeros(1, opts.nFrames);
    results.ml_psnr        = zeros(1, opts.nFrames);
    results.K              = zeros(1, opts.nFrames);
    results.mcs_base       = zeros(1, opts.nFrames);
    results.mcs_enh        = zeros(1, opts.nFrames);
    results.success        = true(1, opts.nFrames);
    results.predicted_snr  = zeros(1, opts.nFrames);
    results.channel_trend  = cell(1, opts.nFrames);
    results.enh_valid      = false(1, opts.nFrames);

    channelHistory = struct('snr', []);

    %% Pre-compute block layout and pack DCT blocks into 3D array once.
    %  two_layer_tx_DCT expects blocks as [N x N x numBlocks].
    %  testImg is constant across frames so this only needs to run once.
    N = 8;
    [dctCoeffs, dctInfo] = DCT_Block(double(testImg), N);
    nBlocksW = dctInfo.Wp / N;
    nBlocksH = dctInfo.Hp / N;
    nBlocks  = nBlocksH * nBlocksW;

    blocks3D = zeros(N, N, nBlocks);
    for bi = 0:(nBlocks-1)
        bRow = floor(bi / nBlocksW);
        bCol = mod(bi, nBlocksW);
        rS = bRow*N + 1;  cS = bCol*N + 1;
        blocks3D(:,:,bi+1) = dctCoeffs(rS:rS+N-1, cS:cS+N-1);
    end

    %% Main loop
    for f = 1:opts.nFrames
        frameId = mod(f-1, 256);
        snr     = opts.snrProfile(f);
        channelHistory.snr(end+1) = snr;
        param.SNR_dB = snr;
        param.seed   = f;

        rxSettings.channelHistory = channelHistory;

        policy  = pc.tick(frameId);
        dctReco   = zeros(dctInfo.Hp, dctInfo.Wp);
        enhValid  = false;
        frameNoiseVar = 0.01;  % default prior; updated from LLTF if denoiser active

        try
            %% Choose RX function: ML path uses bold_two_layer_rx_denoised
            %  which applies the trained symbol denoiser before Viterbi decode.
            %  Standard path uses two_layer_rx_testing_dct (no denoising).
            if opts.useML
                % ML path: bold_two_layer_rx_denoised returns rxinfo with
                % denoiserActive=true when wlanNonHTDemodulate is available.
                % We also capture noiseVar directly from bold_demodem_denoised
                % via a thin wrapper for the base layer (feat 16 for predictor).
                rxFn = @(w,t) bold_two_layer_rx_denoised(w, t, snr);

                % Capture noiseVar from base-layer LLTF (Jie Du's estimation)
                % by running bold_demodem_denoised directly on the base wave first.
                % This is done inside the split-mode block below for the base layer.
            else
                rxFn = @(w,t) two_layer_rx_testing_dct(w, t);
            end

            if strcmp(policy.tx_mode, 'combined')
                %% --- COMBINED MODE: one OFDM packet for both layers ---
                [txWave, txInfo] = two_layer_tx_DCT(blocks3D, policy, frameId, 0, 0);
                [rxWave, ~]      = channel_model_v1(txWave, param);
                rxWave = [rxWave; zeros(10,1)];
                [DCTcoe, rxinfo] = rxFn(rxWave, txInfo);
                if rxinfo.blockOK && rxinfo.sanityOK
                    dctReco  = bold_assemble_dct(DCTcoe, dctInfo, N, nBlocksH, nBlocksW);
                    enhValid = true;
                end

            else
                %% --- SPLIT MODE: base and enhancement as separate packets ---

                % --- Base layer (layer_type = 1) ---
                [txWaveBase, txInfoBase] = two_layer_tx_DCT(blocks3D, policy, frameId, 0, 1);
                [rxWaveBase, ~]          = channel_model_v1(txWaveBase, param);
                rxWaveBase = [rxWaveBase; zeros(10,1)];
                [DCTbase, rxinfoBase]    = rxFn(rxWaveBase, txInfoBase);

                % Capture noiseVar from base layer LLTF (Jie Du, Apr 2026)
                if opts.useML && isfield(txInfoBase,'cfgBase') && ~isempty(txInfoBase.cfgBase)
                    try
                        [~, ~, ~, nv] = bold_demodem_denoised(rxWaveBase, txInfoBase.cfgBase, snr);
                        if isfinite(nv), frameNoiseVar = nv; end
                    catch, end
                end

                if rxinfoBase.blockOK
                    dctReco = bold_assemble_dct(DCTbase, dctInfo, N, nBlocksH, nBlocksW);
                end

                % --- Enhancement layer (layer_type = 2) ---
                [txWaveEnh, txInfoEnh] = two_layer_tx_DCT(blocks3D, policy, frameId, 0, 2);
                [rxWaveEnh, ~]         = channel_model_v1(txWaveEnh, param);
                rxWaveEnh = [rxWaveEnh; zeros(10,1)];
                [DCTenh, rxinfoEnh]    = rxFn(rxWaveEnh, txInfoEnh);

                % Enhancement validity gate.
                % Only add enh coefficients if the decode passed ALL checks.
                % A corrupted enh block added to a good base block causes the
                % binary SSIM cliff (Angela flagged Apr 2026): image is either
                % perfect or noise, never a graceful degradation.
                enhOK = rxinfoEnh.blockOK   && ...
                        rxinfoEnh.sanityOK  && ...
                        ~any(isnan(DCTenh(:))) && ...
                        ~any(isinf(DCTenh(:)));

                if enhOK
                    dctReco  = dctReco + bold_assemble_dct(DCTenh, dctInfo, N, nBlocksH, nBlocksW);
                    enhValid = true;
                elseif opts.verbose
                    reason = '';
                    if ~rxinfoEnh.blockOK,  reason = [reason 'blockOK=0 ']; end
                    if ~rxinfoEnh.sanityOK, reason = [reason 'sanity=0 ']; end
                    if any(isnan(DCTenh(:))), reason = [reason 'NaN ']; end
                    if any(isinf(DCTenh(:))), reason = [reason 'Inf ']; end
                    fprintf('[Frame %d] Enh dropped (%s)-> base only\n', f, strtrim(reason));
                end
            end

        catch ME
            results.success(f) = false;
            if opts.verbose
                fprintf('[Frame %d] TX/RX error: %s\n', f, ME.message);
            end
        end

        recon = IDCT(dctReco, dctInfo);

        ssimVal = ssim(recon, testImg);
        psnrVal = psnr(recon, testImg);

        quality = struct();
        quality.hasMetrics = true;
        quality.ssim       = ssimVal;
        quality.psnr       = psnrVal;
        quality.mse        = mean((testImg(:) - recon(:)).^2);

        rxSettings.dctCoeffsReceived = dctReco;
        rxSettings.noiseVar          = frameNoiseVar;  % Jie Du's LLTF estimate (feat 16)

        rxRes = struct('success', results.success(f), 'errors', {{}}, 'warnings', {{}}, ...
            'blkOK', nBlocks * double(enhValid), 'nBlocks', nBlocks);

        if opts.useML
            feedback = bold_rx_feedback(quality, rxRes, policy, rxSettings);
        else
            feedback = rx_feedback(quality, rxRes, policy, rxSettings);
        end

        pc.Policy      = feedback.policy;
        pc.LastVersion = feedback.policy.version;

        results.ssim(f)      = ssimVal;
        results.psnr(f)      = psnrVal;
        results.K(f)         = double(policy.K);
        results.mcs_base(f)  = double(policy.mcs_base);
        results.mcs_enh(f)   = double(policy.mcs_enh);
        results.enh_valid(f) = enhValid;

        if isfield(feedback,'ml_ssim'), results.ml_ssim(f) = feedback.ml_ssim; end
        if isfield(feedback,'ml_psnr'), results.ml_psnr(f) = feedback.ml_psnr; end
        if isfield(feedback,'predictedSNR') && ~isnan(feedback.predictedSNR(1))
            results.predicted_snr(f) = feedback.predictedSNR(1);
        end
        if isfield(feedback,'channelTrend')
            results.channel_trend{f} = feedback.channelTrend;
        end

        mlStatus = '';
        if opts.useML
            if isfield(feedback,'mlQualityUsed') && feedback.mlQualityUsed, mlStatus = [mlStatus 'Q']; end
            if isfield(feedback,'mlChannelUsed') && feedback.mlChannelUsed, mlStatus = [mlStatus 'C']; end
            if isempty(mlStatus), mlStatus = 'fallback'; end
        else
            mlStatus = '-';
        end

        fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %6.2f | %4s | %s\n', ...
            f, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
            ssimVal, psnrVal, mat2str(enhValid), mlStatus);
    end

    %% Summary
    ok = results.success & results.ssim > 0;
    fprintf('\n======================================================================\n');
    if any(ok)
        fprintf('  Avg SSIM = %.4f  |  Avg PSNR = %.2f dB\n', mean(results.ssim(ok)), mean(results.psnr(ok)));
    end
    fprintf('  Mode: %s  |  Success: %d/%d  |  Enh valid: %d/%d frames\n', ...
        mlTag, sum(results.success), opts.nFrames, sum(results.enh_valid), opts.nFrames);
    fprintf('  K range: %d-%d  |  MCS enh: %d-%d\n', ...
        min(results.K), max(results.K), min(results.mcs_enh), max(results.mcs_enh));
    fprintf('======================================================================\n');

    try bold_plot_results(results, opts); catch, end
end

%% -----------------------------------------------------------------------
%  bold_assemble_dct
%  Reassembles 2D DCT coefficient matrix from the 3D block array that
%  two_layer_rx_testing_dct returns [N x N x numBlocks].
%% -----------------------------------------------------------------------
function dctMat = bold_assemble_dct(DCTcoe, dctInfo, N, nBlocksH, nBlocksW)
    dctMat = zeros(dctInfo.Hp, dctInfo.Wp);
    for bi = 0:(nBlocksH * nBlocksW - 1)
        bRow = floor(bi / nBlocksW);
        bCol = mod(bi, nBlocksW);
        rS = bRow*N + 1;  cS = bCol*N + 1;
        dctMat(rS:rS+N-1, cS:cS+N-1) = DCTcoe(:,:,bi+1);
    end
end

%% -----------------------------------------------------------------------
%  bold_plot_results  (unchanged from v1)
%% -----------------------------------------------------------------------
function bold_plot_results(r, opts)
    figure('Position',[50 50 1200 700],'Name','Bold ML Closed-Loop Results v2');
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
    bar(frames, double(r.enh_valid), 'FaceColor', [0.2 0.6 0.4]);
    xlabel('Frame'); ylabel('Enh layer valid');
    title('Enhancement decode success'); ylim([0 1.2]); grid on;

    tag = 'ML'; if ~r.useML, tag = 'Baseline'; end
    sgtitle(sprintf('Bold Closed-Loop v2 [%s] | Avg SSIM=%.4f | Avg PSNR=%.2f dB', ...
        tag, mean(r.ssim), mean(r.psnr)));

    try saveas(gcf, sprintf('bold_results_%s.png', lower(tag))); end
end
