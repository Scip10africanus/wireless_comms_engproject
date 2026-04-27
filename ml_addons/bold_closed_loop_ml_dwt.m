function results = bold_closed_loop_ml_dwt(opts)
%% BOLD_CLOSED_LOOP_ML_DWT  Closed-loop simulation with ML on the DWT pipeline
% Companion to bold_closed_loop_ml.m. Same three ML models plug in, but
% the TX/RX pipeline is DWT (two_layer_tx_wavelet + two_layer_rx_testing_dwt).
% Feeds real DWT features from dwt_extract_features into the quality
% predictor (slots 13-15), instead of the DCT-proxy fallback used by the
% DCT closed loop.
%
% Bold (Unubold) - Apr 2026. Added to address Zibo's concern that the ML
% part had only been tested in closed loop for DCT, with DWT features
% appended as a feature-vector afterthought rather than a validated
% integration path.
%
% ML-closed-loop hooks:
%   1. bold_quality_predictor - gets real DWT features (not priors) via
%      settings.dwtFeatures in bold_rx_feedback.m
%   2. bold_channel_lstm - unchanged vs DCT path, runs off channelHistory
%   3. bold_symbol_denoiser - NOT applied in the DWT loop for now. The
%      denoiser is wired to wlanNonHTDemodulate inside bold_demodem_denoised,
%      which is a drop-in for rx/demodem.m. Activating it on the DWT path
%      would require a bold_two_layer_rx_denoised_dwt wrapper. Left as a
%      follow-up to keep this PR focused on integration validation.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),        opts.nFrames = 50; end
    if ~isfield(opts,'useML'),          opts.useML = true; end
    if ~isfield(opts,'useGaOptimizer'), opts.useGaOptimizer = true; end
    if ~isfield(opts,'channelType'),    opts.channelType = 'RICIAN'; end
    if ~isfield(opts,'policyCsvPath'),  opts.policyCsvPath = fullfile(pwd,'policy_dwt.csv'); end
    if ~isfield(opts,'verbose'),        opts.verbose = true; end
    if ~isfield(opts,'level'),          opts.level = 1; end
    if ~isfield(opts,'rx_Qb'),          opts.rx_Qb = 1; end
    if ~isfield(opts,'rx_Qe'),          opts.rx_Qe = 1; end
    if ~isfield(opts,'ricianK'),        opts.ricianK = 3; end

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
    addpath(fullfile(root,'common'));
    addpath(fullfile(root,'tx','src'));
    addpath(fullfile(root,'rx'));
    addpath(fullfile(root,'channel'));
    addpath(fullfile(root,'ml_addons'));

    %% Test image - loaded at uint8-double scale (0..255) to match DWT baseline
    testImg = imread(fullfile(root,'tx','images','Testing examples.jpg'));
    if size(testImg,3) == 3, testImg = rgb2gray(testImg); end
    testImg = double(testImg(1:64, 1:64));   % 0..255

    %% Init PolicyController
    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end
    pc = PolicyController(opts.policyCsvPath);
    pc.Verbose = opts.verbose;
    pc.Policy.tx_mode = "split";   % wavelet pipeline is split-only

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
    % Quality targets - populated explicitly because the baseline rx_feedback.m
    % (team rewrite, Risdon d6003d2) dereferences these without isfield guards.
    % bold_rx_feedback.m guards with isfield so only strictly required for the
    % baseline branch, but we set them unconditionally for consistency.
    rxSettings.targetSsim            = 0.9;
    rxSettings.targetPsnr            = 30;

    %% Header
    mlTag = 'ML-ON';
    if ~opts.useML, mlTag = 'BASELINE'; end
    fprintf('\n======================================================================\n');
    fprintf('  BOLD DWT CLOSED-LOOP [%s] | %dx%d | %d frames | %s | level=%d\n', ...
        mlTag, size(testImg,1), size(testImg,2), opts.nFrames, opts.channelType, double(opts.level));
    fprintf('======================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %6s | %7s | %7s | %s\n', ...
        'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR','basePER','enhPER','ML');
    fprintf('%s\n', repmat('-',1,90));

    %% Logging (same shape as bold_closed_loop_ml.m for easy comparison)
    results = struct();
    results.pipeline       = "DWT";
    results.nFrames        = opts.nFrames;
    results.snrProfile     = opts.snrProfile;
    results.useML          = opts.useML;
    results.useGaOptimizer = logical(opts.useGaOptimizer);
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
    results.basePER        = zeros(1, opts.nFrames);
    results.enhPER         = zeros(1, opts.nFrames);
    results.enh_valid      = false(1, opts.nFrames);
    % DWT feature trace - useful for debugging predictor behaviour
    results.feat_E_LL      = zeros(1, opts.nFrames);
    results.feat_E_HH      = zeros(1, opts.nFrames);
    results.feat_base_share = zeros(1, opts.nFrames);

    channelHistory = struct('snr', []);

    %% Main loop
    for f = 1:opts.nFrames
        frameId = uint8(mod(f-1, 256));
        snr     = opts.snrProfile(f);
        channelHistory.snr(end+1) = snr;
        param.SNR_dB = snr;
        param.seed   = f;
        param.ricianK = double(opts.ricianK);

        rxSettings.channelHistory = channelHistory;

        policy = pc.tick(frameId);
        policy.use_crc16 = false;   % DWT RX doesn't implement CRC strip

        recon    = zeros(size(testImg));
        enhValid = false;
        dwtFeat  = struct();         % DWT features for the quality predictor
        rxinfo   = struct('basePER',1,'enhPER',1, ...
                          'numBaseOk',0,'numEnhOk',0, ...
                          'numBasePkts',0,'numEnhPkts',0);

        try
            % TX
            [txWaves, txInfo] = two_layer_tx_wavelet(testImg, policy, frameId);

            % Channel
            rxWaves = apply_channel_to_waves(txWaves, param);

            % RX - note: DWT denoiser wrapper not yet wired (see header comment)
            [res_hat, rxinfo] = two_layer_rx_testing_dwt(rxWaves, txInfo, opts.rx_Qb, opts.rx_Qe);

            % Reconstruct
            recon = IDWT(res_hat, txInfo.level);
            recon = max(0, min(255, recon));   % clip to valid range

            % DWT features from decoded res_hat - these are the real values,
            % not priors. Feeds slots 13-15 of the quality predictor.
            featOpts = struct('featureVersion', uint32(1), ...
                              'wavelet', string(txInfo.DWTInfo.wavelet), ...
                              'level',   double(txInfo.level));
            dwtFeat = dwt_extract_features(res_hat, txInfo, featOpts);

            % Enhancement-valid gate analogous to the DCT loop's enhOK check.
            % For DWT we say "enh valid" if all enh packets decoded AND the
            % final reconstruction is finite (no NaN/Inf leaked through from
            % a partial subband).
            enhValid = (rxinfo.numEnhOk >= rxinfo.numEnhPkts) && ...
                       (rxinfo.numBaseOk >= rxinfo.numBasePkts) && ...
                       all(isfinite(recon(:)));

        catch ME
            results.success(f) = false;
            % Always surface the first few frame errors — silent catches hide real bugs.
            if opts.verbose || f <= 3
                fprintf('[Frame %d] TX/RX error: %s (%s)\n', f, ME.message, ME.identifier);
                if f == 1 && ~isempty(ME.stack)
                    fprintf('           at %s line %d\n', ME.stack(1).name, ME.stack(1).line);
                end
            end
        end

        % Quality metrics
        ssimVal = safe_ssim(recon, testImg);
        psnrVal = safe_psnr(recon, testImg);

        quality = struct();
        quality.hasMetrics = true;
        quality.ssim       = ssimVal;
        quality.psnr       = psnrVal;
        quality.mse        = mean((testImg(:) - recon(:)).^2);

        % For the quality predictor we synthesise a DCT coefficient matrix
        % by block-DCT-ing the RECONSTRUCTED image. This keeps features 1-12
        % populated with meaningful signal (energy concentration, DC stats,
        % etc.) that matches the predictor's training distribution far better
        % than passing zeros. The predictor was trained on DCT features of
        % received DCT coefficients; DCT of a reconstructed image is the
        % closest natural analogue in the DWT pipeline.
        dctCoeffsProxy = block_dct_proxy(recon, 8);

        % feedback settings for this frame
        rxSettings.dctCoeffsReceived = dctCoeffsProxy;
        rxSettings.dwtFeatures       = dwtFeat;       % real DWT features (slots 13-15)
        rxSettings.noiseVar          = 0.01;          % DWT RX doesn't expose LLTF noiseVar
                                                      % via public demodem path yet; use prior

        rxRes = struct( ...
            'success',      results.success(f), ...
            'errors',       {{}}, ...
            'warnings',     {{}}, ...
            'blkOK',        double(rxinfo.numBaseOk + rxinfo.numEnhOk), ...
            'nBlocks',      max(1, double(rxinfo.numBasePkts + rxinfo.numEnhPkts)), ...
            'enhFailRatio', double(rxinfo.enhPER), ...
            'baseOnlyRatio', double(rxinfo.numBaseOk)/max(1,double(rxinfo.numBasePkts)) * double(rxinfo.numEnhOk < rxinfo.numEnhPkts));

        if opts.useML
            feedback = bold_rx_feedback(quality, rxRes, policy, rxSettings);
        else
            feedback = rx_feedback(quality, rxRes, policy, rxSettings);
        end

        pc.Policy      = feedback.policy;
        pc.LastVersion = feedback.policy.version;

        % log
        results.ssim(f)      = ssimVal;
        results.psnr(f)      = psnrVal;
        results.K(f)         = double(policy.K);
        results.mcs_base(f)  = double(policy.mcs_base);
        results.mcs_enh(f)   = double(policy.mcs_enh);
        results.basePER(f)   = double(rxinfo.basePER);
        results.enhPER(f)    = double(rxinfo.enhPER);
        results.enh_valid(f) = enhValid;
        if isfield(dwtFeat,'E_LL_ratio'), results.feat_E_LL(f)       = double(dwtFeat.E_LL_ratio); end
        if isfield(dwtFeat,'E_HH_ratio'), results.feat_E_HH(f)       = double(dwtFeat.E_HH_ratio); end
        if isfield(dwtFeat,'base_share'), results.feat_base_share(f) = double(dwtFeat.base_share); end

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

        fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %6.2f | %6.1f%% | %6.1f%% | %s\n', ...
            f, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
            ssimVal, psnrVal, 100*rxinfo.basePER, 100*rxinfo.enhPER, mlStatus);
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
    fprintf('  Avg base PER: %.1f%% | Avg enh PER: %.1f%%\n', ...
        100*mean(results.basePER), 100*mean(results.enhPER));
    if any(results.feat_E_LL > 0)
        fprintf('  DWT features: E_LL=%.3f+-%.3f  E_HH=%.3f+-%.3f  base_share=%.3f+-%.3f\n', ...
            mean(results.feat_E_LL), std(results.feat_E_LL), ...
            mean(results.feat_E_HH), std(results.feat_E_HH), ...
            mean(results.feat_base_share), std(results.feat_base_share));
    end
    fprintf('======================================================================\n');

    try bold_plot_results(results, opts); catch, end
end


%% -----------------------------------------------------------------------
%  Helpers
%% -----------------------------------------------------------------------

function rxWaves = apply_channel_to_waves(txWaves, chanParam)
    rxWaves = struct('base', [], 'enh', {{}});
    if isfield(txWaves,'base') && ~isempty(txWaves.base)
        rxWaves.base = cell(size(txWaves.base));
        for i = 1:numel(txWaves.base)
            [rx, ~] = channel_model_v1(txWaves.base{i}, chanParam);
            rxWaves.base{i} = [rx; zeros(10,1)];
        end
    else
        rxWaves.base = {};
    end
    if isfield(txWaves,'enh') && ~isempty(txWaves.enh)
        rxWaves.enh = cell(size(txWaves.enh));
        for i = 1:numel(txWaves.enh)
            [rx, ~] = channel_model_v1(txWaves.enh{i}, chanParam);
            rxWaves.enh{i} = [rx; zeros(10,1)];
        end
    else
        rxWaves.enh = {};
    end
end

function dctMat = block_dct_proxy(img, N)
    % Block DCT (NxN) of a reconstructed image, returned as a Hp x Wp matrix
    % in the same layout that two_layer_rx_testing_dct produces. Used as the
    % dctCoeffsReceived input to bold_quality_predictor so slots 1-12 of the
    % feature vector carry real signal instead of zeros.
    [H, W] = size(img);
    Hp = ceil(H/N)*N;
    Wp = ceil(W/N)*N;
    if Hp ~= H || Wp ~= W
        img = padarray(img, [Hp-H, Wp-W], 'symmetric', 'post');
    end
    dctMat = zeros(Hp, Wp);
    for r = 1:N:Hp
        for c = 1:N:Wp
            dctMat(r:r+N-1, c:c+N-1) = dct2(img(r:r+N-1, c:c+N-1));
        end
    end
end

function s = safe_ssim(a, b)
    try
        s = ssim(a, b);
    catch
        s = 0;
    end
    if ~isfinite(s), s = 0; end
end

function p = safe_psnr(a, b)
    try
        p = psnr(a, b, 255);
    catch
        p = 0;
    end
    if ~isfinite(p), p = 0; end
end

function bold_plot_results(r, opts)
    figure('Position',[50 50 1200 800],'Name','Bold DWT ML Closed-Loop Results');
    frames = 1:r.nFrames;

    subplot(3,3,1);
    plot(frames, opts.snrProfile, 'b-o','MarkerSize',3); hold on;
    if any(r.predicted_snr > 0)
        plot(frames, r.predicted_snr, 'r--s','MarkerSize',3);
        legend('Actual','LSTM Predicted');
    end
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel SNR + LSTM'); grid on;

    subplot(3,3,2);
    plot(frames, r.ssim, 'g-o','MarkerSize',3); hold on;
    if any(r.ml_ssim > 0)
        plot(frames, r.ml_ssim, 'm--s','MarkerSize',3);
        legend('Actual','ML Pred');
    end
    yline(0.9,'r--','Target');
    xlabel('Frame'); ylabel('SSIM'); title('SSIM'); ylim([0 1]); grid on;

    subplot(3,3,3);
    plot(frames, r.psnr, 'b-o','MarkerSize',3); hold on;
    if any(r.ml_psnr > 0)
        plot(frames, r.ml_psnr, 'm--s','MarkerSize',3);
    end
    yline(30,'r--','Target');
    xlabel('Frame'); ylabel('PSNR (dB)'); title('PSNR'); grid on;

    subplot(3,3,4);
    plot(frames, r.mcs_base,'b-s','MarkerSize',3); hold on;
    plot(frames, r.mcs_enh, 'r-o','MarkerSize',3);
    legend('Base','Enh'); xlabel('Frame'); ylabel('MCS'); title('MCS Adaptation'); grid on;

    subplot(3,3,5);
    plot(frames, r.K,'k-d','MarkerSize',3);
    xlabel('Frame'); ylabel('K'); title('K (policy)'); grid on;

    subplot(3,3,6);
    plot(frames, 100*r.basePER,'r-','LineWidth',1.5); hold on;
    plot(frames, 100*r.enhPER, 'm-','LineWidth',1.5);
    legend('Base PER','Enh PER');
    xlabel('Frame'); ylabel('PER (%)'); title('Packet Error Rate'); grid on;

    subplot(3,3,7);
    plot(frames, r.feat_E_LL,'b-','LineWidth',1.5); hold on;
    plot(frames, r.feat_E_HH,'r-','LineWidth',1.5);
    legend('E\_LL','E\_HH');
    xlabel('Frame'); ylabel('Ratio'); title('DWT energy features'); grid on; ylim([0 1]);

    subplot(3,3,8);
    plot(frames, r.feat_base_share,'k-','LineWidth',1.5);
    xlabel('Frame'); ylabel('base share'); title('DWT base\_share'); grid on; ylim([0 1]);

    subplot(3,3,9);
    bar(frames, double(r.enh_valid), 'FaceColor', [0.2 0.6 0.4]);
    xlabel('Frame'); ylabel('Enh valid'); title('Enh decode success'); ylim([0 1.2]); grid on;

    tag = 'ML'; if ~r.useML, tag = 'Baseline'; end
    sgtitle(sprintf('Bold DWT Closed-Loop [%s] | Avg SSIM=%.4f | Avg PSNR=%.2f dB', ...
        tag, mean(r.ssim), mean(r.psnr)));

    try saveas(gcf, sprintf('bold_dwt_results_%s.png', lower(tag))); end
end
