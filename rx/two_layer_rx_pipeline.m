function [outputImage, feedback, results] = two_layer_rx_pipeline(rxSignal, txInfo, settings)
% two_layer_rx_pipeline - full RX chain for the wavelet+DCT two-layer system
%
% steps: OFDM demod -> wavelet base decoder -> DCT residual decoder
%        -> layer fusion -> quality metrics -> feedback -> policy update
%
% if the enhancement layer is too noisy (EVM > 50%) it gets dropped
% and we just use the base layer image (graceful degradation)

    if nargin < 3, settings = struct(); end
    if ~isfield(settings, 'policyCsvPath'),       settings.policyCsvPath = ''; end
    if ~isfield(settings, 'verbose'),             settings.verbose = true; end
    if ~isfield(settings, 'useGaOptimizer'),      settings.useGaOptimizer = false; end
    if ~isfield(settings, 'gaSettings'),          settings.gaSettings = struct(); end
    if ~isfield(settings, 'useChannelPredictor'), settings.useChannelPredictor = false; end
    if ~isfield(settings, 'predictorParams'),     settings.predictorParams = struct(); end
    if ~isfield(settings, 'targetSsim'),          settings.targetSsim = 0.9; end
    if ~isfield(settings, 'targetPsnr'),          settings.targetPsnr = 30; end

    policy = txInfo.policy;
    results = struct('success', true, 'errors', {{}}, 'warnings', {{}});

    if settings.verbose
        fprintf('[RX-Wavelet] Policy v%d: levels=%d Qb=%d Qe=%d | %dx%d\n', ...
            policy.version, txInfo.waveletLevels, txInfo.Qb, txInfo.Qe, ...
            txInfo.imgHeight, txInfo.imgWidth);
    end

    %% OFDM demod
    try
        [baseBits, enhBits, demapInfo] = two_layer_demapper(rxSignal, txInfo.ofdmInfo);
        results.demapInfo = demapInfo;
        if settings.verbose
            if demapInfo.headerValid, hStr = 'valid'; else, hStr = 'invalid'; end
            fprintf('[RX-Wavelet] Demod: baseEVM=%.1f%% enhEVM=%.1f%% header=%s\n', ...
                demapInfo.baseEVM, demapInfo.enhEVM, hStr);
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('OFDM demod failed: %s', ME.message);
        outputImage = uint8(zeros(txInfo.imgHeight, txInfo.imgWidth));
        % generate empty feedback so the loop can continue
        quality0 = struct('hasMetrics',false,'ssim',NaN,'psnr',NaN,'mse',NaN);
        feedback = rx_feedback(quality0, struct('success',false), policy, ...
            struct('targetSsim',0.9,'targetPsnr',30,'useGaOptimizer',false, ...
                   'useChannelPredictor',false,'policyCsvPath',''));
        return;
    end

    %% decode base layer (wavelet LL reconstruction)
    enhLost = false;
    try
        [baseImage, ~] = wavelet_base_decoder(baseBits, txInfo);
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('Base decode failed: %s', ME.message);
        outputImage = uint8(zeros(txInfo.imgHeight, txInfo.imgWidth));
        quality0 = struct('hasMetrics',false,'ssim',NaN,'psnr',NaN,'mse',NaN);
        feedback = rx_feedback(quality0, struct('success',false), policy, ...
            struct('targetSsim',0.9,'targetPsnr',30,'useGaOptimizer',false, ...
                   'useChannelPredictor',false,'policyCsvPath',''));
        return;
    end

    %% decode enhancement layer (DCT residual)
    residual = [];
    try
        residual = residual_dct_decoder(enhBits, txInfo);
    catch ME
        enhLost = true;
        results.warnings{end+1} = sprintf('Enhancement decode failed: %s', ME.message);
        if settings.verbose
            fprintf('[RX-Wavelet] Enhancement lost, using base only\n');
        end
    end

    % drop enhancement if EVM is too high (>50% means too many errors)
    if ~enhLost && isfield(demapInfo, 'enhEVM') && demapInfo.enhEVM > 50
        enhLost = true;
        results.warnings{end+1} = 'Enhancement EVM too high, dropping enhancement layer';
        if settings.verbose
            fprintf('[RX-Wavelet] Enhancement EVM=%.1f%% > 50%%, dropping\n', demapInfo.enhEVM);
        end
    end

    %% fuse the two layers
    if enhLost
        [outputImage, fusionInfo] = layer_fusion(baseImage, [], 'base_only');
    else
        [outputImage, fusionInfo] = layer_fusion(baseImage, residual, 'full');
    end
    results.fusionInfo = fusionInfo;

    %% quality metrics
    quality = struct('hasMetrics', false, 'ssim', NaN, 'psnr', NaN, 'mse', NaN);
    if isfield(settings, 'originalImage') && ~isempty(settings.originalImage)
        quality = image_quality_metrics(settings.originalImage, outputImage);
    end
    results.quality = quality;
    results.enhLost = enhLost;

    if settings.verbose && quality.hasMetrics
        if enhLost, modeStr = '(base only)'; else, modeStr = '(full)'; end
        fprintf('[RX-Wavelet] SSIM=%.4f PSNR=%.2f dB %s\n', quality.ssim, quality.psnr, modeStr);
    end

    %% feedback and policy update for next frame
    feedback = rx_feedback(quality, results, policy, settings);
    results.feedback = feedback;

    if settings.verbose
        fprintf('[RX-Wavelet] Policy v%d -> v%d\n', policy.version, feedback.policy.version);
    end
end
