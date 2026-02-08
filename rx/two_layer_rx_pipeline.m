function [finalImage, feedback, results] = two_layer_rx_pipeline(rxSignal, params)
% TWO_LAYER_RX_PIPELINE - Complete RX for two-layer residual coding system
%
% Implements the receiver for:
%   Layer 1 (Base): DWT LL subband -> QPSK -> Strong FEC -> Priority SCs
%   Layer 2 (Enhancement): DCT residual -> 64-QAM -> Weak FEC -> Performance SCs
%
% Processing stages:
%   1. OFDM demodulation
%   2. Two-layer separation and demapping
%   3. (Optional) FEC decoding per layer
%   4. Base layer: Wavelet reconstruction
%   5. Enhancement layer: DCT residual reconstruction
%   6. Fusion: final = base + residual
%   7. Quality assessment and feedback
%
% Key property: If enhancement is lost, base still provides usable image.
%
% =========================================================================
% TODO: Confirm with TX team:
%   - FEC codes used for each layer
%   - Exact subcarrier allocation
%   - Wavelet and DCT parameters
%   - Synchronization/header format
% =========================================================================
%
% Inputs:
%   rxSignal - Received time-domain OFDM signal
%   params   - Configuration (see config_two_layer.m for full list)
%
% Outputs:
%   finalImage - Reconstructed image (uint8)
%   feedback   - Feedback structure for TX adaptation
%   results    - Detailed results from all stages

    %% Initialize
    results = struct();
    results.success = true;
    results.errors = {};
    results.warnings = {};
    startTime = tic;

    %% Apply defaults
    params = apply_defaults(params);

    if params.verbose
        print_header(params);
    end

    %% ==================== STAGE 1: OFDM DEMODULATION ====================
    if params.verbose
        fprintf('\n[1/8] OFDM Demodulation...\n');
    end

    try
        ofdmParams = struct();
        ofdmParams.nFFT = params.nFFT;
        ofdmParams.nCP = params.nCP;
        ofdmParams.equalizer = params.equalizer;
        ofdmParams.noiseVar = params.noiseVar;

        if isfield(params, 'pilotIndices') && isfield(params, 'pilotValues')
            ofdmParams.pilotIndices = params.pilotIndices;
            ofdmParams.pilotValues = params.pilotValues;
        end

        [rxSymbols, channelEst, demodMetrics] = ofdm_demodulator(rxSignal, ofdmParams);
        results.ofdmDemod = demodMetrics;
        results.channelEst = channelEst;

        if params.verbose
            fprintf('  ✓ Demodulated %d OFDM symbols\n', demodMetrics.nSymbols);
        end
    catch ME
        return_error(results, 'OFDM demodulation', ME);
        finalImage = []; feedback = [];
        return;
    end

    %% ==================== STAGE 2: TWO-LAYER DEMAPPING ====================
    if params.verbose
        fprintf('\n[2/8] Two-Layer Demapping...\n');
    end

    try
        % Create layer map
        layerMap = struct();
        layerMap.baseIndices = params.baseSubcarriers;
        layerMap.baseMod = params.baseMod;
        layerMap.enhancementIndices = params.enhancementSubcarriers;
        layerMap.enhancementMod = params.enhancementMod;
        layerMap.pilotIndices = params.pilotIndices;

        demapParams = struct();
        demapParams.noiseVar = params.noiseVar;

        [baseBits, enhancementBits, demapMetrics] = two_layer_demapper(rxSymbols, layerMap, demapParams);
        results.demapper = demapMetrics;

        if params.verbose
            fprintf('  ✓ Base layer: %d bits (EVM: %.1f%%, %s)\n', ...
                length(baseBits), demapMetrics.base.EVM_rms, demapMetrics.base.health);
            fprintf('  ✓ Enhancement: %d bits (EVM: %.1f%%, %s)\n', ...
                length(enhancementBits), demapMetrics.enhancement.EVM_rms, demapMetrics.enhancement.health);
        end
    catch ME
        return_error(results, 'Two-layer demapping', ME);
        finalImage = []; feedback = [];
        return;
    end

    %% ==================== STAGE 3: FEC DECODING (Optional) ====================
    if params.verbose
        fprintf('\n[3/8] FEC Decoding...\n');
    end

    if params.useFEC
        try
            [baseBits, baseFecOk] = decode_fec(baseBits, params.baseFEC, params);
            [enhancementBits, enhFecOk] = decode_fec(enhancementBits, params.enhancementFEC, params);

            results.fec.baseSuccess = baseFecOk;
            results.fec.enhancementSuccess = enhFecOk;

            if params.verbose
                fprintf('  ✓ Base FEC: %s\n', ternary(baseFecOk, 'OK', 'FAILED'));
                fprintf('  ✓ Enhancement FEC: %s\n', ternary(enhFecOk, 'OK', 'FAILED'));
            end
        catch ME
            results.warnings{end+1} = sprintf('FEC decoding warning: %s', ME.message);
            baseFecOk = true;
            enhFecOk = true;
        end
    else
        baseFecOk = demapMetrics.base.usable;
        enhFecOk = demapMetrics.enhancement.usable;

        if params.verbose
            fprintf('  (FEC disabled - using raw demapped bits)\n');
        end
    end

    %% ==================== STAGE 4: HEADER DECODING ====================
    if params.verbose
        fprintf('\n[4/8] Header Decoding...\n');
    end

    headerSize = header_codec('size');  % 128 bits
    headerValid = false;

    try
        if length(baseBits) >= headerSize
            headerBits = baseBits(1:headerSize);
            [headerParams, headerValid] = header_codec('decode', headerBits);

            if headerValid
                % Use header parameters (override input params where applicable)
                params.imageHeight = headerParams.imageHeight;
                params.imageWidth = headerParams.imageWidth;
                params.nChannels = headerParams.nChannels;
                params.waveletType = headerParams.waveletType;
                params.waveletLevels = headerParams.waveletLevels;
                params.baseBitsPerCoeff = headerParams.baseBitsPerCoeff;
                params.dctBlockSize = headerParams.dctBlockSize;
                params.maxDctCoeffs = headerParams.maxDctCoeffs;
                params.enhancementBitsPerCoeff = headerParams.enhancementBitsPerCoeff;
                params.residualScale = headerParams.residualScale;

                results.header = headerParams;

                if params.verbose
                    fprintf('  ✓ Header valid\n');
                    fprintf('  ✓ Image: %dx%dx%d\n', params.imageHeight, params.imageWidth, params.nChannels);
                    fprintf('  ✓ Wavelet: %s, %d levels\n', params.waveletType, params.waveletLevels);
                end
            else
                results.warnings{end+1} = 'Header CRC failed - using default params';
                if params.verbose
                    fprintf('  ⚠ Header CRC failed - using input params\n');
                end
            end

            % Remove header from base bits
            baseBits = baseBits(headerSize+1:end);
        else
            results.warnings{end+1} = 'Base bits too short for header';
            if params.verbose
                fprintf('  ⚠ No header found - using input params\n');
            end
        end
    catch ME
        results.warnings{end+1} = sprintf('Header decode warning: %s', ME.message);
        if params.verbose
            fprintf('  ⚠ Header error: %s\n', ME.message);
        end
    end

    results.headerValid = headerValid;

    %% ==================== STAGE 5: BASE LAYER DECODING ====================
    if params.verbose
        fprintf('\n[5/8] Base Layer (Wavelet) Decoding...\n');
    end

    try
        waveletParams = struct();
        waveletParams.imageHeight = params.imageHeight;
        waveletParams.imageWidth = params.imageWidth;
        waveletParams.nChannels = params.nChannels;
        waveletParams.waveletType = params.waveletType;
        waveletParams.waveletLevels = params.waveletLevels;
        waveletParams.bitsPerCoeff = params.baseBitsPerCoeff;

        [baseImage, baseInfo] = wavelet_base_decoder(baseBits, waveletParams);
        results.baseLayer = baseInfo;
        results.baseImage = baseImage;

        if params.verbose
            fprintf('  ✓ Reconstructed base: %dx%d (LL: %dx%d)\n', ...
                params.imageHeight, params.imageWidth, baseInfo.llHeight, baseInfo.llWidth);
        end
    catch ME
        results.warnings{end+1} = sprintf('Base decoding warning: %s', ME.message);
        baseImage = zeros(params.imageHeight, params.imageWidth, params.nChannels);
        baseFecOk = false;
    end

    %% ==================== STAGE 6: ENHANCEMENT LAYER DECODING ====================
    if params.verbose
        fprintf('\n[6/8] Enhancement Layer (DCT Residual) Decoding...\n');
    end

    try
        dctParams = struct();
        dctParams.imageHeight = params.imageHeight;
        dctParams.imageWidth = params.imageWidth;
        dctParams.nChannels = params.nChannels;
        dctParams.dctBlockSize = params.dctBlockSize;
        dctParams.coeffOrdering = params.coeffOrdering;
        dctParams.bitsPerCoeff = params.enhancementBitsPerCoeff;
        dctParams.maxCoeffsPerBlock = params.maxDctCoeffs;

        [residual, residualInfo] = residual_dct_decoder(enhancementBits, dctParams);
        results.enhancementLayer = residualInfo;
        results.residual = residual;

        if params.verbose
            fprintf('  ✓ Reconstructed residual (Std: %.2f, Energy: %.0f)\n', ...
                residualInfo.stdValue, residualInfo.energy);
        end
    catch ME
        results.warnings{end+1} = sprintf('Enhancement decoding warning: %s', ME.message);
        residual = zeros(params.imageHeight, params.imageWidth, params.nChannels);
        enhFecOk = false;
    end

    %% ==================== STAGE 7: LAYER FUSION ====================
    if params.verbose
        fprintf('\n[7/8] Layer Fusion...\n');
    end

    try
        fusionParams = struct();
        fusionParams.baseValid = baseFecOk;
        fusionParams.enhancementValid = enhFecOk;
        fusionParams.residualScale = params.residualScale;
        fusionParams.clipOutput = true;
        fusionParams.outputType = 'uint8';

        [finalImage, fusionInfo] = layer_fusion(baseImage, residual, fusionParams);
        results.fusion = fusionInfo;

        if params.verbose
            fprintf('  ✓ Fusion mode: %s\n', fusionInfo.fusionMode);
            fprintf('  ✓ Quality tier: %s\n', fusionInfo.qualityTier);
        end
    catch ME
        return_error(results, 'Layer fusion', ME);
        finalImage = uint8(baseImage);  % Fallback to base
        feedback = [];
        return;
    end

    %% ==================== STAGE 8: QUALITY & FEEDBACK ====================
    if params.verbose
        fprintf('\n[8/8] Quality Assessment & Feedback...\n');
    end

    try
        feedback = generate_two_layer_feedback(finalImage, baseImage, residual, ...
            results, params);
        results.feedback = feedback;

        if params.verbose
            if feedback.hasQualityMetrics
                fprintf('  ✓ SSIM: %.4f, PSNR: %.2f dB\n', feedback.ssim, feedback.psnr);
            end
            fprintf('  ✓ Base layer: %s\n', feedback.baseStatus);
            fprintf('  ✓ Enhancement: %s\n', feedback.enhancementStatus);
            if feedback.urgentAdapt
                fprintf('  ⚠ URGENT: %s\n', feedback.urgentReason);
            end
        end
    catch ME
        results.warnings{end+1} = sprintf('Feedback warning: %s', ME.message);
        feedback = struct('error', ME.message);
    end

    %% Finalize
    results.processingTime = toc(startTime);
    results.finalImageSize = size(finalImage);

    if params.verbose
        print_footer(results);
    end
end

%% Parameter Functions

function params = apply_defaults(params)
% APPLY_DEFAULTS - Set default parameter values

    % OFDM
    if ~isfield(params, 'nFFT'), error('nFFT required'); end
    if ~isfield(params, 'nCP'), error('nCP required'); end
    if ~isfield(params, 'equalizer'), params.equalizer = 'ZF'; end
    if ~isfield(params, 'noiseVar'), params.noiseVar = 0.01; end

    % Image
    if ~isfield(params, 'imageHeight'), error('imageHeight required'); end
    if ~isfield(params, 'imageWidth'), error('imageWidth required'); end
    if ~isfield(params, 'nChannels'), params.nChannels = 1; end

    % Base layer (Wavelet)
    if ~isfield(params, 'baseMod'), params.baseMod = 4; end  % QPSK
    if ~isfield(params, 'waveletType'), params.waveletType = 'haar'; end
    if ~isfield(params, 'waveletLevels'), params.waveletLevels = 2; end
    if ~isfield(params, 'baseBitsPerCoeff'), params.baseBitsPerCoeff = 8; end

    % Enhancement layer (DCT)
    if ~isfield(params, 'enhancementMod'), params.enhancementMod = 64; end  % 64-QAM
    if ~isfield(params, 'dctBlockSize'), params.dctBlockSize = 8; end
    if ~isfield(params, 'coeffOrdering'), params.coeffOrdering = 'zigzag'; end
    if ~isfield(params, 'enhancementBitsPerCoeff'), params.enhancementBitsPerCoeff = 8; end
    if ~isfield(params, 'maxDctCoeffs'), params.maxDctCoeffs = 64; end

    % Subcarrier allocation
    if ~isfield(params, 'baseSubcarriers')
        % Default: first 25% of data subcarriers
        nData = params.nFFT - 10;  % Rough estimate
        nBase = floor(nData * 0.25);
        params.baseSubcarriers = 5:(5 + nBase - 1);
    end
    if ~isfield(params, 'enhancementSubcarriers')
        params.enhancementSubcarriers = setdiff(5:(params.nFFT-5), params.baseSubcarriers);
    end
    if ~isfield(params, 'pilotIndices')
        params.pilotIndices = [];
    end

    % FEC
    if ~isfield(params, 'useFEC'), params.useFEC = false; end
    if ~isfield(params, 'baseFEC'), params.baseFEC = 'none'; end
    if ~isfield(params, 'enhancementFEC'), params.enhancementFEC = 'none'; end

    % Fusion
    if ~isfield(params, 'residualScale'), params.residualScale = 1.0; end

    % Quality targets
    if ~isfield(params, 'targetSSIM'), params.targetSSIM = 0.9; end
    if ~isfield(params, 'targetPSNR'), params.targetPSNR = 30; end

    % Display
    if ~isfield(params, 'verbose'), params.verbose = true; end
end

%% FEC Functions (Placeholder)

function [decodedBits, success] = decode_fec(bits, fecType, params)
% DECODE_FEC - FEC decoding placeholder
%
% TODO: Implement actual FEC decoding based on TX scheme
%   - Convolutional + Viterbi
%   - LDPC
%   - Turbo codes

    switch fecType
        case 'none'
            decodedBits = bits;
            success = true;

        case 'convolutional'
            % TODO: Implement Viterbi decoding
            decodedBits = bits;
            success = true;

        case 'ldpc'
            % TODO: Implement LDPC decoding
            decodedBits = bits;
            success = true;

        otherwise
            decodedBits = bits;
            success = true;
    end
end

%% Feedback Generation

function feedback = generate_two_layer_feedback(finalImage, baseImage, residual, results, params)
% GENERATE_TWO_LAYER_FEEDBACK - Create feedback for two-layer system

    feedback = struct();
    feedback.timestamp = now;

    % Quality metrics if reference available
    if isfield(params, 'refImage') && ~isempty(params.refImage)
        [feedback.ssim, feedback.psnr] = calculate_quality(finalImage, params.refImage);
        feedback.hasQualityMetrics = true;
        feedback.qualityMet = (feedback.ssim >= params.targetSSIM) && ...
                              (feedback.psnr >= params.targetPSNR);
    else
        feedback.ssim = NaN;
        feedback.psnr = NaN;
        feedback.hasQualityMetrics = false;
        feedback.qualityMet = false;
    end

    % Layer status
    dm = results.demapper;
    feedback.baseEVM = dm.base.EVM_rms;
    feedback.enhancementEVM = dm.enhancement.EVM_rms;
    feedback.baseStatus = dm.base.health;
    feedback.enhancementStatus = dm.enhancement.health;

    % Determine recommendations
    feedback.recommendations = struct();

    % Base layer recommendations
    if dm.base.EVM_rms > 20
        feedback.recommendations.baseAction = 'reduce_mod_or_increase_fec';
        feedback.recommendations.suggestedBaseMod = max(2, params.baseMod / 2);
    else
        feedback.recommendations.baseAction = 'maintain';
        feedback.recommendations.suggestedBaseMod = params.baseMod;
    end

    % Enhancement layer recommendations
    if dm.enhancement.EVM_rms > 30
        feedback.recommendations.enhAction = 'reduce_mod';
        feedback.recommendations.suggestedEnhMod = max(16, params.enhancementMod / 2);
    elseif dm.enhancement.EVM_rms < 15 && feedback.qualityMet
        feedback.recommendations.enhAction = 'can_increase_mod';
        feedback.recommendations.suggestedEnhMod = min(256, params.enhancementMod * 2);
    else
        feedback.recommendations.enhAction = 'maintain';
        feedback.recommendations.suggestedEnhMod = params.enhancementMod;
    end

    % Urgency assessment
    feedback.urgentAdapt = false;
    feedback.urgentReason = '';

    if strcmp(dm.base.health, 'poor')
        feedback.urgentAdapt = true;
        feedback.urgentReason = 'Base layer critical - image quality at risk';
    elseif feedback.hasQualityMetrics && feedback.ssim < 0.7
        feedback.urgentAdapt = true;
        feedback.urgentReason = 'SSIM below acceptable threshold';
    end

    % Encode feedback bits
    feedback.feedbackBits = encode_two_layer_feedback(feedback, params);
end

function [ssim_val, psnr_val] = calculate_quality(img1, img2)
% CALCULATE_QUALITY - Quick SSIM and PSNR calculation

    img1 = double(img1);
    img2 = double(img2);

    % PSNR
    mse = mean((img1(:) - img2(:)).^2);
    psnr_val = 10 * log10(255^2 / (mse + eps));

    % Simplified SSIM
    K1 = 0.01; K2 = 0.03; L = 255;
    C1 = (K1*L)^2; C2 = (K2*L)^2;

    if ndims(img1) == 3
        img1 = mean(img1, 3);
        img2 = mean(img2, 3);
    end

    mu1 = mean(img1(:)); mu2 = mean(img2(:));
    s1 = std(img1(:)); s2 = std(img2(:));
    s12 = mean((img1(:)-mu1).*(img2(:)-mu2));

    ssim_val = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1^2+s2^2+C2));
end

function bits = encode_two_layer_feedback(feedback, params)
% ENCODE_TWO_LAYER_FEEDBACK - Encode feedback to bits
%
% 16-bit format:
%   Bits 1-4:   SSIM (0-15)
%   Bits 5-6:   Base layer status (00=good, 01=marginal, 10=poor, 11=failed)
%   Bits 7-8:   Enhancement status
%   Bits 9-10:  Suggested base mod change (-1, 0, +1, +2)
%   Bits 11-12: Suggested enh mod change
%   Bit 13:     Urgent flag
%   Bits 14-16: Reserved

    bits = zeros(16, 1);

    % SSIM
    ssimQ = round(min(1, max(0, feedback.ssim)) * 15);
    bits(1:4) = de2bi(ssimQ, 4, 'left-msb')';

    % Base status
    baseCode = status_to_code(feedback.baseStatus);
    bits(5:6) = de2bi(baseCode, 2, 'left-msb')';

    % Enhancement status
    enhCode = status_to_code(feedback.enhancementStatus);
    bits(7:8) = de2bi(enhCode, 2, 'left-msb')';

    % Urgent
    bits(13) = feedback.urgentAdapt;
end

function code = status_to_code(status)
    switch status
        case 'good', code = 0;
        case 'marginal', code = 1;
        case 'poor', code = 2;
        otherwise, code = 3;
    end
end

%% Utility Functions

function val = ternary(cond, t, f)
    if cond, val = t; else, val = f; end
end

function return_error(results, stage, ME)
    results.success = false;
    results.errors{end+1} = sprintf('%s failed: %s', stage, ME.message);
end

function print_header(params)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║       Two-Layer RX Pipeline (Wavelet + DCT Residual)         ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\nConfiguration:\n');
    fprintf('  Image: %dx%d, FFT: %d, CP: %d\n', ...
        params.imageHeight, params.imageWidth, params.nFFT, params.nCP);
    fprintf('  Base: %s wavelet, %d levels, %d-QAM\n', ...
        params.waveletType, params.waveletLevels, params.baseMod);
    fprintf('  Enhancement: %dx%d DCT, %d-QAM\n', ...
        params.dctBlockSize, params.dctBlockSize, params.enhancementMod);
end

function print_footer(results)
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════════\n');
    fprintf('Pipeline complete in %.3f seconds\n', results.processingTime);
    if results.success
        fprintf('Status: SUCCESS (%s)\n', results.fusion.fusionMode);
    else
        fprintf('Status: FAILED\n');
        for i = 1:length(results.errors)
            fprintf('  Error: %s\n', results.errors{i});
        end
    end
    fprintf('════════════════════════════════════════════════════════════════\n');
end
