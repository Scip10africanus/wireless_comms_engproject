function [rxImage, feedback, results] = content_aware_rx_pipeline(rxSignal, params)
% CONTENT_AWARE_RX_PIPELINE - Complete RX for DCT-based UEP image transmission
%
% Full receiver chain for content-aware adaptive transmission:
%   1. OFDM demodulation with channel estimation
%   2. UEP-aware symbol demapping (mixed modulations)
%   3. Bitstream reorganization by importance
%   4. DCT coefficient extraction
%   5. IDCT image reconstruction
%   6. Quality assessment (SSIM/PSNR)
%   7. Adaptive feedback generation
%
% =========================================================================
% TODO: Key parameters to confirm with TX team:
%   - Subcarrier allocation (which SC carries which DCT band)
%   - Coefficient packing order
%   - Header format (if any)
%   - Pilot structure
%   - Feedback message format
% =========================================================================
%
% Inputs:
%   rxSignal - Received time-domain OFDM signal
%   params   - Configuration structure (see below)
%
% Outputs:
%   rxImage  - Reconstructed image
%   feedback - Feedback structure for TX adaptation
%   results  - Detailed results from all processing stages

    %% Initialize
    results = struct();
    results.success = true;
    results.errors = {};
    results.warnings = {};
    results.stages = struct();
    startTime = tic;

    %% Validate and apply defaults
    params = validate_and_default_params(params);

    if params.verbose
        fprintf('\n');
        fprintf('╔══════════════════════════════════════════════════════════╗\n');
        fprintf('║     Content-Aware RX Pipeline (DCT-based UEP)            ║\n');
        fprintf('╚══════════════════════════════════════════════════════════╝\n');
        fprintf('\nConfiguration:\n');
        fprintf('  FFT: %d, CP: %d, Image: %dx%d\n', ...
            params.nFFT, params.nCP, params.imageHeight, params.imageWidth);
        fprintf('  Modulations: Critical=%d, Normal=%d, NonCrit=%d\n', ...
            params.criticalMod, params.normalMod, params.noncriticalMod);
    end

    %% Stage 1: OFDM Demodulation
    if params.verbose
        fprintf('\n[1/6] OFDM Demodulation...\n');
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
        results.stages.ofdmDemod = demodMetrics;
        results.channelEst = channelEst;

        if params.verbose
            fprintf('  ✓ Demodulated %d OFDM symbols\n', demodMetrics.nSymbols);
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('OFDM demod failed: %s', ME.message);
        rxImage = []; feedback = [];
        return;
    end

    %% Stage 2: Generate/Load Subcarrier Map
    if params.verbose
        fprintf('\n[2/6] Subcarrier Mapping...\n');
    end

    if isfield(params, 'subcarrierMap')
        scMap = params.subcarrierMap;
    else
        % Generate default map
        scConfig = struct();
        scConfig.criticalRatio = params.criticalRatio;
        scConfig.normalRatio = params.normalRatio;
        scConfig.criticalMod = params.criticalMod;
        scConfig.normalMod = params.normalMod;
        scConfig.noncriticalMod = params.noncriticalMod;
        scConfig.mappingType = params.mappingType;

        scMap = subcarrier_mapper(params.nFFT, scConfig);
    end
    results.subcarrierMap = scMap;

    if params.verbose
        fprintf('  ✓ %d data SC: %d critical, %d normal, %d non-critical\n', ...
            scMap.summary.nData, scMap.summary.nCritical, ...
            scMap.summary.nNormal, scMap.summary.nNoncritical);
    end

    %% Stage 3: UEP Symbol Demapping
    if params.verbose
        fprintf('\n[3/6] UEP Demapping...\n');
    end

    try
        demapParams = struct();
        demapParams.softOutput = params.softDemap;
        demapParams.noiseVar = params.noiseVar;

        [demappedBits, demapMetrics] = uep_demapper(rxSymbols, scMap, demapParams);
        results.stages.demapper = demapMetrics;
        results.demappedBits = demappedBits;

        if params.verbose
            fprintf('  ✓ Demapped %d bits total\n', demapMetrics.totalBits);
            fprintf('    Critical: %d bits (EVM: %.1f%%)\n', ...
                demapMetrics.critical.count, demapMetrics.critical.avgEVM);
            fprintf('    Non-critical: %d bits (EVM: %.1f%%)\n', ...
                demapMetrics.noncritical.count, demapMetrics.noncritical.avgEVM);
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('UEP demapping failed: %s', ME.message);
        rxImage = []; feedback = [];
        return;
    end

    %% Stage 4: Reconstruct DCT Coefficients from Bits
    if params.verbose
        fprintf('\n[4/6] Bitstream to DCT Coefficients...\n');
    end

    try
        % TODO: This conversion depends heavily on TX packing format
        % Current assumption: bits directly map to 8-bit coefficient values
        dctCoeffs = bits_to_dct_coefficients(demappedBits, params);
        results.dctCoeffs = dctCoeffs;

        if params.verbose
            fprintf('  ✓ Extracted %d DCT coefficients\n', length(dctCoeffs));
        end
    catch ME
        results.warnings{end+1} = sprintf('DCT extraction warning: %s', ME.message);
        % Try to continue with what we have
        dctCoeffs = [];
    end

    %% Stage 5: IDCT Image Reconstruction
    if params.verbose
        fprintf('\n[5/6] Image Reconstruction (IDCT)...\n');
    end

    try
        if ~isempty(dctCoeffs)
            dctParams = struct();
            dctParams.imageHeight = params.imageHeight;
            dctParams.imageWidth = params.imageWidth;
            dctParams.nChannels = params.nChannels;
            dctParams.blockSize = params.dctBlockSize;
            dctParams.coeffOrdering = params.coeffOrdering;
            dctParams.freqThreshold = params.freqThreshold;

            [rxImage, coeffInfo] = dct_reconstructor(dctCoeffs, dctParams);
            results.stages.dctRecon = coeffInfo;
        else
            % Fallback: try direct bit-to-pixel (for testing without DCT)
            rxImage = bits_to_image_direct(demappedBits.all, params);
            results.stages.dctRecon = struct('fallback', true);
        end

        if params.verbose
            fprintf('  ✓ Reconstructed %dx%d image\n', size(rxImage,1), size(rxImage,2));
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('Image reconstruction failed: %s', ME.message);
        rxImage = []; feedback = [];
        return;
    end

    %% Stage 6: Quality Assessment & Feedback Generation
    if params.verbose
        fprintf('\n[6/6] Quality Assessment & Feedback...\n');
    end

    try
        % Prepare metrics for feedback
        rxMetrics = struct();
        rxMetrics.demapMetrics = demapMetrics;
        if isfield(results.stages, 'dctRecon')
            rxMetrics.coeffInfo = results.stages.dctRecon;
        end

        % Generate feedback
        feedbackParams = struct();
        feedbackParams.targetSSIM = params.targetSSIM;
        feedbackParams.targetPSNR = params.targetPSNR;
        feedbackParams.currentThreshold = params.freqThreshold;
        feedbackParams.currentMods = [params.criticalMod, params.normalMod, params.noncriticalMod];

        refImage = [];
        if isfield(params, 'refImage')
            refImage = params.refImage;
        end

        feedback = content_aware_feedback(rxImage, refImage, rxMetrics, feedbackParams);
        results.feedback = feedback;

        if params.verbose
            if feedback.hasQualityMetrics
                fprintf('  ✓ SSIM: %.4f, PSNR: %.2f dB\n', feedback.ssim, feedback.psnr);
                fprintf('    Status: %s\n', feedback.diagnostics.qualityStatus);
            end
            fprintf('  ✓ Threshold recommendation: %+d\n', feedback.thresholdDelta);
            fprintf('  ✓ Mod recommendation: [%d, %d, %d]\n', ...
                feedback.modRecommend(1), feedback.modRecommend(2), feedback.modRecommend(3));
            if feedback.urgentAdapt
                fprintf('  ⚠ URGENT ADAPTATION NEEDED\n');
            end
        end
    catch ME
        results.warnings{end+1} = sprintf('Feedback generation warning: %s', ME.message);
        feedback = struct('error', ME.message);
    end

    %% Finalize
    results.processingTime = toc(startTime);

    if params.verbose
        fprintf('\n');
        fprintf('════════════════════════════════════════════════════════════\n');
        fprintf('Pipeline complete in %.3f seconds\n', results.processingTime);
        fprintf('Errors: %d, Warnings: %d\n', length(results.errors), length(results.warnings));
        fprintf('════════════════════════════════════════════════════════════\n');
    end
end

%% Parameter Validation

function params = validate_and_default_params(params)
% VALIDATE_AND_DEFAULT_PARAMS - Set all default values

    % Required OFDM parameters
    if ~isfield(params, 'nFFT')
        error('nFFT is required');
    end
    if ~isfield(params, 'nCP')
        error('nCP is required');
    end

    % Image parameters
    if ~isfield(params, 'imageHeight')
        error('imageHeight is required');
    end
    if ~isfield(params, 'imageWidth')
        error('imageWidth is required');
    end
    if ~isfield(params, 'nChannels')
        params.nChannels = 1;
    end

    % OFDM defaults
    if ~isfield(params, 'equalizer')
        params.equalizer = 'ZF';
    end
    if ~isfield(params, 'noiseVar')
        params.noiseVar = 0.01;
    end

    % UEP modulation defaults
    if ~isfield(params, 'criticalMod')
        params.criticalMod = 4;      % QPSK
    end
    if ~isfield(params, 'normalMod')
        params.normalMod = 16;       % 16-QAM
    end
    if ~isfield(params, 'noncriticalMod')
        params.noncriticalMod = 64;  % 64-QAM
    end

    % Subcarrier allocation defaults
    if ~isfield(params, 'criticalRatio')
        params.criticalRatio = 0.3;
    end
    if ~isfield(params, 'normalRatio')
        params.normalRatio = 0.2;
    end
    if ~isfield(params, 'mappingType')
        params.mappingType = 'fixed';
    end

    % DCT defaults
    if ~isfield(params, 'dctBlockSize')
        params.dctBlockSize = 8;
    end
    if ~isfield(params, 'coeffOrdering')
        params.coeffOrdering = 'zigzag';
    end
    if ~isfield(params, 'freqThreshold')
        params.freqThreshold = 21;  % ~1/3 of 64 coeffs in 8x8 block
    end

    % Quality targets
    if ~isfield(params, 'targetSSIM')
        params.targetSSIM = 0.9;
    end
    if ~isfield(params, 'targetPSNR')
        params.targetPSNR = 30;
    end

    % Processing options
    if ~isfield(params, 'softDemap')
        params.softDemap = false;
    end
    if ~isfield(params, 'verbose')
        params.verbose = false;
    end
end

%% Bit Conversion Functions

function dctCoeffs = bits_to_dct_coefficients(demappedBits, params)
% BITS_TO_DCT_COEFFICIENTS - Convert demapped bits to DCT coefficients
%
% TODO: Adapt based on TX coefficient encoding format
%
% Current assumptions:
%   - 8 bits per coefficient (signed, -128 to 127 mapped to 0-255)
%   - Critical bits = low-freq coefficients
%   - Non-critical bits = high-freq coefficients

    % Use the organized bits (critical first, then others)
    allBits = [demappedBits.critical; demappedBits.normal; demappedBits.noncritical];

    % Calculate expected coefficients
    nBlocks = ceil(params.imageHeight / params.dctBlockSize) * ...
              ceil(params.imageWidth / params.dctBlockSize);
    coeffsPerBlock = params.dctBlockSize^2;
    totalCoeffs = nBlocks * coeffsPerBlock * params.nChannels;

    bitsPerCoeff = 8;  % TODO: Confirm with TX team
    expectedBits = totalCoeffs * bitsPerCoeff;

    % Pad or truncate
    if length(allBits) < expectedBits
        allBits = [allBits; zeros(expectedBits - length(allBits), 1)];
    end
    allBits = allBits(1:expectedBits);

    % Convert bits to coefficients
    dctCoeffs = zeros(totalCoeffs, 1);
    for i = 1:totalCoeffs
        startBit = (i-1) * bitsPerCoeff + 1;
        endBit = i * bitsPerCoeff;
        coeffBits = allBits(startBit:endBit);

        % Convert to integer (MSB first)
        val = 0;
        for b = 1:bitsPerCoeff
            val = val * 2 + coeffBits(b);
        end

        % Convert from unsigned to signed
        % TODO: Confirm encoding with TX team
        dctCoeffs(i) = val - 128;  % Assume centered at 128
    end
end

function image = bits_to_image_direct(bits, params)
% BITS_TO_IMAGE_DIRECT - Fallback: direct bits to pixels (no DCT)
%
% Used for testing when DCT path has issues

    nPixels = params.imageHeight * params.imageWidth * params.nChannels;
    bitsPerPixel = 8;
    expectedBits = nPixels * bitsPerPixel;

    if length(bits) < expectedBits
        bits = [bits; zeros(expectedBits - length(bits), 1)];
    end
    bits = bits(1:expectedBits);

    pixels = zeros(nPixels, 1);
    for i = 1:nPixels
        startBit = (i-1) * bitsPerPixel + 1;
        pixelBits = bits(startBit:startBit+bitsPerPixel-1);
        val = 0;
        for b = 1:bitsPerPixel
            val = val * 2 + pixelBits(b);
        end
        pixels(i) = val;
    end

    if params.nChannels == 1
        image = uint8(reshape(pixels, [params.imageWidth, params.imageHeight])');
    else
        image = uint8(reshape(pixels, [params.imageWidth, params.imageHeight, params.nChannels]));
        image = permute(image, [2, 1, 3]);
    end
end
