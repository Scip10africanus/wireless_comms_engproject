function [rxImage, results] = rx_pipeline(rxSignal, params)
% RX_PIPELINE - Complete RX-side signal processing pipeline for image transmission
%
% Integrates all RX processing stages:
%   1. OFDM demodulation with channel estimation
%   2. Symbol demapping
%   3. Bitstream unpacking
%   4. Image reconstruction
%   5. Quality assessment
%   6. Feedback generation
%
% Inputs:
%   rxSignal - Received time-domain OFDM signal (column vector)
%   params   - Configuration structure with fields:
%       % OFDM parameters
%       .nFFT          - FFT size (required)
%       .nCP           - Cyclic prefix length (required)
%       .nSymbols      - Number of OFDM symbols (auto-detected if not provided)
%       .pilotIndices  - Pilot subcarrier indices (optional)
%       .pilotValues   - Known pilot values (optional)
%       .equalizer     - 'ZF' or 'MMSE' (default: 'ZF')
%       .noiseVar      - Noise variance for MMSE (default: 0.01)
%
%       % Modulation parameters
%       .modOrder      - Modulation order (required)
%       .softDemap     - Use soft demapping (default: false)
%
%       % Image parameters
%       .imageHeight   - Image height (required if no header in bitstream)
%       .imageWidth    - Image width (required if no header in bitstream)
%       .nChannels     - Number of color channels (default: 1)
%       .bitDepth      - Bits per pixel (default: 8)
%       .hasHeader     - Bitstream includes header (default: true)
%
%       % Quality assessment
%       .refImage      - Reference image for quality metrics (optional)
%       .txBits        - Original transmitted bits for BER (optional)
%
%       % Feedback
%       .generateFeedback - Generate feedback for TX (default: true)
%       .targetPSNR    - Target PSNR (default: 30 dB)
%       .targetSSIM    - Target SSIM (default: 0.9)
%
%       % Debug/Display
%       .verbose       - Print progress (default: false)
%       .plotResults   - Generate plots (default: false)
%
% Outputs:
%   rxImage - Reconstructed image (uint8)
%   results - Structure containing all intermediate results and metrics

    %% Initialize results structure
    results = struct();
    results.success = true;
    results.errors = {};
    results.warnings = {};
    startTime = tic;

    %% Validate and set default parameters
    params = validate_params(params);

    if params.verbose
        fprintf('\n========== RX Pipeline Started ==========\n');
        fprintf('Signal length: %d samples\n', length(rxSignal));
    end

    %% Stage 1: OFDM Demodulation
    if params.verbose
        fprintf('\n[Stage 1] OFDM Demodulation...\n');
    end

    try
        ofdmParams = struct();
        ofdmParams.nFFT = params.nFFT;
        ofdmParams.nCP = params.nCP;
        ofdmParams.equalizer = params.equalizer;
        ofdmParams.noiseVar = params.noiseVar;

        if isfield(params, 'pilotIndices')
            ofdmParams.pilotIndices = params.pilotIndices;
            ofdmParams.pilotValues = params.pilotValues;
        end

        [rxSymbols, channelEst, demodMetrics] = ofdm_demodulator(rxSignal, ofdmParams);
        results.ofdmDemod = demodMetrics;
        results.channelEstimate = channelEst;

        if params.verbose
            fprintf('  Demodulated %d OFDM symbols\n', demodMetrics.nSymbols);
            fprintf('  Channel gain: %.2f dB\n', demodMetrics.channelGain_dB);
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('OFDM demodulation failed: %s', ME.message);
        rxImage = [];
        return;
    end

    %% Stage 2: Symbol Demapping
    if params.verbose
        fprintf('\n[Stage 2] Symbol Demapping...\n');
    end

    try
        demapParams = struct();
        demapParams.modOrder = params.modOrder;
        demapParams.outputType = ternary(params.softDemap, 'soft', 'hard');
        demapParams.noiseVar = params.noiseVar;

        [rxBits, softBits, demapMetrics] = symbol_demapper(rxSymbols, demapParams);
        results.symbolDemap = demapMetrics;
        results.rxBits = rxBits;

        if params.verbose
            fprintf('  Demapped %d bits (EVM: %.2f%%)\n', demapMetrics.totalBits, demapMetrics.EVM_rms);
        end

        % Calculate BER if reference bits provided
        if isfield(params, 'txBits') && ~isempty(params.txBits)
            minLen = min(length(params.txBits), length(rxBits));
            [numErrors, ber] = biterr(params.txBits(1:minLen), rxBits(1:minLen));
            results.BER = ber;
            results.numBitErrors = numErrors;

            if params.verbose
                fprintf('  BER: %.6f (%d errors)\n', ber, numErrors);
            end
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('Symbol demapping failed: %s', ME.message);
        rxImage = [];
        return;
    end

    %% Stage 3: Bitstream Unpacking & Image Reconstruction
    if params.verbose
        fprintf('\n[Stage 3] Image Reconstruction...\n');
    end

    try
        unpackParams = struct();
        unpackParams.hasHeader = params.hasHeader;
        unpackParams.bitDepth = params.bitDepth;

        if ~params.hasHeader
            unpackParams.imageHeight = params.imageHeight;
            unpackParams.imageWidth = params.imageWidth;
            unpackParams.nChannels = params.nChannels;
        end

        [rxImage, imageMetadata, unpackStatus] = bitstream_unpacker(rxBits, unpackParams);
        results.imageMetadata = imageMetadata;
        results.unpackStatus = unpackStatus;

        if ~isempty(unpackStatus.warnings)
            results.warnings = [results.warnings, unpackStatus.warnings];
        end

        if params.verbose
            fprintf('  Reconstructed image: %dx%dx%d\n', ...
                imageMetadata.height, imageMetadata.width, imageMetadata.nChannels);
        end
    catch ME
        results.success = false;
        results.errors{end+1} = sprintf('Image reconstruction failed: %s', ME.message);
        rxImage = [];
        return;
    end

    %% Stage 4: Image Quality Assessment
    if params.verbose
        fprintf('\n[Stage 4] Quality Assessment...\n');
    end

    try
        if isfield(params, 'refImage') && ~isempty(params.refImage)
            qualParams = struct();
            qualParams.maxVal = 255;
            qualParams.verbose = false;

            qualityMetrics = image_quality_metrics(params.refImage, rxImage, qualParams);
            results.qualityMetrics = qualityMetrics;

            if params.verbose
                fprintf('  PSNR: %.2f dB\n', qualityMetrics.PSNR);
                fprintf('  SSIM: %.4f\n', qualityMetrics.SSIM);
                fprintf('  MS-SSIM: %.4f\n', qualityMetrics.MSSSIM);
            end
        else
            results.qualityMetrics = [];
            if params.verbose
                fprintf('  (No reference image provided - skipping quality metrics)\n');
            end
        end
    catch ME
        results.warnings{end+1} = sprintf('Quality assessment warning: %s', ME.message);
        results.qualityMetrics = [];
    end

    %% Stage 5: Generate Feedback
    if params.generateFeedback
        if params.verbose
            fprintf('\n[Stage 5] Generating Feedback...\n');
        end

        try
            % Combine channel metrics
            channelMetrics = struct();
            channelMetrics.EVM = demapMetrics.EVM_rms;
            channelMetrics.MER_dB = demapMetrics.MER_dB;
            if isfield(results, 'BER')
                channelMetrics.BER = results.BER;
            end

            feedbackParams = struct();
            feedbackParams.targetPSNR = params.targetPSNR;
            feedbackParams.targetSSIM = params.targetSSIM;
            if isfield(params, 'refImage')
                feedbackParams.refImage = params.refImage;
            end

            feedback = quality_feedback(rxImage, channelMetrics, feedbackParams);
            results.feedback = feedback;

            if params.verbose
                fprintf('  CQI: %d/15\n', feedback.CQI);
                fprintf('  Quality Indicator: %.2f\n', feedback.QI);
                fprintf('  Recommended: %d-QAM, rate %.2f\n', ...
                    feedback.recModOrder, feedback.recCodingRate);
                if feedback.urgent
                    fprintf('  *** URGENT adaptation recommended ***\n');
                end
            end
        catch ME
            results.warnings{end+1} = sprintf('Feedback generation warning: %s', ME.message);
            results.feedback = [];
        end
    end

    %% Finalize
    results.processingTime = toc(startTime);

    if params.verbose
        fprintf('\n========== RX Pipeline Complete ==========\n');
        fprintf('Total processing time: %.3f seconds\n', results.processingTime);
        fprintf('Warnings: %d, Errors: %d\n', length(results.warnings), length(results.errors));
    end

    %% Generate plots if requested
    if params.plotResults
        plot_rx_results(rxImage, rxSymbols, results, params);
    end
end

%% Helper Functions

function params = validate_params(params)
% VALIDATE_PARAMS - Validate and set default parameter values

    % Required parameters
    if ~isfield(params, 'nFFT')
        error('nFFT is required');
    end
    if ~isfield(params, 'nCP')
        error('nCP is required');
    end
    if ~isfield(params, 'modOrder')
        error('modOrder is required');
    end

    % OFDM defaults
    if ~isfield(params, 'equalizer')
        params.equalizer = 'ZF';
    end
    if ~isfield(params, 'noiseVar')
        params.noiseVar = 0.01;
    end

    % Modulation defaults
    if ~isfield(params, 'softDemap')
        params.softDemap = false;
    end

    % Image defaults
    if ~isfield(params, 'nChannels')
        params.nChannels = 1;
    end
    if ~isfield(params, 'bitDepth')
        params.bitDepth = 8;
    end
    if ~isfield(params, 'hasHeader')
        params.hasHeader = true;
    end

    % Quality defaults
    if ~isfield(params, 'targetPSNR')
        params.targetPSNR = 30;
    end
    if ~isfield(params, 'targetSSIM')
        params.targetSSIM = 0.9;
    end

    % Feedback defaults
    if ~isfield(params, 'generateFeedback')
        params.generateFeedback = true;
    end

    % Display defaults
    if ~isfield(params, 'verbose')
        params.verbose = false;
    end
    if ~isfield(params, 'plotResults')
        params.plotResults = false;
    end
end

function result = ternary(condition, trueVal, falseVal)
% TERNARY - Simple ternary operator
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

function plot_rx_results(rxImage, rxSymbols, results, params)
% PLOT_RX_RESULTS - Generate visualization of RX results

    figure('Position', [100, 100, 1400, 800], 'Name', 'RX Pipeline Results');

    % Subplot 1: Received constellation
    subplot(2, 3, 1);
    plot(real(rxSymbols(:)), imag(rxSymbols(:)), 'b.', 'MarkerSize', 2);
    grid on;
    axis equal;
    title(sprintf('Received Constellation (EVM: %.1f%%)', results.symbolDemap.EVM_rms));
    xlabel('In-Phase');
    ylabel('Quadrature');

    % Subplot 2: Channel frequency response
    subplot(2, 3, 2);
    if ~isempty(results.channelEstimate)
        H_mag = 20*log10(abs(results.channelEstimate(:,1)) + 1e-10);
        plot(H_mag, 'b-', 'LineWidth', 1.5);
        grid on;
        title('Channel Frequency Response');
        xlabel('Subcarrier Index');
        ylabel('Magnitude (dB)');
    end

    % Subplot 3: Reconstructed image
    subplot(2, 3, 3);
    if size(rxImage, 3) == 1
        imshow(rxImage, []);
    else
        imshow(rxImage);
    end
    title('Reconstructed Image');

    % Subplot 4: Reference image (if available)
    subplot(2, 3, 4);
    if isfield(params, 'refImage') && ~isempty(params.refImage)
        if size(params.refImage, 3) == 1
            imshow(params.refImage, []);
        else
            imshow(params.refImage);
        end
        title('Reference Image');
    else
        text(0.5, 0.5, 'No Reference', 'HorizontalAlignment', 'center');
        axis off;
    end

    % Subplot 5: Quality metrics bar chart
    subplot(2, 3, 5);
    if ~isempty(results.qualityMetrics)
        metrics = [results.qualityMetrics.PSNR/50, ...  % Normalize PSNR
                   results.qualityMetrics.SSIM, ...
                   results.qualityMetrics.MSSSIM, ...
                   results.qualityMetrics.Correlation];
        bar(metrics);
        set(gca, 'XTickLabel', {'PSNR/50', 'SSIM', 'MS-SSIM', 'Corr'});
        ylim([0, 1.2]);
        title('Quality Metrics (Normalized)');
        grid on;
    else
        text(0.5, 0.5, 'No Quality Metrics', 'HorizontalAlignment', 'center');
        axis off;
    end

    % Subplot 6: Summary text
    subplot(2, 3, 6);
    axis off;
    summaryText = {
        sprintf('Processing Time: %.3f s', results.processingTime),
        sprintf('Image Size: %dx%d', results.imageMetadata.height, results.imageMetadata.width),
        sprintf('Modulation: %d-QAM', params.modOrder),
        ''
    };

    if isfield(results, 'BER')
        summaryText{end+1} = sprintf('BER: %.2e', results.BER);
    end

    if ~isempty(results.qualityMetrics)
        summaryText{end+1} = sprintf('PSNR: %.2f dB', results.qualityMetrics.PSNR);
        summaryText{end+1} = sprintf('SSIM: %.4f', results.qualityMetrics.SSIM);
    end

    if isfield(results, 'feedback') && ~isempty(results.feedback)
        summaryText{end+1} = '';
        summaryText{end+1} = sprintf('CQI: %d/15', results.feedback.CQI);
        summaryText{end+1} = sprintf('Recommended: %d-QAM', results.feedback.recModOrder);
    end

    text(0.1, 0.9, summaryText, 'VerticalAlignment', 'top', 'FontSize', 10, ...
         'FontName', 'FixedWidth');
    title('Summary');
end
