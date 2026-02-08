%% Configuration Template for Content-Aware RX Pipeline
%
% This file contains ALL configurable parameters for the RX system.
% Use this as a reference and starting point for integration.
%
% =========================================================================
%                    *** MEETING CHECKLIST ***
%
% Questions to ask the TX team:
%
% OFDM Structure:
%   [ ] FFT size (64? 128? 256?)
%   [ ] Cyclic prefix length
%   [ ] Pilot positions and values
%   [ ] Guard band sizes
%
% DCT Configuration:
%   [ ] Block size (8x8? 16x16?)
%   [ ] Coefficient ordering (zigzag? raster? custom?)
%   [ ] Frequency threshold for low/mid/high split
%   [ ] Quantization table (if used)
%   [ ] Bits per coefficient
%
% UEP Mapping:
%   [ ] Which subcarriers carry critical data?
%   [ ] Modulation order per band (QPSK/16-QAM/64-QAM?)
%   [ ] Is mapping static or adaptive?
%   [ ] Any interleaving?
%
% Feedback Channel:
%   [ ] How many bits for feedback?
%   [ ] Update rate (every frame? every N frames?)
%   [ ] What parameters can TX adjust?
%
% Data Format:
%   [ ] Header format (if any)
%   [ ] How are bits packed?
%   [ ] Byte ordering (MSB/LSB first)
% =========================================================================

function params = config_template()
% CONFIG_TEMPLATE - Returns default configuration structure
%
% Usage:
%   params = config_template();
%   % Modify as needed...
%   params.nFFT = 128;
%   [rxImage, feedback, results] = content_aware_rx_pipeline(rxSignal, params);

    params = struct();

    %% ==================== OFDM PARAMETERS ====================
    % TODO: Confirm with TX team

    params.nFFT = 64;               % FFT size (subcarriers)
    params.nCP = 16;                % Cyclic prefix length

    % Channel estimation
    params.equalizer = 'ZF';        % 'ZF' (zero-forcing) or 'MMSE'
    params.noiseVar = 0.01;         % Noise variance (for MMSE)

    % Pilot configuration
    % TODO: Get exact pilot positions from TX
    params.pilotSpacing = 8;        % Pilot every N subcarriers
    params.pilotIndices = 1:params.pilotSpacing:params.nFFT;
    params.pilotValues = ones(length(params.pilotIndices), 1);

    %% ==================== IMAGE PARAMETERS ====================
    % TODO: Confirm image specs

    params.imageHeight = 64;        % Image height in pixels
    params.imageWidth = 64;         % Image width in pixels
    params.nChannels = 1;           % 1 = grayscale, 3 = RGB

    %% ==================== DCT PARAMETERS ====================
    % TODO: Confirm with TX team

    params.dctBlockSize = 8;        % DCT block size (8 = standard JPEG)

    % Coefficient ordering
    %   'zigzag' - Standard JPEG zigzag scan
    %   'raster' - Row-by-row
    %   'custom' - Define your own (modify dct_reconstructor.m)
    params.coeffOrdering = 'zigzag';

    % Frequency threshold
    % Coefficients 1 to freqThreshold are "low frequency" (critical)
    % For 8x8 block (64 coeffs):
    %   21 = roughly upper-left triangle (1/3)
    %   10 = DC + first AC coefficients (very critical only)
    %   32 = half the coefficients
    params.freqThreshold = 21;      % TODO: Tune based on image content

    % Quantization (if used)
    % params.quantTable = jpeg_qtable();  % Uncomment if TX uses quantization

    % Bits per DCT coefficient
    params.bitsPerCoeff = 8;        % TODO: Confirm with TX

    %% ==================== UEP / MODULATION PARAMETERS ====================
    % TODO: Confirm modulation scheme with TX team

    % Modulation orders per importance level
    params.criticalMod = 4;         % QPSK for low-freq DCT (most protected)
    params.normalMod = 16;          % 16-QAM for mid-freq DCT
    params.noncriticalMod = 64;     % 64-QAM for high-freq DCT (least protected)

    % Subcarrier allocation ratios
    % What fraction of data subcarriers for each band?
    params.criticalRatio = 0.3;     % 30% for critical
    params.normalRatio = 0.2;       % 20% for normal
    % Remaining 50% for non-critical

    % Mapping type
    %   'fixed'            - Static allocation
    %   'frequency_ordered'- Lower SC index = critical
    %   'adaptive'         - Changes based on channel (TODO)
    %   'custom'           - Define your own in subcarrier_mapper.m
    params.mappingType = 'fixed';

    % Or provide explicit subcarrier map
    % params.subcarrierMap = your_custom_map;

    %% ==================== QUALITY TARGETS ====================
    % Used for feedback generation

    params.targetSSIM = 0.90;       % Target SSIM (0-1)
    params.targetPSNR = 30;         % Target PSNR in dB

    % Urgency thresholds
    params.ssimUrgentThresh = 0.70; % Below this = urgent adaptation
    params.evmUrgentThresh = 25;    % Above this % = urgent adaptation

    %% ==================== FEEDBACK PARAMETERS ====================
    % TODO: Confirm feedback format with TX team

    params.feedbackBits = 16;       % Total bits for feedback message

    % What can the feedback control?
    params.canAdjustThreshold = true;   % Can recommend DCT threshold change?
    params.canAdjustMods = true;        % Can recommend modulation changes?
    params.canRequestRetx = false;      % Can request retransmission?

    %% ==================== PROCESSING OPTIONS ====================

    params.softDemap = false;       % Use soft demapping (LLRs)? (for FEC)
    params.verbose = true;          % Print progress messages
    params.plotResults = false;     % Generate plots

    %% ==================== REFERENCE IMAGE (FOR TESTING) ====================
    % For quality calculation during development
    % In real deployment, you may not have this

    % params.refImage = imread('test_image.png');

end

%% ==================== PRESET CONFIGURATIONS ====================

function params = config_low_snr()
% CONFIG_LOW_SNR - Settings optimized for poor channel conditions

    params = config_template();

    % More protection
    params.criticalMod = 2;         % BPSK
    params.normalMod = 4;           % QPSK
    params.noncriticalMod = 16;     % 16-QAM

    % Larger critical region
    params.criticalRatio = 0.5;
    params.normalRatio = 0.3;

    % Lower thresholds
    params.targetSSIM = 0.80;
    params.targetPSNR = 25;
end

function params = config_high_snr()
% CONFIG_HIGH_SNR - Settings for good channel conditions (high throughput)

    params = config_template();

    % Less protection, higher throughput
    params.criticalMod = 16;        % 16-QAM
    params.normalMod = 64;          % 64-QAM
    params.noncriticalMod = 256;    % 256-QAM

    % Smaller critical region
    params.criticalRatio = 0.2;
    params.normalRatio = 0.2;

    % Higher quality targets
    params.targetSSIM = 0.95;
    params.targetPSNR = 40;
end

function params = config_test_no_dct()
% CONFIG_TEST_NO_DCT - For testing without DCT (raw pixels)
%
% Use this to test the OFDM/demapping chain independently

    params = config_template();

    % Disable DCT-specific processing
    params.useDCT = false;          % Flag for pipeline to skip DCT
    params.bitsPerPixel = 8;
end

%% ==================== UTILITY FUNCTIONS ====================

function qtable = jpeg_qtable()
% JPEG_QTABLE - Standard JPEG luminance quantization table

    qtable = [
        16  11  10  16  24  40  51  61;
        12  12  14  19  26  58  60  55;
        14  13  16  24  40  57  69  56;
        14  17  22  29  51  87  80  62;
        18  22  37  56  68 109 103  77;
        24  35  55  64  81 104 113  92;
        49  64  78  87 103 121 120 101;
        72  92  95  98 112 100 103  99
    ];
end

function print_config(params)
% PRINT_CONFIG - Display current configuration

    fprintf('\n========== Current Configuration ==========\n\n');

    fprintf('OFDM:\n');
    fprintf('  FFT Size: %d\n', params.nFFT);
    fprintf('  CP Length: %d\n', params.nCP);
    fprintf('  Equalizer: %s\n', params.equalizer);

    fprintf('\nImage:\n');
    fprintf('  Dimensions: %dx%dx%d\n', params.imageHeight, params.imageWidth, params.nChannels);

    fprintf('\nDCT:\n');
    fprintf('  Block Size: %d\n', params.dctBlockSize);
    fprintf('  Ordering: %s\n', params.coeffOrdering);
    fprintf('  Freq Threshold: %d\n', params.freqThreshold);

    fprintf('\nUEP Modulation:\n');
    fprintf('  Critical: %d-QAM (%.0f%% of data SCs)\n', params.criticalMod, params.criticalRatio*100);
    fprintf('  Normal: %d-QAM (%.0f%% of data SCs)\n', params.normalMod, params.normalRatio*100);
    fprintf('  Non-critical: %d-QAM (%.0f%% of data SCs)\n', params.noncriticalMod, (1-params.criticalRatio-params.normalRatio)*100);

    fprintf('\nQuality Targets:\n');
    fprintf('  SSIM: %.2f\n', params.targetSSIM);
    fprintf('  PSNR: %.0f dB\n', params.targetPSNR);

    fprintf('\n=============================================\n');
end
