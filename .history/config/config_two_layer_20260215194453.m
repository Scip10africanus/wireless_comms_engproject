%% Configuration Template for Two-Layer Residual Coding RX System
%
% System Architecture:
%   Layer 1 (BASE): DWT LL subband → QPSK → Strong FEC → Priority subcarriers
%   Layer 2 (ENHANCEMENT): DCT residual → 64-QAM → Weak FEC → Performance subcarriers
%
% Key Property: If enhancement is lost, base still provides usable (blurry) image
%
% =========================================================================
%                    *** MEETING CHECKLIST ***
%
% Questions for TX team:
%
% WAVELET BASE LAYER:
%   [ ] Wavelet type? (Haar, db2, db4, bior, etc.)
%   [ ] Number of decomposition levels? (1, 2, 3?)
%   [ ] LL coefficient bit depth?
%   [ ] Any quantization on LL coefficients?
%   [ ] FEC code for base layer? (Conv rate? LDPC?)
%
% DCT ENHANCEMENT LAYER:
%   [ ] DCT block size? (8x8 standard?)
%   [ ] Coefficient ordering? (Zigzag? Raster?)
%   [ ] How many DCT coefficients per block? (All 64? Truncated?)
%   [ ] Quantization table for residual?
%   [ ] FEC code for enhancement? (Weaker than base?)
%
% SUBCARRIER ALLOCATION:
%   [ ] What % of subcarriers for base layer?
%   [ ] Exact subcarrier indices for each layer?
%   [ ] Is allocation static or adaptive?
%   [ ] Pilot positions?
%
% RESIDUAL CALCULATION (at TX):
%   [ ] R = Original - Base_reconstructed (confirm)
%   [ ] Is residual scaled before DCT?
%   [ ] Coefficient clipping/range?
%
% =========================================================================

function params = config_two_layer()
% CONFIG_TWO_LAYER - Returns configuration for two-layer system
%
% Usage:
%   params = config_two_layer();
%   [finalImage, feedback, results] = two_layer_rx_pipeline(rxSignal, params);

    params = struct();

    %% ==================== OFDM PARAMETERS ====================

    params.nFFT = 64;               % FFT size
    params.nCP = 16;                % Cyclic prefix length
    params.equalizer = 'ZF';        % 'ZF' or 'MMSE'
    params.noiseVar = 0.01;         % Noise variance estimate

    % Pilot configuration
    params.pilotIndices = [5, 13, 21, 29, 37, 45, 53, 61];  % Example
    params.pilotValues = ones(8, 1);

    %% ==================== IMAGE PARAMETERS ====================

    params.imageHeight = 64;        % Original image height
    params.imageWidth = 64;         % Original image width
    params.nChannels = 1;           % 1=grayscale, 3=RGB

    %% ==================== BASE LAYER (WAVELET) ====================
    % High Priority: Structure/recognizability
    % Protected with: QPSK + Strong FEC + Best subcarriers

    params.baseMod = 4;             % QPSK (most robust)

    % Wavelet parameters
    params.waveletType = 'haar';    % Options: 'haar', 'db2', 'db4'
    params.waveletLevels = 2;       % Decomposition levels
                                    % After 2 levels: LL is 1/4 size
                                    % After 3 levels: LL is 1/8 size

    params.baseBitsPerCoeff = 8;    % Bits per LL coefficient

    % FEC for base layer
    params.baseFEC = 'none';        % TODO: 'convolutional', 'ldpc'
    params.baseFECRate = 1/2;       % Strong protection

    %% ==================== ENHANCEMENT LAYER (DCT RESIDUAL) ====================
    % Lower Priority: Sharpness/detail (R = I - I_base)
    % Protected with: 64-QAM + Weak FEC + Remaining subcarriers

    params.enhancementMod = 64;     % 64-QAM (high throughput)

    % DCT parameters
    params.dctBlockSize = 8;        % 8x8 DCT blocks
    params.coeffOrdering = 'zigzag';% Coefficient scan order
    params.maxDctCoeffs = 64;       % Coefficients per block (64 = all)
                                    % Can truncate to e.g., 32 for compression

    params.enhancementBitsPerCoeff = 8;  % Bits per DCT coefficient

    % FEC for enhancement layer
    params.enhancementFEC = 'none'; % TODO: 'convolutional', 'ldpc'
    params.enhancementFECRate = 3/4;% Weaker protection (higher rate)

    %% ==================== SUBCARRIER ALLOCATION ====================
    % Defines which subcarriers carry which layer

    % Guard bands and DC
    guardLeft = 4;
    guardRight = 4;
    dcNull = params.nFFT/2 + 1;

    % Available data subcarriers
    allData = setdiff((guardLeft+1):(params.nFFT-guardRight), dcNull);
    allData = setdiff(allData, params.pilotIndices);

    % Allocation ratio
    baseRatio = 0.25;               % 25% for base (TODO: confirm with TX)
    nBase = round(length(allData) * baseRatio);

    % Assign subcarriers
    % Option 1: First N subcarriers for base
    params.baseSubcarriers = allData(1:nBase);
    params.enhancementSubcarriers = allData(nBase+1:end);

    % Option 2: Interleaved (uncomment to use)
    % params.baseSubcarriers = allData(1:4:end);
    % params.enhancementSubcarriers = setdiff(allData, params.baseSubcarriers);

    %% ==================== FUSION PARAMETERS ====================

    params.residualScale = 1.0;     % Scale factor for residual
                                    % TODO: Confirm with TX if residual is scaled

    %% ==================== QUALITY TARGETS ====================

    params.targetSSIM = 0.90;       % Target SSIM quality
    params.targetPSNR = 30;         % Target PSNR (dB)


    %% ==================== GA DEMAPPER (OPTIONAL) ====================
    % Genetic Algorithm-enhanced adaptive demapping
    % Set to true to enable GA optimization of decision boundaries

    params.useGAForBase = false;         % Enable GA for base layer (QPSK)
    params.useGAForEnhancement = false;  % Enable GA for enhancement (not implemented yet)

    % GA Parameters (used if GA is enabled)
    params.gaPopSize = 30;               % Population size
    params.gaGenerations = 15;           % Number of generations
    params.gaMutationRate = 0.15;        % Mutation probability

    % Pilot data for GA training (must be set by user if GA enabled)
    % params.pilotBaseSymbols = [];      % Known pilot symbols for base layer
    % params.pilotBaseBits = [];         % Known pilot bits

    %% ==================== DISPLAY ====================

    params.verbose = true;          % Print progress

end

%% ==================== PRESETS ====================

function params = config_two_layer_robust()
% CONFIG_TWO_LAYER_ROBUST - Maximum protection for poor channels

    params = config_two_layer();

    % More robust base
    params.baseMod = 2;             % BPSK
    params.waveletLevels = 3;       % Smaller LL (more compression)

    % More robust enhancement
    params.enhancementMod = 16;     % 16-QAM instead of 64-QAM
    params.maxDctCoeffs = 32;       % Truncate high-freq DCT

    % Larger base allocation
    allData = setdiff(5:60, [33, params.pilotIndices]);
    params.baseSubcarriers = allData(1:round(length(allData)*0.4));
    params.enhancementSubcarriers = allData(round(length(allData)*0.4)+1:end);

    % Lower targets
    params.targetSSIM = 0.80;
    params.targetPSNR = 25;
end

function params = config_two_layer_highrate()
% CONFIG_TWO_LAYER_HIGHRATE - Maximum throughput for good channels

    params = config_two_layer();

    % Higher order modulations
    params.baseMod = 16;            % 16-QAM for base
    params.enhancementMod = 256;    % 256-QAM for enhancement

    % Full DCT coefficients
    params.maxDctCoeffs = 64;

    % Smaller base allocation
    allData = setdiff(5:60, [33, params.pilotIndices]);
    params.baseSubcarriers = allData(1:round(length(allData)*0.15));
    params.enhancementSubcarriers = allData(round(length(allData)*0.15)+1:end);

    % Higher targets
    params.targetSSIM = 0.95;
    params.targetPSNR = 40;
end

%% ==================== HELPER FUNCTIONS ====================

function print_two_layer_config(params)
% PRINT_TWO_LAYER_CONFIG - Display configuration summary

    fprintf('\n');
    fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║           Two-Layer System Configuration                      ║\n');
    fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf('║ OFDM: FFT=%d, CP=%d                                          ║\n', params.nFFT, params.nCP);
    fprintf('║ Image: %dx%dx%d                                              ║\n', params.imageHeight, params.imageWidth, params.nChannels);
    fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf('║ BASE LAYER (Priority Path)                                    ║\n');
    fprintf('║   Wavelet: %s, %d levels                                      ║\n', params.waveletType, params.waveletLevels);
    fprintf('║   Modulation: %d-QAM (robust)                                 ║\n', params.baseMod);
    fprintf('║   Subcarriers: %d                                             ║\n', length(params.baseSubcarriers));
    fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf('║ ENHANCEMENT LAYER (Performance Path)                          ║\n');
    fprintf('║   DCT: %dx%d blocks, %d coeffs                                ║\n', params.dctBlockSize, params.dctBlockSize, params.maxDctCoeffs);
    fprintf('║   Modulation: %d-QAM (high throughput)                        ║\n', params.enhancementMod);
    fprintf('║   Subcarriers: %d                                             ║\n', length(params.enhancementSubcarriers));
    fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
    fprintf('║ TARGETS: SSIM >= %.2f, PSNR >= %.0f dB                         ║\n', params.targetSSIM, params.targetPSNR);
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n');
end

function visualize_allocation(params)
% VISUALIZE_ALLOCATION - Show subcarrier allocation graphically

    figure('Position', [100, 100, 900, 250]);

    nFFT = params.nFFT;
    colors = ones(nFFT, 3) * 0.7;  % Gray = unused

    % Base (green)
    for sc = params.baseSubcarriers
        colors(sc, :) = [0.2, 0.8, 0.3];
    end

    % Enhancement (orange)
    for sc = params.enhancementSubcarriers
        colors(sc, :) = [1.0, 0.6, 0.2];
    end

    % Pilots (blue)
    for sc = params.pilotIndices
        colors(sc, :) = [0.3, 0.5, 0.9];
    end

    bar(1:nFFT, ones(nFFT,1), 'FaceColor', 'flat', 'CData', colors, 'EdgeColor', 'none');

    xlabel('Subcarrier Index');
    title(sprintf('Two-Layer Allocation: Green=Base(%d-QAM), Orange=Enhancement(%d-QAM), Blue=Pilot', ...
        params.baseMod, params.enhancementMod));
    xlim([0, nFFT+1]);

    % Add legend
    hold on;
    h1 = bar(NaN, NaN, 'FaceColor', [0.2, 0.8, 0.3]);
    h2 = bar(NaN, NaN, 'FaceColor', [1.0, 0.6, 0.2]);
    h3 = bar(NaN, NaN, 'FaceColor', [0.3, 0.5, 0.9]);
    legend([h1, h2, h3], {'Base (QPSK)', 'Enhancement (64-QAM)', 'Pilot'}, 'Location', 'best');
end

%% ==================== QUICK REFERENCE ====================
%
% TWO-LAYER RESIDUAL CODING SYSTEM
% ================================
%
% TRANSMITTER:
%   1. Apply DWT to image → Extract LL subband (base)
%   2. Reconstruct base at TX: I_base = IDWT(LL, zeros, zeros, zeros)
%   3. Calculate residual: R = I_original - I_base
%   4. Apply DCT to residual
%   5. Transmit:
%      - Base (LL coeffs) → QPSK → Strong FEC → Priority subcarriers
%      - Enhancement (DCT coeffs) → 64-QAM → Weak FEC → Other subcarriers
%
% RECEIVER (this code):
%   1. OFDM demodulation
%   2. Separate layers by subcarrier
%   3. Demap base (QPSK) and enhancement (64-QAM)
%   4. FEC decode each layer
%   5. Wavelet decode base → I_base
%   6. DCT decode enhancement → R
%   7. Fuse: I_final = I_base + R
%   8. Generate feedback for TX adaptation
%
% GRACEFUL DEGRADATION:
%   - Both OK: Full quality image
%   - Enhancement lost: Blurry but recognizable image (base only)
%   - Base lost: CRITICAL FAILURE (enhancement alone is meaningless)
%
