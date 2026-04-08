function [baseBits, enhancementBits, metrics] = two_layer_demapper(rxSymbols, layerMap, params)
% TWO_LAYER_DEMAPPER - Demapper for two-layer residual coding system
%
% Separates and demaps the two layers:
%   Layer 1 (Base): Wavelet LL coefficients via QPSK + strong FEC
%   Layer 2 (Enhancement): DCT residual via 64-QAM + weak FEC
%
% The layers use different subcarriers with different protection levels.
%
% =========================================================================
% TODO: Confirm with TX team:
%   - Exact subcarrier allocation for each layer
%   - FEC schemes used (convolutional? LDPC? Turbo?)
%   - Interleaving between layers
%   - Pilot structure
% =========================================================================
%
% Inputs:
%   rxSymbols - Received frequency-domain OFDM symbols (nFFT x nSymbols)
%   layerMap  - Structure defining layer allocation:
%       .baseIndices       - Subcarrier indices for base layer
%       .baseMod           - Modulation order for base (default: 4 = QPSK)
%       .enhancementIndices- Subcarrier indices for enhancement
%       .enhancementMod    - Modulation for enhancement (default: 64)
%       .pilotIndices      - Pilot subcarrier indices
%   params    - Optional parameters:
%       .softOutput        - Generate soft bits (default: false)
%       .noiseVar          - Noise variance for LLR calculation
%
% Outputs:
%   baseBits        - Demapped bits for base layer (high priority)
%   enhancementBits - Demapped bits for enhancement layer
%   metrics         - Structure with per-layer quality metrics

    %% Default parameters
    if nargin < 3
        params = struct();
    end
    if ~isfield(params, 'softOutput')
        params.softOutput = false;
    end
    if ~isfield(params, 'noiseVar')
        params.noiseVar = 0.1;
    end

    [nFFT, nSymbols] = size(rxSymbols);

    %% Validate and default layer map
    layerMap = validate_layer_map(layerMap, nFFT);

    %% Initialize metrics
    metrics = struct();
    metrics.nFFT = nFFT;
    metrics.nSymbols = nSymbols;

    %% Demap Base Layer (High Priority - QPSK)
    baseSymbols = rxSymbols(layerMap.baseIndices, :);
    baseSymbols = baseSymbols(:);

    if isfield(params, 'useGAForBase') && params.useGAForBase && layerMap.baseMod == 4
    % Use GA-optimized demapper for QPSK base layer
    
        % Check if pilot data is available for GA training
        if isfield(params, 'pilotBaseSymbols') && isfield(params, 'pilotBaseBits')
            % Setup GA parameters
            gaParams = struct();
            gaParams.popSize = 30;
            gaParams.generations = 15;
            gaParams.mutationRate = 0.15;
            gaParams.verbose = isfield(params, 'verbose') && params.verbose;
            
            % Run GA demapper
            [baseBits, bestThresh, gaInfo] = ga_qpsk_demapper( ...
                baseSymbols, params.pilotBaseSymbols, params.pilotBaseBits, gaParams);
            
            % Still calculate standard metrics for comparison
            [~, baseMetrics] = demap_layer(baseSymbols, layerMap.baseMod, params);
            
            % Add GA-specific metrics
            baseMetrics.gaEnabled = true;
            baseMetrics.gaThresholds = bestThresh;
            baseMetrics.gaPilotBER = gaInfo.bestPilotBER;
            baseMetrics.gaFitnessHistory = gaInfo.fitnessHistory;
            
            if gaParams.verbose
                fprintf('   GA Base Demapper: Optimal thresholds = [%.3f, %.3f], Pilot BER = %.3e\n', ...
                    bestThresh(1), bestThresh(2), gaInfo.bestPilotBER);
            end
        else
            % GA enabled but no pilot data - fall back to standard
            warning('two_layer_demapper:noPilots', ...
                'GA base demapper enabled but pilotBaseSymbols/pilotBaseBits not provided. Using standard demapper.');
            [baseBits, baseMetrics] = demap_layer(baseSymbols, layerMap.baseMod, params);
            baseMetrics.gaEnabled = false;
        end
    else
        % Standard demapper (original code path)
        [baseBits, baseMetrics] = demap_layer(baseSymbols, layerMap.baseMod, params);
        baseMetrics.gaEnabled = false;
    end
    
    metrics.base = baseMetrics;
    metrics.base.nSubcarriers = length(layerMap.baseIndices);
    metrics.base.modOrder = layerMap.baseMod;

    %% Demap Enhancement Layer (Lower Priority - 64-QAM)
    enhSymbols = rxSymbols(layerMap.enhancementIndices, :);
    enhSymbols = enhSymbols(:);

    [enhancementBits, enhMetrics] = demap_layer(enhSymbols, layerMap.enhancementMod, params);
    metrics.enhancement = enhMetrics;
    metrics.enhancement.nSubcarriers = length(layerMap.enhancementIndices);
    metrics.enhancement.modOrder = layerMap.enhancementMod;

    %% Calculate layer health and warnings
    metrics = assess_layer_health(metrics, params);

    %% Summary
    metrics.totalBaseBits = length(baseBits);
    metrics.totalEnhancementBits = length(enhancementBits);
    metrics.totalBits = metrics.totalBaseBits + metrics.totalEnhancementBits;

    % Bits per OFDM symbol
    metrics.baseBitsPerSymbol = metrics.totalBaseBits / nSymbols;
    metrics.enhBitsPerSymbol = metrics.totalEnhancementBits / nSymbols;
end

%% Layer Map Validation

function layerMap = validate_layer_map(layerMap, nFFT)
% VALIDATE_LAYER_MAP - Ensure layer map is properly configured
%
% TODO: Update defaults based on TX implementation

    % Default modulation orders
    if ~isfield(layerMap, 'baseMod')
        layerMap.baseMod = 4;      % QPSK for base
    end
    if ~isfield(layerMap, 'enhancementMod')
        layerMap.enhancementMod = 64;  % 64-QAM for enhancement
    end

    % Default subcarrier allocation if not provided
    if ~isfield(layerMap, 'baseIndices') || isempty(layerMap.baseIndices)
        % Default: First 1/4 of data subcarriers for base
        % TODO: Get actual allocation from TX team
        dataStart = 5;  % Skip guard band
        dataEnd = nFFT - 4;
        nData = dataEnd - dataStart + 1;
        nBase = floor(nData * 0.25);

        layerMap.baseIndices = dataStart:(dataStart + nBase - 1);
    end

    if ~isfield(layerMap, 'enhancementIndices') || isempty(layerMap.enhancementIndices)
        % Default: Remaining data subcarriers for enhancement
        dataStart = 5;
        dataEnd = nFFT - 4;
        allData = dataStart:dataEnd;

        % Remove base indices and pilots
        layerMap.enhancementIndices = setdiff(allData, layerMap.baseIndices);
        if isfield(layerMap, 'pilotIndices')
            layerMap.enhancementIndices = setdiff(layerMap.enhancementIndices, layerMap.pilotIndices);
        end
    end

    % Default pilot indices
    if ~isfield(layerMap, 'pilotIndices')
        layerMap.pilotIndices = [];
    end
end

%% Demapping Functions

function [bits, metrics] = demap_layer(symbols, modOrder, params)
% DEMAP_LAYER - Demap symbols for a single layer

    nSymbols = length(symbols);
    bitsPerSymbol = log2(modOrder);

    % QAM demodulation
    bits = qamdemod(symbols, modOrder, 'OutputType', 'bit', 'UnitAveragePower', true);

    % Calculate metrics
    metrics = struct();

    % Hard decision reference for EVM
    hardDecision = qamdemod(symbols, modOrder, 'UnitAveragePower', true);
    idealSymbols = qammod(hardDecision, modOrder, 'UnitAveragePower', true);

    % Error Vector Magnitude
    errorVector = symbols - idealSymbols;
    metrics.EVM_rms = sqrt(mean(abs(errorVector).^2)) / sqrt(mean(abs(idealSymbols).^2)) * 100;
    metrics.EVM_peak = max(abs(errorVector)) / sqrt(mean(abs(idealSymbols).^2)) * 100;

    % Modulation Error Ratio
    signalPower = mean(abs(idealSymbols).^2);
    errorPower = mean(abs(errorVector).^2);
    metrics.MER_dB = 10 * log10(signalPower / (errorPower + eps));

    % Symbol count
    metrics.nSymbols = nSymbols;
    metrics.nBits = length(bits);

    % Constellation spread (clustering quality)
    metrics.avgMinDist = mean(abs(errorVector));
end

%% Health Assessment

function metrics = assess_layer_health(metrics, params)
% ASSESS_LAYER_HEALTH - Determine if each layer is usable

    % Base layer health (critical)
    baseEVM = metrics.base.EVM_rms;
    if baseEVM < 15
        metrics.base.health = 'good';
        metrics.base.usable = true;
    elseif baseEVM < 25
        metrics.base.health = 'marginal';
        metrics.base.usable = true;
        metrics.base.warning = 'Base layer quality marginal';
    else
        metrics.base.health = 'poor';
        metrics.base.usable = false;
        metrics.base.error = 'Base layer quality critical - image may be unusable';
    end

    % Enhancement layer health (less critical)
    enhEVM = metrics.enhancement.EVM_rms;
    if enhEVM < 20
        metrics.enhancement.health = 'good';
        metrics.enhancement.usable = true;
    elseif enhEVM < 35
        metrics.enhancement.health = 'marginal';
        metrics.enhancement.usable = true;
        metrics.enhancement.warning = 'Enhancement quality reduced';
    else
        metrics.enhancement.health = 'poor';
        metrics.enhancement.usable = false;
        metrics.enhancement.warning = 'Enhancement layer degraded - base-only mode';
    end

    % Overall assessment
    if metrics.base.usable && metrics.enhancement.usable
        metrics.overallHealth = 'full_quality';
    elseif metrics.base.usable
        metrics.overallHealth = 'graceful_degradation';
    else
        metrics.overallHealth = 'critical_failure';
    end
end

%% Utility Functions

function map = create_default_layer_map(nFFT, baseRatio)
% CREATE_DEFAULT_LAYER_MAP - Create a default two-layer allocation
%
% Usage: map = create_default_layer_map(64, 0.25)

    if nargin < 2
        baseRatio = 0.25;  % 25% for base layer
    end

    map = struct();
    map.baseMod = 4;           % QPSK
    map.enhancementMod = 64;   % 64-QAM

    % Guard bands
    guardLeft = 4;
    guardRight = 4;
    dcNull = nFFT/2 + 1;

    % Pilot spacing
    pilotSpacing = 8;

    % Calculate subcarrier allocation
    allIndices = (guardLeft + 1):(nFFT - guardRight);
    allIndices = setdiff(allIndices, dcNull);

    % Pilots
    map.pilotIndices = allIndices(1:pilotSpacing:end);
    dataIndices = setdiff(allIndices, map.pilotIndices);

    % Split between base and enhancement
    nData = length(dataIndices);
    nBase = round(nData * baseRatio);

    map.baseIndices = dataIndices(1:nBase);
    map.enhancementIndices = dataIndices(nBase+1:end);
end

function visualize_layer_map(layerMap, nFFT)
% VISUALIZE_LAYER_MAP - Display subcarrier allocation
%
% Usage: visualize_layer_map(layerMap, 64)

    figure('Position', [100, 100, 800, 200]);

    colors = zeros(nFFT, 3) + 0.8;  % Gray for unused

    % Base layer (green)
    for idx = layerMap.baseIndices
        colors(idx, :) = [0.2, 0.7, 0.2];
    end

    % Enhancement layer (orange)
    for idx = layerMap.enhancementIndices
        colors(idx, :) = [0.9, 0.5, 0.1];
    end

    % Pilots (blue)
    if isfield(layerMap, 'pilotIndices')
        for idx = layerMap.pilotIndices
            colors(idx, :) = [0.2, 0.2, 0.9];
        end
    end

    bar(1:nFFT, ones(nFFT,1), 'FaceColor', 'flat', 'CData', colors);
    xlabel('Subcarrier Index');
    title(sprintf('Two-Layer Allocation: Green=Base(%d-QAM), Orange=Enhancement(%d-QAM), Blue=Pilot', ...
        layerMap.baseMod, layerMap.enhancementMod));
    xlim([0, nFFT+1]);
end
