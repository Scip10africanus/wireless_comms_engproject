function scMap = subcarrier_mapper(nFFT, config)
% SUBCARRIER_MAPPER - Generate subcarrier allocation map for UEP
%
% Creates a mapping structure that defines:
%   - Which subcarriers carry data, pilots, or nulls
%   - Modulation order per data subcarrier
%   - Importance level (critical/normal/noncritical)
%   - Mapping to DCT frequency bands
%
% =========================================================================
% TODO: Adapt after meeting with TX team:
%   - Confirm subcarrier allocation strategy
%   - Confirm pilot placement pattern
%   - Confirm guard band sizes
%   - Confirm DCT-to-subcarrier mapping logic
% =========================================================================
%
% Inputs:
%   nFFT   - Number of FFT points (subcarriers)
%   config - Configuration structure:
%       .mappingType     - 'fixed', 'adaptive', or 'custom' (default: 'fixed')
%       .criticalRatio   - Fraction of subcarriers for critical data (default: 0.3)
%       .normalRatio     - Fraction for normal data (default: 0.2)
%       .criticalMod     - Modulation for critical (default: 4 = QPSK)
%       .normalMod       - Modulation for normal (default: 16 = 16-QAM)
%       .noncriticalMod  - Modulation for non-critical (default: 64 = 64-QAM)
%       .pilotSpacing    - Pilot subcarrier spacing (default: 8)
%       .guardBands      - [left, right] guard subcarriers (default: [4, 4])
%       .dcNull          - Null DC subcarrier (default: true)
%
% Outputs:
%   scMap - Subcarrier map structure for use with uep_demapper

    %% Default configuration
    if nargin < 2
        config = struct();
    end

    config = set_defaults(config);

    %% Initialize map structure
    scMap = struct();
    scMap.nFFT = nFFT;
    scMap.modOrders = zeros(nFFT, 1);
    scMap.importance = cell(nFFT, 1);
    scMap.dctBand = zeros(nFFT, 1);  % Which DCT band this SC maps to

    %% Determine subcarrier types
    % Guard bands (null)
    leftGuard = 1:config.guardBands(1);
    rightGuard = (nFFT - config.guardBands(2) + 1):nFFT;

    % DC subcarrier (null)
    if config.dcNull
        dcIdx = nFFT/2 + 1;  % For centered FFT
    else
        dcIdx = [];
    end

    % Pilot subcarriers
    pilotCandidates = (config.guardBands(1) + 1):(nFFT - config.guardBands(2));
    pilotCandidates = setdiff(pilotCandidates, dcIdx);
    scMap.pilotIndices = pilotCandidates(1:config.pilotSpacing:end);

    % Null subcarriers
    scMap.nullIndices = unique([leftGuard, rightGuard, dcIdx]);

    % Data subcarriers (everything else)
    allIndices = 1:nFFT;
    scMap.dataIndices = setdiff(allIndices, [scMap.nullIndices, scMap.pilotIndices]);

    %% Assign modulation and importance based on mapping type
    switch config.mappingType
        case 'fixed'
            scMap = assign_fixed_mapping(scMap, config);

        case 'adaptive'
            % Placeholder for dynamic mapping based on channel state
            % TODO: Implement adaptive logic after understanding feedback mechanism
            scMap = assign_fixed_mapping(scMap, config);

        case 'snr_ordered'
            % Map critical data to subcarriers with best SNR
            % TODO: Requires channel state information
            scMap = assign_fixed_mapping(scMap, config);

        case 'frequency_ordered'
            % Lower subcarriers = lower DCT freq = more critical
            scMap = assign_frequency_ordered(scMap, config);

        case 'custom'
            % TODO: Implement custom mapping from TX team
            scMap = assign_fixed_mapping(scMap, config);

        otherwise
            scMap = assign_fixed_mapping(scMap, config);
    end

    %% Generate summary
    scMap.summary = generate_summary(scMap, config);
end

%% Configuration Functions

function config = set_defaults(config)
% SET_DEFAULTS - Apply default configuration values

    if ~isfield(config, 'mappingType')
        config.mappingType = 'fixed';
    end
    if ~isfield(config, 'criticalRatio')
        config.criticalRatio = 0.3;  % 30% for critical (low-freq DCT)
    end
    if ~isfield(config, 'normalRatio')
        config.normalRatio = 0.2;    % 20% for normal (mid-freq DCT)
    end
    if ~isfield(config, 'criticalMod')
        config.criticalMod = 4;      % QPSK
    end
    if ~isfield(config, 'normalMod')
        config.normalMod = 16;       % 16-QAM
    end
    if ~isfield(config, 'noncriticalMod')
        config.noncriticalMod = 64;  % 64-QAM
    end
    if ~isfield(config, 'pilotSpacing')
        config.pilotSpacing = 8;
    end
    if ~isfield(config, 'guardBands')
        config.guardBands = [4, 4];
    end
    if ~isfield(config, 'dcNull')
        config.dcNull = true;
    end
end

%% Mapping Assignment Functions

function scMap = assign_fixed_mapping(scMap, config)
% ASSIGN_FIXED_MAPPING - Static subcarrier-to-importance mapping
%
% Divides data subcarriers into three groups based on ratios

    nData = length(scMap.dataIndices);
    nCritical = round(nData * config.criticalRatio);
    nNormal = round(nData * config.normalRatio);

    % Sort data indices (lower index = lower frequency assumption)
    sortedData = sort(scMap.dataIndices);

    % Assign critical (first portion)
    criticalSC = sortedData(1:nCritical);
    for sc = criticalSC
        scMap.modOrders(sc) = config.criticalMod;
        scMap.importance{sc} = 'critical';
        scMap.dctBand(sc) = 1;  % Low-freq band
    end

    % Assign normal (middle portion)
    normalSC = sortedData(nCritical+1:nCritical+nNormal);
    for sc = normalSC
        scMap.modOrders(sc) = config.normalMod;
        scMap.importance{sc} = 'normal';
        scMap.dctBand(sc) = 2;  % Mid-freq band
    end

    % Assign non-critical (remaining)
    noncriticalSC = sortedData(nCritical+nNormal+1:end);
    for sc = noncriticalSC
        scMap.modOrders(sc) = config.noncriticalMod;
        scMap.importance{sc} = 'noncritical';
        scMap.dctBand(sc) = 3;  % High-freq band
    end

    % Mark pilots and nulls
    for sc = scMap.pilotIndices
        scMap.importance{sc} = 'pilot';
        scMap.modOrders(sc) = 0;
    end
    for sc = scMap.nullIndices
        scMap.importance{sc} = 'null';
        scMap.modOrders(sc) = 0;
    end
end

function scMap = assign_frequency_ordered(scMap, config)
% ASSIGN_FREQUENCY_ORDERED - Map based on OFDM frequency position
%
% Lower frequency subcarriers get lower modulation (more protection)
% This may align naturally with DCT frequency ordering

    dataIdx = sort(scMap.dataIndices);
    nData = length(dataIdx);

    % Calculate thresholds
    thresh1 = round(nData * config.criticalRatio);
    thresh2 = round(nData * (config.criticalRatio + config.normalRatio));

    for i = 1:nData
        sc = dataIdx(i);
        if i <= thresh1
            scMap.modOrders(sc) = config.criticalMod;
            scMap.importance{sc} = 'critical';
            scMap.dctBand(sc) = 1;
        elseif i <= thresh2
            scMap.modOrders(sc) = config.normalMod;
            scMap.importance{sc} = 'normal';
            scMap.dctBand(sc) = 2;
        else
            scMap.modOrders(sc) = config.noncriticalMod;
            scMap.importance{sc} = 'noncritical';
            scMap.dctBand(sc) = 3;
        end
    end

    % Mark pilots and nulls
    for sc = scMap.pilotIndices
        scMap.importance{sc} = 'pilot';
    end
    for sc = scMap.nullIndices
        scMap.importance{sc} = 'null';
    end
end

%% Utility Functions

function summary = generate_summary(scMap, config)
% GENERATE_SUMMARY - Create summary of subcarrier allocation

    summary = struct();

    % Count by type
    summary.nData = length(scMap.dataIndices);
    summary.nPilots = length(scMap.pilotIndices);
    summary.nNull = length(scMap.nullIndices);

    % Count by importance
    summary.nCritical = sum(strcmp(scMap.importance, 'critical'));
    summary.nNormal = sum(strcmp(scMap.importance, 'normal'));
    summary.nNoncritical = sum(strcmp(scMap.importance, 'noncritical'));

    % Bits per OFDM symbol
    criticalBits = summary.nCritical * log2(config.criticalMod);
    normalBits = summary.nNormal * log2(config.normalMod);
    noncriticalBits = summary.nNoncritical * log2(config.noncriticalMod);
    summary.bitsPerSymbol = criticalBits + normalBits + noncriticalBits;

    % Effective spectral efficiency
    summary.spectralEfficiency = summary.bitsPerSymbol / scMap.nFFT;

    % Protection ratio (lower mod = more protected)
    summary.avgModOrder = (summary.nCritical * config.criticalMod + ...
                           summary.nNormal * config.normalMod + ...
                           summary.nNoncritical * config.noncriticalMod) / ...
                          max(summary.nData, 1);
end

%% Visualization Function

function visualize_map(scMap)
% VISUALIZE_MAP - Plot subcarrier allocation
% Call separately: visualize_map(scMap)

    figure('Position', [100, 100, 800, 300]);

    colors = zeros(scMap.nFFT, 3);
    for i = 1:scMap.nFFT
        switch scMap.importance{i}
            case 'critical'
                colors(i,:) = [0.2, 0.6, 0.2];  % Green
            case 'normal'
                colors(i,:) = [0.9, 0.7, 0.1];  % Yellow
            case 'noncritical'
                colors(i,:) = [0.8, 0.3, 0.3];  % Red
            case 'pilot'
                colors(i,:) = [0.3, 0.3, 0.9];  % Blue
            case 'null'
                colors(i,:) = [0.5, 0.5, 0.5];  % Gray
        end
    end

    bar(1:scMap.nFFT, ones(scMap.nFFT,1), 'FaceColor', 'flat', 'CData', colors);
    xlabel('Subcarrier Index');
    ylabel('Allocation');
    title('Subcarrier Mapping (Green=Critical, Yellow=Normal, Red=NonCritical, Blue=Pilot)');
    xlim([0, scMap.nFFT+1]);
end
