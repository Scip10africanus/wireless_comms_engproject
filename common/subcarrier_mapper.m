function scMap = subcarrier_mapper(nFFT, config)
% subcarrier_mapper - assigns subcarriers to critical/normal/noncritical groups
% critical SCs get lower-order modulation (more protection for base layer)
% noncritical SCs get higher-order mod for the enhancement layer
%
% TODO: revisit pilot spacing and guard band sizes after testing on real channel

    if nargin < 2, config = struct(); end
    config = set_defaults(config);

    scMap = struct();
    scMap.nFFT = nFFT;
    scMap.modOrders = zeros(nFFT, 1);
    scMap.importance = cell(nFFT, 1);
    scMap.dctBand = zeros(nFFT, 1);

    % guard bands at edges, DC null in middle, pilots every pilotSpacing SCs
    leftGuard = 1:config.guardBands(1);
    rightGuard = (nFFT - config.guardBands(2) + 1):nFFT;
    if config.dcNull
        dcIdx = nFFT/2 + 1;
    else
        dcIdx = [];
    end

    pilotCandidates = (config.guardBands(1) + 1):(nFFT - config.guardBands(2));
    pilotCandidates = setdiff(pilotCandidates, dcIdx);
    scMap.pilotIndices = pilotCandidates(1:config.pilotSpacing:end);

    scMap.nullIndices = unique([leftGuard, rightGuard, dcIdx]);
    allIndices = 1:nFFT;
    scMap.dataIndices = setdiff(allIndices, [scMap.nullIndices, scMap.pilotIndices]);

    switch config.mappingType
        case 'fixed'
            scMap = assign_fixed_mapping(scMap, config);

        case 'adaptive'
            % TODO: implement CSI-based adaptive mapping later
            scMap = assign_fixed_mapping(scMap, config);

        case 'snr_ordered'
            % TODO: need channel state info for this, using fixed for now
            scMap = assign_fixed_mapping(scMap, config);

        case 'frequency_ordered'
            scMap = assign_frequency_ordered(scMap, config);

        case 'custom'
            % TODO: custom mapping
            scMap = assign_fixed_mapping(scMap, config);

        otherwise
            scMap = assign_fixed_mapping(scMap, config);
    end

    scMap.summary = generate_summary(scMap, config);
end

function config = set_defaults(config)

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

function scMap = assign_fixed_mapping(scMap, config)
% split data SCs into critical/normal/noncritical by ratio

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
% lower freq SCs -> more protected (lower mod order)

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

function summary = generate_summary(scMap, config)

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

function visualize_map(scMap)
% plots the subcarrier allocation map (call manually if needed)

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
