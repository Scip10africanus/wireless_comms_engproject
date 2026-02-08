function [bits, metrics] = uep_demapper(rxSymbols, subcarrierMap, params)
% UEP_DEMAPPER - Unequal Error Protection aware symbol demapper
%
% Handles OFDM symbols where different subcarriers use different
% modulation orders (e.g., QPSK for low-freq DCT, 64-QAM for high-freq).
%
% =========================================================================
% TODO: Adapt after meeting with TX team:
%   - Confirm subcarrier-to-modulation mapping strategy
%   - Confirm if mapping is static or varies per frame
%   - Confirm bit interleaving scheme (if any)
%   - Confirm pilot subcarrier positions
% =========================================================================
%
% Inputs:
%   rxSymbols    - Received frequency-domain symbols (nFFT x nSymbols)
%   subcarrierMap - Structure defining subcarrier allocation:
%       .modOrders    - Modulation order per subcarrier (nFFT x 1)
%                       e.g., [4 4 4 16 16 64 64 64...] for mixed
%       .dataIndices  - Indices of data subcarriers
%       .pilotIndices - Indices of pilot subcarriers
%       .nullIndices  - Indices of null/guard subcarriers
%       .importance   - 'critical', 'normal', 'noncritical' per subcarrier
%   params       - Optional parameters:
%       .softOutput  - Generate soft bits/LLRs (default: false)
%       .noiseVar    - Noise variance for LLR calculation
%
% Outputs:
%   bits    - Demapped bits, organized by importance level:
%       .critical    - Bits from low-freq/QPSK subcarriers
%       .normal      - Bits from mid-freq subcarriers (if used)
%       .noncritical - Bits from high-freq/64-QAM subcarriers
%       .all         - All bits in transmission order
%   metrics - Demapping quality metrics per importance level

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

    %% Validate subcarrier map
    subcarrierMap = validate_subcarrier_map(subcarrierMap, nFFT);

    %% Initialize outputs
    bits = struct();
    bits.critical = [];
    bits.normal = [];
    bits.noncritical = [];
    bits.all = [];

    metrics = struct();
    metrics.critical = struct('EVM', [], 'count', 0);
    metrics.normal = struct('EVM', [], 'count', 0);
    metrics.noncritical = struct('EVM', [], 'count', 0);

    %% Process each subcarrier group by modulation order
    uniqueMods = unique(subcarrierMap.modOrders(subcarrierMap.dataIndices));

    for modOrder = uniqueMods'
        if modOrder == 0
            continue;  % Skip null subcarriers
        end

        % Find subcarriers using this modulation
        scIndices = find(subcarrierMap.modOrders == modOrder);
        scIndices = intersect(scIndices, subcarrierMap.dataIndices);

        if isempty(scIndices)
            continue;
        end

        % Extract symbols for these subcarriers
        scSymbols = rxSymbols(scIndices, :);
        scSymbols = scSymbols(:);  % Flatten

        % Demap using standard QAM demodulation
        scBits = qamdemod(scSymbols, modOrder, ...
            'OutputType', 'bit', 'UnitAveragePower', true);

        % Calculate EVM for this group
        idealSymbols = qammod(qamdemod(scSymbols, modOrder, 'UnitAveragePower', true), ...
            modOrder, 'UnitAveragePower', true);
        evm = sqrt(mean(abs(scSymbols - idealSymbols).^2)) / ...
              sqrt(mean(abs(idealSymbols).^2)) * 100;

        % Categorize by importance level
        % TODO: Update categorization based on TX team's mapping
        importance = categorize_modulation(modOrder, subcarrierMap, scIndices);

        switch importance
            case 'critical'
                bits.critical = [bits.critical; scBits];
                metrics.critical.EVM = [metrics.critical.EVM; evm];
                metrics.critical.count = metrics.critical.count + length(scBits);
            case 'normal'
                bits.normal = [bits.normal; scBits];
                metrics.normal.EVM = [metrics.normal.EVM; evm];
                metrics.normal.count = metrics.normal.count + length(scBits);
            case 'noncritical'
                bits.noncritical = [bits.noncritical; scBits];
                metrics.noncritical.EVM = [metrics.noncritical.EVM; evm];
                metrics.noncritical.count = metrics.noncritical.count + length(scBits);
        end
    end

    %% Reconstruct full bitstream in transmission order
    % TODO: This ordering depends on how TX interleaves the data
    % Option 1: Critical first, then normal, then non-critical
    % Option 2: Interleaved by subcarrier
    % Option 3: Interleaved by OFDM symbol

    % Default: Sequential by importance (modify based on TX)
    bits.all = [bits.critical; bits.normal; bits.noncritical];

    % Alternative: Reconstruct by subcarrier order
    bits.bySubcarrier = reconstruct_by_subcarrier(rxSymbols, subcarrierMap);

    %% Aggregate metrics
    metrics.critical.avgEVM = mean(metrics.critical.EVM);
    metrics.normal.avgEVM = mean(metrics.normal.EVM);
    metrics.noncritical.avgEVM = mean(metrics.noncritical.EVM);

    metrics.totalBits = length(bits.all);
    metrics.criticalRatio = metrics.critical.count / max(metrics.totalBits, 1);
end

%% Helper Functions

function scMap = validate_subcarrier_map(scMap, nFFT)
% VALIDATE_SUBCARRIER_MAP - Ensure subcarrier map is properly configured
%
% TODO: Update defaults based on TX implementation

    % Default modulation orders if not provided
    if ~isfield(scMap, 'modOrders') || isempty(scMap.modOrders)
        % Default: QPSK on first half, 64-QAM on second half
        % TODO: Update based on actual TX mapping
        scMap.modOrders = zeros(nFFT, 1);
        scMap.modOrders(1:floor(nFFT/2)) = 4;      % QPSK for low-freq
        scMap.modOrders(floor(nFFT/2)+1:end) = 64; % 64-QAM for high-freq
    end

    % Default data indices
    if ~isfield(scMap, 'dataIndices') || isempty(scMap.dataIndices)
        % Assume all non-DC, non-edge subcarriers are data
        scMap.dataIndices = 2:(nFFT-1);
    end

    % Default pilot indices
    if ~isfield(scMap, 'pilotIndices')
        scMap.pilotIndices = [];
    end

    % Default null indices
    if ~isfield(scMap, 'nullIndices')
        scMap.nullIndices = [];
    end

    % Default importance mapping
    if ~isfield(scMap, 'importance')
        scMap.importance = cell(nFFT, 1);
        for i = 1:nFFT
            if scMap.modOrders(i) <= 4
                scMap.importance{i} = 'critical';
            elseif scMap.modOrders(i) <= 16
                scMap.importance{i} = 'normal';
            else
                scMap.importance{i} = 'noncritical';
            end
        end
    end
end

function importance = categorize_modulation(modOrder, scMap, scIndices)
% CATEGORIZE_MODULATION - Determine importance level from modulation
%
% TODO: This mapping should match TX team's UEP strategy
%
% Default assumption:
%   QPSK (4)     -> critical (low-freq DCT, needs protection)
%   16-QAM (16)  -> normal (mid-freq, moderate protection)
%   64-QAM (64)  -> noncritical (high-freq, can tolerate errors)

    % Check if explicit importance is provided
    if isfield(scMap, 'importance') && ~isempty(scMap.importance)
        % Use first subcarrier's importance as representative
        importance = scMap.importance{scIndices(1)};
        return;
    end

    % Default mapping based on modulation order
    switch modOrder
        case {2, 4}  % BPSK, QPSK
            importance = 'critical';
        case {8, 16}  % 8-PSK, 16-QAM
            importance = 'normal';
        otherwise  % 64-QAM, 256-QAM, etc.
            importance = 'noncritical';
    end
end

function bits = reconstruct_by_subcarrier(rxSymbols, scMap)
% RECONSTRUCT_BY_SUBCARRIER - Demap in subcarrier order
%
% This preserves the original transmission order if data is sent
% subcarrier-by-subcarrier rather than grouped by modulation.

    [nFFT, nSymbols] = size(rxSymbols);
    bits = [];

    for sym = 1:nSymbols
        for sc = scMap.dataIndices
            modOrder = scMap.modOrders(sc);
            if modOrder == 0
                continue;
            end

            symbol = rxSymbols(sc, sym);
            scBits = qamdemod(symbol, modOrder, ...
                'OutputType', 'bit', 'UnitAveragePower', true);
            bits = [bits; scBits];
        end
    end
end
