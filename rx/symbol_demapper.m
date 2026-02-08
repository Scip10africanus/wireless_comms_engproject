function [bits, softBits, metrics] = symbol_demapper(rxSymbols, params)
% SYMBOL_DEMAPPER - Flexible QAM symbol demapping with soft/hard output
%
% Supports BPSK, QPSK, 16-QAM, 64-QAM, and 256-QAM with Gray coding.
% Can output hard bits or soft bits (LLRs) for channel decoding.
%
% Inputs:
%   rxSymbols - Received QAM symbols (column vector or matrix)
%   params    - Structure with fields:
%       .modOrder     - Modulation order (2, 4, 16, 64, 256)
%       .outputType   - 'hard' or 'soft' (default: 'hard')
%       .noiseVar     - Noise variance for LLR calculation (for soft output)
%       .unitPower    - true if unit average power (default: true)
%
% Outputs:
%   bits     - Demodulated hard bits
%   softBits - Soft bits (LLRs) if outputType='soft', otherwise []
%   metrics  - Structure with demapping quality metrics

    %% Input validation and defaults
    if ~isfield(params, 'outputType')
        params.outputType = 'hard';
    end
    if ~isfield(params, 'noiseVar')
        params.noiseVar = 0.1;
    end
    if ~isfield(params, 'unitPower')
        params.unitPower = true;
    end

    % Flatten input
    rxFlat = rxSymbols(:);
    nSymbols = length(rxFlat);
    bitsPerSymbol = log2(params.modOrder);

    %% Generate reference constellation
    [constellation, grayMap] = generate_constellation(params.modOrder, params.unitPower);

    %% Hard demapping
    if strcmp(params.outputType, 'hard')
        bits = hard_demap(rxFlat, constellation, grayMap, bitsPerSymbol);
        softBits = [];
    else
        % Soft demapping (LLR calculation)
        [bits, softBits] = soft_demap(rxFlat, constellation, grayMap, ...
                                       bitsPerSymbol, params.noiseVar);
    end

    %% Calculate metrics
    metrics = calculate_demap_metrics(rxFlat, constellation, grayMap, params);
    metrics.nSymbols = nSymbols;
    metrics.bitsPerSymbol = bitsPerSymbol;
    metrics.totalBits = length(bits);
end

%% Helper Functions

function [constellation, grayMap] = generate_constellation(M, unitPower)
% GENERATE_CONSTELLATION - Generate QAM constellation with Gray mapping

    k = log2(M);

    switch M
        case 2  % BPSK
            constellation = [-1; 1];
            grayMap = [0; 1];

        case 4  % QPSK
            constellation = (1/sqrt(2)) * [(-1-1j); (-1+1j); (1-1j); (1+1j)];
            grayMap = [0; 1; 2; 3];  % Gray: 00, 01, 10, 11

        otherwise  % Square QAM (16, 64, 256)
            sqrtM = sqrt(M);
            [re, im] = meshgrid(-(sqrtM-1):2:(sqrtM-1), -(sqrtM-1):2:(sqrtM-1));
            constellation = re(:) + 1j*im(:);

            % Generate Gray code mapping
            grayMap = zeros(M, 1);
            for i = 0:M-1
                grayMap(i+1) = i;
            end

            % Normalize power if requested
            if unitPower
                avgPower = mean(abs(constellation).^2);
                constellation = constellation / sqrt(avgPower);
            end
    end

    if unitPower && M <= 4
        avgPower = mean(abs(constellation).^2);
        constellation = constellation / sqrt(avgPower);
    end
end

function bits = hard_demap(rxSymbols, constellation, grayMap, bitsPerSymbol)
% HARD_DEMAP - Minimum distance hard demapping

    nSymbols = length(rxSymbols);
    M = length(constellation);

    % Calculate distances to all constellation points
    distances = abs(repmat(rxSymbols, 1, M) - repmat(constellation.', nSymbols, 1));

    % Find minimum distance indices
    [~, minIdx] = min(distances, [], 2);

    % Map indices to Gray-coded bit patterns
    symbolIdx = grayMap(minIdx);

    % Convert to bits
    bits = zeros(nSymbols * bitsPerSymbol, 1);
    for i = 1:nSymbols
        bitPattern = de2bi(symbolIdx(i), bitsPerSymbol, 'left-msb');
        bits((i-1)*bitsPerSymbol + 1 : i*bitsPerSymbol) = bitPattern(:);
    end
end

function [bits, llr] = soft_demap(rxSymbols, constellation, grayMap, bitsPerSymbol, noiseVar)
% SOFT_DEMAP - Soft demapping with LLR calculation

    nSymbols = length(rxSymbols);
    M = length(constellation);

    % Preallocate
    llr = zeros(nSymbols * bitsPerSymbol, 1);

    % Calculate LLRs for each bit position
    for sym = 1:nSymbols
        y = rxSymbols(sym);

        for bit = 1:bitsPerSymbol
            % Find constellation points where this bit = 0 and = 1
            S0 = [];
            S1 = [];

            for m = 1:M
                bitPattern = de2bi(grayMap(m), bitsPerSymbol, 'left-msb');
                if bitPattern(bit) == 0
                    S0 = [S0; constellation(m)];
                else
                    S1 = [S1; constellation(m)];
                end
            end

            % Max-Log-MAP LLR approximation
            dist0 = min(abs(y - S0).^2);
            dist1 = min(abs(y - S1).^2);

            llrIdx = (sym-1)*bitsPerSymbol + bit;
            llr(llrIdx) = (dist1 - dist0) / noiseVar;
        end
    end

    % Hard decision from LLRs
    bits = double(llr < 0);
end

function metrics = calculate_demap_metrics(rxSymbols, constellation, grayMap, params)
% CALCULATE_DEMAP_METRICS - Calculate demapping quality metrics

    nSymbols = length(rxSymbols);
    M = length(constellation);

    % Find nearest constellation points
    distances = abs(repmat(rxSymbols, 1, M) - repmat(constellation.', nSymbols, 1));
    [minDist, minIdx] = min(distances, [], 2);

    % Ideal symbols (decision)
    idealSymbols = constellation(minIdx);

    % Error Vector Magnitude (EVM)
    errorVector = rxSymbols - idealSymbols;
    metrics.EVM_rms = sqrt(mean(abs(errorVector).^2)) / sqrt(mean(abs(idealSymbols).^2)) * 100;
    metrics.EVM_peak = max(abs(errorVector)) / sqrt(mean(abs(idealSymbols).^2)) * 100;

    % Modulation Error Ratio (MER)
    signalPower = mean(abs(idealSymbols).^2);
    errorPower = mean(abs(errorVector).^2);
    metrics.MER_dB = 10 * log10(signalPower / errorPower);

    % Average minimum distance (clustering quality)
    metrics.avgMinDist = mean(minDist);
    metrics.stdMinDist = std(minDist);

    % Constellation dispersion
    metrics.rxPowerSpread = std(abs(rxSymbols));
end
