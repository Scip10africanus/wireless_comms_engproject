function [rxSymbols, channelEst, metrics] = ofdm_demodulator(rxSignal, params)
% OFDM_DEMODULATOR - Advanced OFDM demodulation with channel estimation
%
% Performs OFDM demodulation including:
%   - CP removal
%   - FFT processing
%   - Pilot-based channel estimation
%   - Zero-forcing or MMSE equalization
%
% Inputs:
%   rxSignal - Received time-domain signal (column vector)
%   params   - Structure with fields:
%       .nFFT         - FFT size (number of subcarriers)
%       .nCP          - Cyclic prefix length
%       .nSymbols     - Number of OFDM symbols
%       .pilotIndices - Indices of pilot subcarriers (optional)
%       .pilotValues  - Known pilot symbol values (optional)
%       .equalizer    - 'ZF' (zero-forcing) or 'MMSE' (default: 'ZF')
%       .noiseVar     - Noise variance for MMSE (optional)
%
% Outputs:
%   rxSymbols  - Equalized frequency-domain symbols (nFFT x nSymbols)
%   channelEst - Estimated channel frequency response
%   metrics    - Structure with demodulation metrics

    %% Input validation and defaults
    if ~isfield(params, 'equalizer')
        params.equalizer = 'ZF';
    end
    if ~isfield(params, 'noiseVar')
        params.noiseVar = 0.01;
    end

    %% 1. Reshape and remove cyclic prefix
    symbolLength = params.nFFT + params.nCP;
    nSymbols = floor(length(rxSignal) / symbolLength);

    % Truncate signal to fit complete symbols
    rxSignal = rxSignal(1:nSymbols * symbolLength);
    rxWithCP = reshape(rxSignal, symbolLength, nSymbols);

    % Remove CP
    rxTimeDomain = rxWithCP(params.nCP + 1:end, :);

    %% 2. FFT Demodulation
    rxFreqDomain = fft(rxTimeDomain, params.nFFT);

    %% 3. Channel Estimation
    if isfield(params, 'pilotIndices') && isfield(params, 'pilotValues')
        % Pilot-based channel estimation
        channelEst = channel_estimation_pilots(rxFreqDomain, params);
    else
        % Blind estimation (simple averaging across symbols)
        channelEst = channel_estimation_blind(rxFreqDomain, params);
    end

    %% 4. Equalization
    switch upper(params.equalizer)
        case 'ZF'
            % Zero-Forcing equalization
            rxSymbols = zf_equalize(rxFreqDomain, channelEst);
        case 'MMSE'
            % MMSE equalization
            rxSymbols = mmse_equalize(rxFreqDomain, channelEst, params.noiseVar);
        otherwise
            warning('Unknown equalizer type. Using ZF.');
            rxSymbols = zf_equalize(rxFreqDomain, channelEst);
    end

    %% 5. Calculate demodulation metrics
    metrics = calculate_demod_metrics(rxFreqDomain, rxSymbols, channelEst);
    metrics.nSymbols = nSymbols;
    metrics.nFFT = params.nFFT;
end

%% Helper Functions

function H_est = channel_estimation_pilots(rxFreq, params)
% CHANNEL_ESTIMATION_PILOTS - LS channel estimation using pilots

    pilotIdx = params.pilotIndices;
    pilotVals = params.pilotValues(:);
    nFFT = params.nFFT;
    nSymbols = size(rxFreq, 2);

    % LS estimation at pilot positions
    H_pilot = zeros(length(pilotIdx), nSymbols);
    for sym = 1:nSymbols
        H_pilot(:, sym) = rxFreq(pilotIdx, sym) ./ pilotVals;
    end

    % Interpolate to all subcarriers
    H_est = zeros(nFFT, nSymbols);
    allIdx = (1:nFFT)';

    for sym = 1:nSymbols
        % Linear interpolation
        H_est(:, sym) = interp1(pilotIdx, H_pilot(:, sym), allIdx, 'linear', 'extrap');
    end

    % Optional: Apply smoothing filter
    H_est = movmean(H_est, 3, 1);
end

function H_est = channel_estimation_blind(rxFreq, params)
% CHANNEL_ESTIMATION_BLIND - Blind channel estimation
% Assumes channel is relatively constant across symbols

    nFFT = params.nFFT;
    nSymbols = size(rxFreq, 2);

    % Use average magnitude as rough estimate (assumes unit power TX)
    avgPower = mean(abs(rxFreq).^2, 2);
    H_mag = sqrt(avgPower);

    % Phase estimation (simple approach)
    avgPhase = angle(mean(rxFreq, 2));

    H_est = repmat(H_mag .* exp(1j * avgPhase), 1, nSymbols);
end

function rxEq = zf_equalize(rxFreq, H_est)
% ZF_EQUALIZE - Zero-forcing equalization

    % Avoid division by zero
    H_safe = H_est;
    H_safe(abs(H_safe) < 1e-6) = 1e-6;

    rxEq = rxFreq ./ H_safe;
end

function rxEq = mmse_equalize(rxFreq, H_est, noiseVar)
% MMSE_EQUALIZE - MMSE equalization

    H_conj = conj(H_est);
    H_sq = abs(H_est).^2;

    % MMSE filter: H* / (|H|^2 + sigma^2)
    rxEq = (H_conj .* rxFreq) ./ (H_sq + noiseVar);
end

function metrics = calculate_demod_metrics(rxFreq, rxEq, H_est)
% CALCULATE_DEMOD_METRICS - Calculate demodulation quality metrics

    % Channel statistics
    metrics.channelGain_dB = 20 * log10(mean(abs(H_est(:))));
    metrics.channelVar = var(abs(H_est(:)));

    % Received signal power
    metrics.rxPower_dB = 10 * log10(mean(abs(rxFreq(:)).^2));

    % Equalized signal power (should be ~1 for unit power TX)
    metrics.eqPower = mean(abs(rxEq(:)).^2);

    % Frequency selectivity measure
    H_avg = mean(abs(H_est), 2);
    metrics.freqSelectivity = std(H_avg) / mean(H_avg);
end
