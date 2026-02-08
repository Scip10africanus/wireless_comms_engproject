function [rxBits, BER, metrics] = ofdm_receiver(rxSignal, txBits, params)
% OFDM_RECEIVER - Demodulates OFDM signal and calculates performance metrics
%
% Inputs:
%   rxSignal - Received time-domain signal (column vector)
%   txBits   - Original transmitted bits (for BER calculation)
%   params   - Structure with: nFFT, nCP, modOrder, nSymbols
%
% Outputs:
%   rxBits   - Recovered bits
%   BER      - Bit error rate
%   metrics  - Structure with performance metrics

    %% 1. Reshape and remove CP
    rxWithCP = reshape(rxSignal, params.nFFT + params.nCP, params.nSymbols);
    rxTimeDomain = rxWithCP(params.nCP+1:end, :);
    
    %% 2. FFT Demodulation
    rxSymbols = fft(rxTimeDomain, params.nFFT);
    
    %% 3. Symbol Demapping
    rxSymbols_flat = rxSymbols(:);
    rxBits = qamdemod(rxSymbols_flat, params.modOrder, ...
                      'OutputType', 'bit', 'UnitAveragePower', true);
    
    %% 4. Calculate BER
    [numErrors, BER] = biterr(txBits, rxBits);
    
    %% 5. Calculate additional metrics
    metrics.numErrors = numErrors;
    metrics.BER = BER;
    metrics.totalBits = length(txBits);
    
    % Estimate SNR from received symbols (simple method)
    % This is a rough estimate - you'll refine this later
    signalPower = mean(abs(rxSymbols(:)).^2);
    metrics.estimatedSNR_linear = signalPower;
    metrics.estimatedSNR_dB = 10*log10(signalPower);
    
    % EVM (Error Vector Magnitude) - measure of constellation quality
    idealSymbols = qammod(qamdemod(rxSymbols_flat, params.modOrder, 'UnitAveragePower', true), ...
                          params.modOrder, 'UnitAveragePower', true);
    errorVector = rxSymbols_flat - idealSymbols;
    metrics.EVM = sqrt(mean(abs(errorVector).^2)) / sqrt(mean(abs(idealSymbols).^2)) * 100;
    
end
