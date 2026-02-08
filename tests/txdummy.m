%% Basic OFDM RX Chain Test Script
% This creates a simple transmitter and your receiver chain

clear all; close all; clc;

%% Parameters
nFFT = 64;              % FFT size (number of subcarriers)
nCP = 16;               % Cyclic prefix length
modOrder = 4;           % QPSK (4-QAM)
nSymbols = 10;          % Number of OFDM symbols to transmit
SNR_dB = 15;            % Signal-to-noise ratio in dB

%% ========== TRANSMITTER SIDE (Simplified) ==========
% In your actual project, this comes from your teammates

% 1. Generate random bits
nBitsPerSymbol = nFFT * log2(modOrder);
txBits = randi([0 1], nBitsPerSymbol * nSymbols, 1);

% 2. QAM Modulation
txSymbols = qammod(txBits, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
txSymbols = reshape(txSymbols, nFFT, nSymbols);

% 3. IFFT (convert to time domain)
txTimeDomain = ifft(txSymbols, nFFT);

% 4. Add Cyclic Prefix
txWithCP = [txTimeDomain(end-nCP+1:end, :); txTimeDomain];

% 5. Serialize for transmission
txSignal = txWithCP(:);

% 6. Add AWGN noise (simulating channel)
txSignal = awgn(txSignal, SNR_dB, 'measured');

fprintf('Transmitted %d OFDM symbols\n', nSymbols);

%% ========== YOUR RECEIVER CHAIN ==========

% 1. Reshape received signal
rxWithCP = reshape(txSignal, nFFT + nCP, nSymbols);

% 2. Remove Cyclic Prefix
rxTimeDomain = rxWithCP(nCP+1:end, :);

% 3. FFT Demodulation (convert back to frequency domain)
rxSymbols = fft(rxTimeDomain, nFFT);

% 4. Symbol Demapping (QAM demodulation)
rxSymbols_flat = rxSymbols(:);
rxBits = qamdemod(rxSymbols_flat, modOrder, 'OutputType', 'bit', 'UnitAveragePower', true);

% 5. BER Calculation
[numErrors, BER] = biterr(txBits, rxBits);

fprintf('\n--- RX CHAIN RESULTS ---\n');
fprintf('Number of bit errors: %d\n', numErrors);
fprintf('Bit Error Rate (BER): %.6f\n', BER);
fprintf('SNR: %d dB\n', SNR_dB);

%% ========== VISUALIZATION ==========

% Plot constellation diagrams
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
scatterplot(txSymbols(:));
title('Transmitted Constellation');
grid on;

subplot(1,3,2);
scatterplot(rxSymbols(:));
title('Received Constellation (After FFT)');
grid on;

subplot(1,3,3);
plot(1:length(txSignal), real(txSignal), 'b');
hold on;
plot(1:length(txSignal), imag(txSignal), 'r');
title('Time Domain Signal');
legend('Real', 'Imaginary');
grid on;

%% Test different SNR values
fprintf('\n--- BER vs SNR Test ---\n');
SNR_range = 0:5:30;
BER_results = zeros(size(SNR_range));

for idx = 1:length(SNR_range)
    % Add noise
    testSignal = awgn(txWithCP(:), SNR_range(idx), 'measured');
    
    % RX chain
    testRxWithCP = reshape(testSignal, nFFT + nCP, nSymbols);
    testRxTimeDomain = testRxWithCP(nCP+1:end, :);
    testRxSymbols = fft(testRxTimeDomain, nFFT);
    testRxBits = qamdemod(testRxSymbols(:), modOrder, 'OutputType', 'bit', 'UnitAveragePower', true);
    
    % Calculate BER
    [~, BER_results(idx)] = biterr(txBits, testRxBits);
end

% Plot BER curve
figure;
semilogy(SNR_range, BER_results, 'b-o', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title('BER vs SNR Performance');

