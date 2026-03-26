function cleaned_symbols = bold_symbol_denoiser(noisy_symbols, snr_dB, modOrder, modelPath)
% BOLD_SYMBOL_DENOISER  MLP-based QAM symbol denoiser
% Denoises received QAM constellation symbols BEFORE hard-decision demapping.
% Operates at the physical layer to directly reduce BER.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Theory:
%   Traditional receiver: noisy symbol -> nearest constellation point (hard decision)
%   ML receiver: noisy symbol -> MLP denoiser -> cleaner symbol -> nearest point
%   The MLP learns the channel statistics and can denoise better than simple
%   nearest-neighbour, especially at low SNR where symbols overlap.
%
% Inputs:
%   noisy_symbols - Complex column vector of received QAM symbols
%   snr_dB        - Estimated SNR in dB (scalar)
%   modOrder      - Modulation order: 2(BPSK), 4(QPSK), 16(16QAM), 64(64QAM)
%   modelPath     - Path to trained model .mat (optional)
%
% Outputs: 
%   cleaned_symbols - Denoised complex symbols (same size as input)
%
% The network input is [Re(sym), Im(sym), SNR_norm, modOrder_norm]
% The output is [Re(clean), Im(clean)]

    if nargin < 4
        modelPath = fullfile(fileparts(mfilename('fullpath')), ...
            sprintf('symbol_denoiser_mod%d.mat', modOrder));
    end

    
    syms = noisy_symbols(:);
    nSym = length(syms);

    % --- Build input features ---
    %   [Re, Im, normalised_SNR, normalised_modOrder]
    snr_norm = snr_dB / 40;     % normalise to ~[0, 1]
    mod_norm = log2(modOrder) / 6;  % log2(64)=6

    X = zeros(4, nSym);
    X(1, :) = real(syms).';
    X(2, :) = imag(syms).';
    X(3, :) = snr_norm;
    X(4, :) = mod_norm;

    % --- Try ML denoising ---
    persistent denoisers
    if isempty(denoisers)
        denoisers = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end

    modKey = int32(modOrder);
    useML = false;

    if ~denoisers.isKey(modKey)
        if exist(modelPath, 'file')
            net = SimpleNet([4 64 32 2], 1e-3);
            net.load_model(modelPath);
            denoisers(modKey) = net;
            useML = true;
        end
    else
        useML = true;
    end

    if useML
        net = denoisers(modKey);
        Y = net.predict(X);    % [2 x nSym]
        cleaned_symbols = complex(Y(1,:), Y(2,:)).';
    else
        % --- Fallback: soft-decision nearest neighbour with averaging ---
        cleaned_symbols = soft_denoise_fallback(syms, modOrder, snr_dB);
    end
end

%% Fallback: weighted average toward nearest constellation point
function cleaned = soft_denoise_fallback(syms, modOrder, snr_dB)
    % Generate ideal constellation
    constellation = generate_constellation(modOrder);

    % Soft factor: higher SNR -> trust received more; lower -> pull to nearest
    alpha = min(0.9, max(0.3, snr_dB / 30));

    cleaned = zeros(size(syms));
    for i = 1:length(syms)
        dists = abs(syms(i) - constellation).^2;
        [~, nearest] = min(dists);
        % Weighted average: alpha * received + (1-alpha) * nearest point
        cleaned(i) = alpha * syms(i) + (1-alpha) * constellation(nearest);
    end
end

%% Generate standard QAM constellation
function C = generate_constellation(M)
    switch M
        case 2  % BPSK
            C = [-1; 1];
        case 4  % QPSK
            C = [1+1j; 1-1j; -1+1j; -1-1j] / sqrt(2);
        case 16 % 16-QAM
            sqM = 4;
            [I, Q] = meshgrid(-(sqM-1):2:(sqM-1), -(sqM-1):2:(sqM-1));
            C = (I(:) + 1j*Q(:)) / sqrt(10);
        case 64 % 64-QAM
            sqM = 8;
            [I, Q] = meshgrid(-(sqM-1):2:(sqM-1), -(sqM-1):2:(sqM-1));
            C = (I(:) + 1j*Q(:)) / sqrt(42);
        otherwise
            C = [-1; 1];
    end
end
