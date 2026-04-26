function [predictedSNR, predictedFading, confidence] = bold_channel_lstm(channelHistory, params)
% BOLD_CHANNEL_LSTM  LSTM-based channel predictor (replaces EMA+trend)
% Uses a trained LSTM network to predict future SNR values from the
% sequence of past measurements. Captures nonlinear temporal patterns
% that the simple EMA in channel_predictor.m cannot.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Inputs:
%   channelHistory - struct with field .snr (vector of past SNR dB values)
%   params         - struct with optional fields:
%       .predictionHorizon  (default 3)
%       .windowSize         (default 15)
%       .modelPath          (default 'channel_lstm.mat')
%
% Outputs:
%   predictedSNR       - [1 x H] predicted SNR for next H frames
%   predictedFading    - [1 x H] predicted fading (placeholder)
%   confidence         - [1 x H] prediction confidence scores
%
% Falls back to EMA+trend if the LSTM model is not yet trained.

    if nargin < 2, params = struct(); end
    if ~isfield(params, 'predictionHorizon'), params.predictionHorizon = 3; end
    if ~isfield(params, 'windowSize'),        params.windowSize = 15; end
    if ~isfield(params, 'modelPath')
        params.modelPath = fullfile(fileparts(mfilename('fullpath')), 'channel_lstm.mat');
    end

    H = params.predictionHorizon;
    W = params.windowSize;

    snrHist = channelHistory.snr(:).';
    N = length(snrHist);

    % Edge case
    if N == 0
        predictedSNR   = 20 * ones(1, H);
        predictedFading = ones(1, H);
        confidence      = 0.1 * ones(1, H);
        return;
    end

    % Take last W samples, normalise for LSTM input
    win = snrHist(max(1, N-W+1):N);
    Nw = length(win);

    % Normalisation parameters (must match training)
    snr_mean = 20;  % approximate centre of SNR range
    snr_std  = 10;  % approximate spread

    win_norm = (win - snr_mean) / snr_std;

    % --- Try LSTM prediction ---
    persistent lstmNet lstmPath
    useLSTM = false;

    if isempty(lstmNet) || isempty(lstmPath) || ~strcmp(lstmPath, params.modelPath)
        if exist(params.modelPath, 'file')
            lstmNet = SimpleLSTM(1, 32, 1, 1e-3);
            lstmNet.load_model(params.modelPath);
            lstmPath = params.modelPath;
            useLSTM = true;
        end
    else
        useLSTM = true;
    end

    if useLSTM && Nw >= 5
        predictedSNR = zeros(1, H);
        seq = win_norm;

        for h = 1:H
            % Feed sequence through LSTM
            X_seq = seq(:);      % column
            X_seq = X_seq.';     % [1 x seqLen] for inputSize=1
            y_norm = lstmNet.predict_seq(X_seq);

            % Denormalise
            y_snr = y_norm * snr_std + snr_mean;
            predictedSNR(h) = max(0, min(50, y_snr));

            % Autoregressive: append prediction for next step
            seq = [seq, y_norm];
            if length(seq) > W
                seq = seq(end-W+1:end);
            end
        end

        % Confidence from model — decreases with horizon
        confidence = zeros(1, H);
        for h = 1:H
            base_conf = max(0.3, min(0.95, Nw / W));
            confidence(h) = base_conf * exp(-0.15 * (h-1));
        end
    else
        % --- Fallback: EMA + linear trend (same as channel_predictor.m) ---
        [predictedSNR, ~, confidence] = channel_predictor(channelHistory, params);
    end

    % Fading prediction (placeholder - extend if fading history available)
    if isfield(channelHistory, 'fading') && ~isempty(channelHistory.fading)
        alpha = 0.3;
        fWin = channelHistory.fading(max(1,end-W+1):end);
        fEma = fWin(1);
        for i = 2:length(fWin)
            fEma = alpha*fWin(i) + (1-alpha)*fEma;
        end
        predictedFading = fEma * ones(1, H);
    else
        predictedFading = ones(1, H);
    end

    predictedSNR = max(0, min(50, predictedSNR));
    confidence = max(0, min(1, confidence));
end
