function [predictedSNR, predictedFading, confidence] = channel_predictor(channelHistory, params)
% CHANNEL_PREDICTOR  Predict future channel SNR using EMA + linear trend
% channelHistory.snr = past SNR measurements (dB)
% params.predictionHorizon, .windowSize, .emaAlpha (all optional)

    if nargin < 2, params = struct(); end
    if ~isfield(params, 'predictionHorizon'), params.predictionHorizon = 3; end
    if ~isfield(params, 'windowSize'),        params.windowSize = 10; end
    if ~isfield(params, 'emaAlpha'),          params.emaAlpha = 0.3; end

    H = params.predictionHorizon;
    W = params.windowSize;
    alpha = params.emaAlpha;

    snrHist = channelHistory.snr(:).';
    N = length(snrHist);

    % edge case: no data
    if N == 0
        predictedSNR = 20 * ones(1,H);
        predictedFading = ones(1,H);
        confidence = 0.1 * ones(1,H);
        return;
    end

    % take last W samples
    win = snrHist(max(1, N-W+1):N);
    Nw = length(win);

    %% EMA (exponential moving average)
    ema = win(1);
    for i = 2:Nw
        ema = alpha*win(i) + (1-alpha)*ema;
    end

    %% linear trend via least squares
    if Nw >= 3
        x = (1:Nw).';  y = win(:);
        xm = mean(x);  ym = mean(y);
        slope = sum((x-xm).*(y-ym)) / (sum((x-xm).^2) + eps);
        yhat = ym + slope*(x - xm);
        resVar = mean((y - yhat).^2);
    else
        slope = 0;
        resVar = var(win) + eps;
    end

    %% predict
    predictedSNR = zeros(1,H);
    for h = 1:H
        predictedSNR(h) = ema + slope*h;
    end
    predictedSNR = max(0, min(50, predictedSNR));

    %% fading prediction (just EMA if data available)
    if isfield(channelHistory, 'fading') && ~isempty(channelHistory.fading)
        fWin = channelHistory.fading(max(1,end-W+1):end);
        fEma = fWin(1);
        for i = 2:length(fWin)
            fEma = alpha*fWin(i) + (1-alpha)*fEma;
        end
        predictedFading = fEma * ones(1,H);
    else
        predictedFading = ones(1,H);
    end

    %% confidence (decreases with horizon, increases with data)
    confidence = zeros(1,H);
    for h = 1:H
        predVar = resVar * (1 + 0.5*h);
        confidence(h) = 1 / (1 + sqrt(predVar));
        confidence(h) = confidence(h) * max(0.3, min(1.0, Nw/W));
    end
    confidence = max(0, min(1, confidence));
end
