function feedback = quality_feedback(rxImage, channelMetrics, params)
% QUALITY_FEEDBACK - Lightweight perceptual quality feedback for adaptive transmission
%
% Generates compact feedback information for TX-side adaptation:
%   - Quality indicator (QI) score
%   - Recommended modulation order
%   - Recommended coding rate
%   - Channel quality index (CQI)
%
% Designed for low-overhead feedback in adaptive image transmission systems.
%
% Inputs:
%   rxImage       - Received/reconstructed image (or [] for channel-only feedback)
%   channelMetrics - Structure from OFDM demodulator with:
%       .EVM        - Error Vector Magnitude
%       .BER        - Bit Error Rate (optional)
%       .SNR_dB     - Estimated SNR (optional)
%   params        - Optional structure with:
%       .refImage   - Reference image for quality comparison
%       .targetPSNR - Target PSNR threshold (default: 30 dB)
%       .targetSSIM - Target SSIM threshold (default: 0.9)
%       .maxModOrder - Maximum modulation order to consider (default: 64)
%       .feedbackBits - Number of bits for feedback quantization (default: 8)
%
% Outputs:
%   feedback - Structure with:
%       .QI             - Quality Indicator (0-1, higher is better)
%       .CQI            - Channel Quality Index (0-15)
%       .recModOrder    - Recommended modulation order
%       .recCodingRate  - Recommended coding rate
%       .feedbackBits   - Compact bit representation of feedback
%       .urgent         - Flag indicating need for immediate adaptation

    %% Default parameters
    if nargin < 3
        params = struct();
    end
    if ~isfield(params, 'targetPSNR')
        params.targetPSNR = 30;
    end
    if ~isfield(params, 'targetSSIM')
        params.targetSSIM = 0.9;
    end
    if ~isfield(params, 'maxModOrder')
        params.maxModOrder = 64;
    end
    if ~isfield(params, 'feedbackBits')
        params.feedbackBits = 8;
    end

    %% 1. Calculate Channel Quality Index (CQI)
    feedback.CQI = calculate_cqi(channelMetrics);

    %% 2. Calculate Quality Indicator from image (if available)
    if ~isempty(rxImage) && isfield(params, 'refImage')
        feedback.QI = calculate_quality_indicator(rxImage, params.refImage, params);
        feedback.hasImageMetrics = true;
    else
        % Estimate QI from channel metrics alone
        feedback.QI = estimate_qi_from_channel(channelMetrics);
        feedback.hasImageMetrics = false;
    end

    %% 3. Determine recommended parameters
    [feedback.recModOrder, feedback.recCodingRate] = ...
        recommend_parameters(feedback.CQI, feedback.QI, params);

    %% 4. Check for urgent adaptation need
    feedback.urgent = check_urgent_adaptation(channelMetrics, feedback);

    %% 5. Generate compact feedback bitstream
    feedback.feedbackBits = encode_feedback(feedback, params.feedbackBits);

    %% 6. Additional diagnostic info
    feedback.timestamp = now;
    feedback.diagnostics = generate_diagnostics(channelMetrics, feedback);
end

%% Core Functions

function cqi = calculate_cqi(channelMetrics)
% CALCULATE_CQI - Map channel metrics to 4-bit CQI (0-15)
%
% CQI mapping based on approximate SNR ranges:
%   CQI 0:  SNR < 0 dB   (very poor)
%   CQI 1-3: SNR 0-6 dB  (poor)
%   CQI 4-6: SNR 6-12 dB (fair)
%   CQI 7-9: SNR 12-18 dB (good)
%   CQI 10-12: SNR 18-24 dB (very good)
%   CQI 13-15: SNR > 24 dB (excellent)

    % Estimate effective SNR
    if isfield(channelMetrics, 'SNR_dB')
        snr = channelMetrics.SNR_dB;
    elseif isfield(channelMetrics, 'estimatedSNR_dB')
        snr = channelMetrics.estimatedSNR_dB;
    elseif isfield(channelMetrics, 'EVM')
        % Convert EVM (%) to SNR: SNR_dB ≈ -20*log10(EVM/100)
        snr = -20 * log10(channelMetrics.EVM / 100);
    elseif isfield(channelMetrics, 'MER_dB')
        snr = channelMetrics.MER_dB;
    else
        % Default to mid-range if no metrics available
        snr = 15;
    end

    % Map SNR to CQI (0-15)
    if snr < 0
        cqi = 0;
    elseif snr > 30
        cqi = 15;
    else
        cqi = round(snr / 2);
    end
end

function qi = calculate_quality_indicator(rxImage, refImage, params)
% CALCULATE_QUALITY_INDICATOR - Compute composite quality indicator

    % Convert to double
    rxImage = double(rxImage);
    refImage = double(refImage);

    % PSNR component (normalized to 0-1)
    mse = mean((rxImage(:) - refImage(:)).^2);
    if mse == 0
        psnr = 100;  % Cap at very high value
    else
        psnr = 10 * log10(255^2 / mse);
    end
    psnr_score = min(psnr / 50, 1);  % Normalize: 50 dB = 1.0

    % SSIM component (using simplified computation)
    ssim_score = compute_fast_ssim(rxImage, refImage);

    % Perceptual sharpness (gradient magnitude)
    if ndims(rxImage) == 3
        rxGray = 0.299*rxImage(:,:,1) + 0.587*rxImage(:,:,2) + 0.114*rxImage(:,:,3);
        refGray = 0.299*refImage(:,:,1) + 0.587*refImage(:,:,2) + 0.114*refImage(:,:,3);
    else
        rxGray = rxImage;
        refGray = refImage;
    end

    rxGrad = sqrt(gradient_magnitude(rxGray));
    refGrad = sqrt(gradient_magnitude(refGray));
    sharpness_score = min(mean(rxGrad(:)) / (mean(refGrad(:)) + 1e-6), 1);

    % Weighted combination
    qi = 0.4 * psnr_score + 0.4 * ssim_score + 0.2 * sharpness_score;
    qi = max(0, min(1, qi));  % Clamp to [0, 1]
end

function qi = estimate_qi_from_channel(channelMetrics)
% ESTIMATE_QI_FROM_CHANNEL - Estimate image quality from channel metrics only

    % Base estimate from CQI
    cqi = calculate_cqi(channelMetrics);
    qi = cqi / 15;  % Normalize to 0-1

    % Adjust based on BER if available
    if isfield(channelMetrics, 'BER') && channelMetrics.BER > 0
        % High BER significantly reduces quality
        ber_penalty = min(channelMetrics.BER * 1000, 0.5);
        qi = qi * (1 - ber_penalty);
    end

    qi = max(0, min(1, qi));
end

function [modOrder, codingRate] = recommend_parameters(cqi, qi, params)
% RECOMMEND_PARAMETERS - Suggest modulation and coding for next transmission

    % Combined quality metric
    combined = 0.6 * (cqi / 15) + 0.4 * qi;

    % Modulation order selection (conservative approach)
    if combined < 0.2
        modOrder = 2;   % BPSK for very poor channel
    elseif combined < 0.4
        modOrder = 4;   % QPSK
    elseif combined < 0.6
        modOrder = 16;  % 16-QAM
    elseif combined < 0.8
        modOrder = 64;  % 64-QAM
    else
        modOrder = min(256, params.maxModOrder);  % 256-QAM for excellent channel
    end

    % Coding rate selection (higher rate = less redundancy)
    if combined < 0.3
        codingRate = 1/3;  % Heavy coding for poor channel
    elseif combined < 0.5
        codingRate = 1/2;
    elseif combined < 0.7
        codingRate = 2/3;
    elseif combined < 0.85
        codingRate = 3/4;
    else
        codingRate = 5/6;  % Light coding for excellent channel
    end
end

function urgent = check_urgent_adaptation(channelMetrics, feedback)
% CHECK_URGENT_ADAPTATION - Determine if immediate adaptation is needed

    urgent = false;

    % Check for severe quality degradation
    if feedback.QI < 0.3
        urgent = true;
    end

    % Check for high BER
    if isfield(channelMetrics, 'BER') && channelMetrics.BER > 0.01
        urgent = true;
    end

    % Check for CQI drop
    if feedback.CQI < 3
        urgent = true;
    end

    % Check for EVM spike
    if isfield(channelMetrics, 'EVM') && channelMetrics.EVM > 30
        urgent = true;
    end
end

function bits = encode_feedback(feedback, nBits)
% ENCODE_FEEDBACK - Encode feedback into compact bit representation
%
% Format for 8-bit feedback:
%   Bits 1-4: CQI (0-15)
%   Bits 5-6: Modulation recommendation (00=BPSK, 01=QPSK, 10=16QAM, 11=64QAM)
%   Bit 7:    Urgent flag
%   Bit 8:    Reserved / parity

    bits = zeros(nBits, 1);

    % CQI (4 bits)
    cqi_bits = de2bi(feedback.CQI, 4, 'left-msb');
    bits(1:4) = cqi_bits;

    % Modulation order (2 bits)
    switch feedback.recModOrder
        case 2
            mod_bits = [0 0];
        case 4
            mod_bits = [0 1];
        case 16
            mod_bits = [1 0];
        otherwise  % 64 or higher
            mod_bits = [1 1];
    end
    bits(5:6) = mod_bits;

    % Urgent flag (1 bit)
    bits(7) = feedback.urgent;

    % Parity bit (even parity)
    bits(8) = mod(sum(bits(1:7)), 2);
end

function diag = generate_diagnostics(channelMetrics, feedback)
% GENERATE_DIAGNOSTICS - Additional diagnostic information

    diag = struct();

    % Summarize channel state
    diag.channelState = 'Unknown';
    if feedback.CQI >= 12
        diag.channelState = 'Excellent';
    elseif feedback.CQI >= 8
        diag.channelState = 'Good';
    elseif feedback.CQI >= 4
        diag.channelState = 'Fair';
    else
        diag.channelState = 'Poor';
    end

    % Summarize quality
    diag.qualityState = 'Unknown';
    if feedback.QI >= 0.8
        diag.qualityState = 'Excellent';
    elseif feedback.QI >= 0.6
        diag.qualityState = 'Good';
    elseif feedback.QI >= 0.4
        diag.qualityState = 'Fair';
    else
        diag.qualityState = 'Poor';
    end

    % Throughput efficiency estimate
    bitsPerSymbol = log2(feedback.recModOrder);
    diag.spectralEfficiency = bitsPerSymbol * feedback.recCodingRate;

    % Recommendation summary
    diag.recommendation = sprintf('Use %d-QAM with rate %.2f coding', ...
        feedback.recModOrder, feedback.recCodingRate);
end

%% Helper Functions

function ssim = compute_fast_ssim(img1, img2)
% COMPUTE_FAST_SSIM - Fast SSIM approximation using block averaging

    blockSize = 8;
    K1 = 0.01;
    K2 = 0.03;
    L = 255;
    C1 = (K1*L)^2;
    C2 = (K2*L)^2;

    % Handle color images
    if ndims(img1) == 3
        img1 = 0.299*img1(:,:,1) + 0.587*img1(:,:,2) + 0.114*img1(:,:,3);
        img2 = 0.299*img2(:,:,1) + 0.587*img2(:,:,2) + 0.114*img2(:,:,3);
    end

    [h, w] = size(img1);
    nBlocksH = floor(h / blockSize);
    nBlocksW = floor(w / blockSize);

    ssim_vals = zeros(nBlocksH, nBlocksW);

    for i = 1:nBlocksH
        for j = 1:nBlocksW
            rowIdx = (i-1)*blockSize + 1 : i*blockSize;
            colIdx = (j-1)*blockSize + 1 : j*blockSize;

            block1 = img1(rowIdx, colIdx);
            block2 = img2(rowIdx, colIdx);

            mu1 = mean(block1(:));
            mu2 = mean(block2(:));
            sigma1_sq = var(block1(:));
            sigma2_sq = var(block2(:));
            sigma12 = mean((block1(:) - mu1) .* (block2(:) - mu2));

            ssim_vals(i,j) = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ...
                             ((mu1^2 + mu2^2 + C1) * (sigma1_sq + sigma2_sq + C2));
        end
    end

    ssim = mean(ssim_vals(:));
end

function gmag = gradient_magnitude(img)
% GRADIENT_MAGNITUDE - Calculate gradient magnitude of image

    [gx, gy] = gradient(double(img));
    gmag = gx.^2 + gy.^2;
end
