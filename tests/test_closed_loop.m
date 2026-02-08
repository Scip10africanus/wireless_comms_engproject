%% Closed-Loop Adaptive Transmission Test
%
% Demonstrates the feedback loop:
%   Frame N: TX → Channel → RX → Feedback
%   Frame N+1: TX adapts based on feedback
%
% Tests:
%   1. Channel degradation → TX reduces modulation
%   2. Channel improvement → TX increases throughput
%   3. Sudden drop → Urgent adaptation

clear all; close all; clc;

addpath('..');
addpath('../tx');
addpath('../channel');

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║         Closed-Loop Adaptive Transmission Test                ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

%% Setup
params = config_two_layer();
params.verbose = false;

% Test image
testImage = create_test_image(64, 64);
params.refImage = testImage;

%% Simulation: Varying SNR over time
nFrames = 15;

% SNR profile: good → bad → good
snrProfile = [25, 25, 20, 15, 10, 8, 8, 10, 15, 20, 25, 25, 28, 28, 30];

% Storage
results_log = struct();
results_log.snr = snrProfile;
results_log.baseMod = zeros(1, nFrames);
results_log.enhMod = zeros(1, nFrames);
results_log.ssim = zeros(1, nFrames);
results_log.psnr = zeros(1, nFrames);
results_log.adaptations = cell(1, nFrames);

fprintf('Running %d frames with varying SNR...\n\n', nFrames);
fprintf('Frame | SNR(dB) | Base Mod | Enh Mod | SSIM   | PSNR   | Adaptation\n');
fprintf('------+---------+----------+---------+--------+--------+-----------\n');

currentParams = params;

for frame = 1:nFrames
    %% Transmit
    [txSignal, txInfo] = two_layer_tx_pipeline(testImage, currentParams);

    %% Channel (with current SNR)
    channelParams = struct();
    channelParams.channelType = 'AWGN';
    channelParams.SNR_dB = snrProfile(frame);

    [rxSignal, ~] = channel_model(txSignal, channelParams);

    %% Receive
    [rxImage, feedback, rxResults] = two_layer_rx_pipeline(rxSignal, currentParams);

    %% Calculate quality
    ssim_val = calculate_ssim(testImage, rxImage);
    psnr_val = calculate_psnr(testImage, rxImage);

    %% Log results
    results_log.baseMod(frame) = currentParams.baseMod;
    results_log.enhMod(frame) = currentParams.enhancementMod;
    results_log.ssim(frame) = ssim_val;
    results_log.psnr(frame) = psnr_val;

    %% Adapt TX for next frame
    [newParams, adaptInfo] = adaptive_tx_controller(currentParams, feedback);

    if ~isempty(adaptInfo.changes)
        adaptStr = strjoin(adaptInfo.changes, ', ');
    else
        adaptStr = '-';
    end
    results_log.adaptations{frame} = adaptStr;

    %% Print status
    fprintf('%5d | %7d | %8d | %7d | %6.4f | %6.2f | %s\n', ...
        frame, snrProfile(frame), currentParams.baseMod, ...
        currentParams.enhancementMod, ssim_val, psnr_val, adaptStr);

    %% Update params for next frame
    currentParams = newParams;
end

%% Plot Results
figure('Position', [100, 100, 1200, 600], 'Name', 'Closed-Loop Adaptation');

% SNR Profile
subplot(2, 2, 1);
plot(1:nFrames, snrProfile, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'k');
xlabel('Frame');
ylabel('SNR (dB)');
title('Channel Condition (SNR)');
grid on;
ylim([0, 35]);

% Modulation Orders
subplot(2, 2, 2);
semilogy(1:nFrames, results_log.baseMod, 'b-s', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:nFrames, results_log.enhMod, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Frame');
ylabel('Modulation Order');
title('TX Adaptation');
legend('Base (QPSK→16QAM)', 'Enhancement (16QAM→256QAM)', 'Location', 'best');
grid on;
ylim([1, 300]);

% Quality Metrics
subplot(2, 2, 3);
yyaxis left;
plot(1:nFrames, results_log.ssim, 'b-o', 'LineWidth', 2);
ylabel('SSIM');
ylim([0, 1]);

yyaxis right;
plot(1:nFrames, results_log.psnr, 'r-s', 'LineWidth', 2);
ylabel('PSNR (dB)');

xlabel('Frame');
title('Quality vs Frame');
grid on;
legend('SSIM', 'PSNR');

% Summary
subplot(2, 2, 4);
axis off;
text(0.1, 0.9, 'Closed-Loop Summary:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.75, sprintf('Frames: %d', nFrames), 'FontSize', 10);
text(0.1, 0.60, sprintf('SNR range: %d - %d dB', min(snrProfile), max(snrProfile)), 'FontSize', 10);
text(0.1, 0.45, sprintf('Avg SSIM: %.3f', mean(results_log.ssim)), 'FontSize', 10);
text(0.1, 0.30, sprintf('Avg PSNR: %.1f dB', mean(results_log.psnr)), 'FontSize', 10);

nAdapt = sum(~strcmp(results_log.adaptations, '-'));
text(0.1, 0.15, sprintf('Adaptations: %d/%d frames', nAdapt, nFrames), 'FontSize', 10);

sgtitle('Closed-Loop Adaptive Transmission');

%% Show adaptation timeline
fprintf('\n\nAdaptation Timeline:\n');
fprintf('════════════════════\n');
for frame = 1:nFrames
    if ~strcmp(results_log.adaptations{frame}, '-')
        fprintf('Frame %2d (SNR=%2ddB): %s\n', frame, snrProfile(frame), results_log.adaptations{frame});
    end
end

fprintf('\nTest complete!\n');

%% Helper Functions

function img = create_test_image(h, w)
    [X, Y] = meshgrid(1:w, 1:h);
    img = 50 + 0.5*X + 0.5*Y;
    cx = w/2; cy = h/2; r = min(w,h)/4;
    circle = sqrt((X-cx).^2 + (Y-cy).^2) < r;
    img(circle) = 200;
    img = uint8(max(0, min(255, img + 5*randn(h, w))));
end

function ssim_val = calculate_ssim(ref, test)
    ref = double(ref); test = double(test);
    K1 = 0.01; K2 = 0.03; L = 255;
    C1 = (K1*L)^2; C2 = (K2*L)^2;
    mu1 = mean(ref(:)); mu2 = mean(test(:));
    s1 = std(ref(:)); s2 = std(test(:));
    s12 = mean((ref(:)-mu1).*(test(:)-mu2));
    ssim_val = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1^2+s2^2+C2));
end

function psnr_val = calculate_psnr(ref, test)
    mse = mean((double(ref(:)) - double(test(:))).^2);
    if mse == 0, psnr_val = 100; else, psnr_val = 10*log10(255^2/mse); end
end
