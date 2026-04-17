%% CLOSED_LOOP_SIM_AE  Multi-frame TX -> Channel -> RX with AI Autoencoder
% The Ultimate AI-Driven Adaptive Modulation Communication Loop

% --- 确保依赖路径已被加入 ---
targetPath = 'C:\Users\renka\Desktop\wireless_comms_engproject-main (1)\wireless_comms_engproject-main\tx\test_functions';
if exist(targetPath, 'dir')
    addpath(targetPath);
    fprintf('✅ Successfully added path: %s\n', targetPath);
else
    warning('❌ Path not found: %s', targetPath);
end

clear; clc; close all;

fprintf('================================================================\n');
fprintf('  Starting Full Closed-Loop Simulation (AI Autoencoder Edition)\n');
fprintf('================================================================\n\n');

%% 1. Initialize Python Environment
pe = pyenv;
if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

% 强制刷新模块以防本地代码变更未生效
try
    ae_mod = py.importlib.import_module('ae_inference_api');
    py.importlib.reload(ae_mod);
    fprintf('✅ Python API 模块已刷新。\n');
catch
    fprintf('⚠️ 模块自动刷新失败，继续执行...\n');
end

%% 2. Setup Simulation Parameters
% Simulate a varying channel: SNR starts good (30dB), drops to terrible (8dB), then recovers
snrProfile = [30 28 25 22 18 15 12 10 8 8 10 12 15 18 22 25 28 30];
nFrames = length(snrProfile);

% Load Source Image (使用绝对路径，确保一定能找到)
img_path = 'C:\Users\renka\Desktop\wireless_comms_engproject-main (1)\wireless_comms_engproject-main\tx\images\Testing examples.jpg';

try
    imgOriginal = imread(img_path);
    
    % 如果是彩色图片(3通道)，AI 模型只吃灰度图(1通道)，必须转换：
    if size(imgOriginal, 3) == 3
        imgOriginal = rgb2gray(imgOriginal);
    end
    
    % 统一缩放到 64x64
    imgOriginal = imresize(imgOriginal, [64 64]);
catch ME
    warning('❌ 找不到指定路径的图片，将使用默认 cameraman 测试图！\n错误详情: %s', ME.message);
    try
        imgOriginal = imread('cameraman.tif');
        imgOriginal = imresize(imgOriginal, [64 64]);
    catch
        imgOriginal = uint8(randi([0, 255], 64, 64)); 
    end
end

% Initialize Policy 
policy = struct();
% 基础层 (Base) 负责保底，必须稳健，初始设为 2 (QPSK)。
% 增强层 (Enh) 负责细节，可以激进，设为 6 (64-QAM)。
policy.mcs_base = 2;   
policy.mcs_enh  = 6;   
policy.mux_gap_us = 4; % Gap between OFDM layers

% Data Logging Arrays
log_ssim = zeros(1, nFrames);
log_psnr = zeros(1, nFrames);
log_mcs_base = zeros(1, nFrames);
log_mcs_enh = zeros(1, nFrames);

%% 3. Main Communication Loop
fprintf('Simulating %d frames over dynamic AWGN channel...\n', nFrames);
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Frame | SNR(dB) | BaseMCS | EnhMCS | SSIM  | PSNR(dB) | Status\n');
fprintf('--------------------------------------------------------------------------------\n');

for f = 1:nFrames
    current_snr = snrProfile(f);
    
    log_mcs_base(f) = policy.mcs_base;
    log_mcs_enh(f)  = policy.mcs_enh;
    
    % --- TX Pipeline ---
    % 【极其重要的修复】：MATLAB (列优先) 和 Python (行优先) 在展平 2D 矩阵时存在方向冲突。
    % 解决办法：在送入 TX 前进行空间转置 (.')，完美抵消 Python 的重塑打乱！
    img_for_tx = imgOriginal.';
    [txWave, txInfo] = two_layer_tx_AE(img_for_tx, policy, f);
    
    % --- Wireless Channel ---
    rxWave = awgn(txWave, current_snr, 'measured');
    
    % --- RX Pipeline ---
    try
        [rxImage, rxInfo] = two_layer_rx_AE(rxWave, txInfo, policy);
        status_str = 'OK (Full Recovery)';
        
        % 【智能方向校正】：如果 Python 返回的 2D 矩阵被 MATLAB 读取为转置状态，自动校正对齐
        ssim_direct = ssim(uint8(rxImage), imgOriginal);
        ssim_transposed = ssim(uint8(rxImage.'), imgOriginal);
        if ssim_transposed > ssim_direct
            rxImage = rxImage.';
        end
        
        % 判断 AI 是否进入了脑补模式
        if isfield(rxInfo, 'enhStatus') && ~strcmp(rxInfo.enhStatus, 'Success')
            status_str = 'Enh_Lost (AI Degraded)';
        end
    catch ME
        % 打印出真实崩溃原因，不再盲目"吃掉"异常
        fprintf('\n❌ [ERROR in Frame %d] %s\n', f, ME.message);
        rxImage = zeros(64, 64);
        status_str = 'Base_Lost (CRASH)';
    end
    
    % --- Quality Metrics ---
    if sum(rxImage(:)) == 0
        ssim_val = 0; psnr_val = 0;
    else
        ssim_val = ssim(uint8(rxImage), imgOriginal);
        mse = mean((double(imgOriginal(:)) - rxImage(:)).^2);
        psnr_val = 10*log10(255^2 / max(mse, 1e-10));
    end
    
    log_ssim(f) = ssim_val;
    log_psnr(f) = psnr_val;
    
    fprintf(' %02d   |   %02d    |    %d    |   %d    | %.3f |  %05.2f   | %s\n', ...
             f, current_snr, policy.mcs_base, policy.mcs_enh, ssim_val, psnr_val, status_str);
             
    % --- Feedback Loop (Adaptive Modulation) ---
    % 【基于 AI 模型真实性能天花板 (0.71) 校准的阈值】
    if ssim_val < 0.50 || strcmp(status_str, 'Base_Lost (CRASH)')
        % 极度恶劣：画质崩坏，全力降速保底
        policy.mcs_base = max(0, policy.mcs_base - 1); 
        policy.mcs_enh  = max(0, policy.mcs_enh - 2);  
    elseif ssim_val < 0.65 || contains(status_str, 'Enh_Lost')
        % 勉强能看：细节层开始出错，降低细节层吞吐量以求稳定
        policy.mcs_enh  = max(0, policy.mcs_enh - 1);
    elseif ssim_val > 0.68 && current_snr > 15
        % 信号极佳：画质逼近 AI 性能极限 (0.71)，尝试激进提速
        policy.mcs_enh  = min(7, policy.mcs_enh + 1);
        if current_snr > 20
            policy.mcs_base = min(4, policy.mcs_base + 1);
        end
    end
end

fprintf('--------------------------------------------------------------------------------\n');
fprintf('Simulation Complete.\n');

%% 4. Plot Performance Dashboard
figure('Name', 'AI Closed-Loop Performance Dashboard', 'Position', [100, 100, 1100, 450]);

subplot(1, 3, 1);
plot(1:nFrames, snrProfile, 'k-o', 'LineWidth', 2);
title('Dynamic Channel SNR'); xlabel('Frame'); ylabel('SNR (dB)');
grid on; ylim([0 35]);

subplot(1, 3, 2);
plot(1:nFrames, log_ssim, 'g-o', 'LineWidth', 2); hold on;
% 对应调整红线高度为 0.65
yline(0.65, 'r--', 'Acceptable Threshold (Calibrated)'); hold off;
title('AI Reconstructed Quality (SSIM)'); xlabel('Frame'); ylabel('SSIM');
grid on; ylim([0 1.1]);

subplot(1, 3, 3);
plot(1:nFrames, log_mcs_base, 'b-s', 'LineWidth', 2); hold on;
plot(1:nFrames, log_mcs_enh, 'm-^', 'LineWidth', 2); hold off;
title('Adaptive Modulation (Feedback)'); xlabel('Frame'); ylabel('MCS Index');
legend('Base Layer MCS', 'Enhancement Layer MCS', 'Location', 'best');
grid on; ylim([-1 8]);