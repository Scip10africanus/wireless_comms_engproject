%% Test Script for AI-Driven Transmitter (test_tx_AE.m)
% This script tests if MATLAB can successfully call the Python AE model
% and generate the time-domain OFDM waveform.

clear; clc; close all;

fprintf('==================================================\n');
fprintf('  Testing AI-Driven TX Pipeline (AE -> OFDM)\n');
fprintf('==================================================\n\n');

%% 1. Ensure Python environment and path are configured in MATLAB
% Add current folder to Python's sys.path so it can find ae_inference_api.py
pe = pyenv;
fprintf('[0/3] Python Environment Check:\n');
fprintf('      Executable: %s\n', pe.Executable);
fprintf('      Version: %s\n\n', pe.Version);

if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

% Try importing to catch any missing libraries (like torch) early
try
    api_mod = py.importlib.import_module('ae_inference_api');
    py.importlib.reload(api_mod);
catch ME
    fprintf('PYTHON IMPORT ERROR DETECTED:\n');
    disp(ME.message);
    fprintf('\n------------------------------------------------------\n');
    fprintf('[How to fix]\n');
    fprintf('If you see "No module named torch" (or similar),\n');
    fprintf('MATLAB is using the wrong Python interpreter. Point it to your virtual environment (venv).\n\n');
    fprintf('Run a command like this in the MATLAB command window (replace with your actual path):\n');
    fprintf('Windows: pyenv(''Version'', ''C:\\your_project_path\\venv\\Scripts\\python.exe'')\n');
    fprintf('Mac/Linux: pyenv(''Version'', ''/your_project_path/venv/bin/python'')\n');
    fprintf('------------------------------------------------------\n');
    error('Stopping execution due to Python import failure.');
end

%% 2. Prepare a 64x64 Grayscale Test Image
fprintf('[1/3] Preparing 64x64 test image...\n');
try
    img = imread('cameraman.tif'); % Built-in MATLAB image
    img = imresize(img, [64 64]);
catch
    % Fallback to a random image if cameraman is missing
    img = uint8(randi([0, 255], 64, 64)); 
end

%% 3. Setup a dummy Policy
fprintf('[2/3] Setting up transmission policy...\n');
policy = struct();
policy.mcs_base = 2;   % 2 = QPSK (Robust)
policy.mcs_enh  = 6;   % 6 = 64-QAM (High throughput)
policy.mux_gap_us = 4; % 4 microsecond gap between layers

%% 4. Execute the TX Pipeline
fprintf('[3/3] Calling Python AE API and modulating waveform...\n');
try
    tic;
    [txWave, txInfo] = two_layer_tx_AE(img, policy, 1);
    elapsedTime = toc;
    
    fprintf('\nSUCCESS! TX Pipeline executed in %.3f seconds.\n', elapsedTime);
    fprintf('   - Base Layer Payload: %d bytes\n', txInfo.basePktLen);
    fprintf('   - Enh Layer Payload:  %d bytes\n', txInfo.enhPktLen);
    fprintf('   - Total Waveform Length: %d samples\n', length(txWave));
    
    % 5. Plot the generated OFDM waveform
    figure('Name', 'AE TX Waveform', 'Position', [100, 100, 800, 400]);
    plot(abs(txWave), 'b');
    title(sprintf('AE-Encoded OFDM Transmit Waveform (Base MCS=%d, Enh MCS=%d)', policy.mcs_base, policy.mcs_enh));
    xlabel('Sample Index');
    ylabel('Magnitude |txWave|');
    grid on;
    
catch ME
    fprintf('\nFAILED! An error occurred:\n');
    disp(ME.message);
end