%% AI Training Data Generator for Adaptive Modulation Controller
% Bold (Unubold) - EEME30003 Major Project
%
% This generates training data for the AI agent by running
% simulations across diverse SNR scenarios.
%
% Output: ai_training_data.csv

clear all; close all; clc;

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║       AI Training Data Generation - Bold''s ML Pipeline       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

%% Add paths to team's code
% IMPORTANT: Adjust these paths based on your project structure!
addpath('..');           % Parent directory
addpath('../tx');        % Transmitter code
addpath('../rx');        % Receiver code  
addpath('../channel');   % Channel models

fprintf('[1/5] Checking paths...\n');

% Verify required files exist
if ~exist('config_two_layer', 'file')
    error('Cannot find config_two_layer.m\nPlease adjust the addpath() lines above!');
end

if ~exist('two_layer_tx_pipeline', 'file')
    error('Cannot find two_layer_tx_pipeline.m\nPlease adjust the addpath() lines above!');
end

if ~exist('two_layer_rx_pipeline', 'file')
    error('Cannot find two_layer_rx_pipeline.m\nPlease adjust the addpath() lines above!');
end

if ~exist('channel_model', 'file')
    error('Cannot find channel_model.m\nPlease adjust the addpath() lines above!');
end

fprintf('  ✓ All required files found\n\n');

%% Configuration
fprintf('[2/5] Loading configuration...\n');
params = config_two_layer();
params.verbose = false;
fprintf('  ✓ Configuration loaded\n\n');

%% Create test images
fprintf('[3/5] Creating test images...\n');
nTestImages = 5;
testImages = cell(1, nTestImages);

% Image 1: Gradient
[X, Y] = meshgrid(1:64, 1:64);
testImages{1} = uint8(X * 255 / 64);

% Image 2: Circle
cx = 32; cy = 32; r = 16;
circle = sqrt((X-cx).^2 + (Y-cy).^2) < r;
testImages{2} = uint8(zeros(64, 64));
testImages{2}(circle) = 200;

% Image 3: Checkerboard
testImages{3} = uint8(checkerboard(8, 4, 4) * 255);

% Image 4: Random texture
testImages{4} = uint8(rand(64, 64) * 255);

% Image 5: Edge pattern
testImages{5} = uint8(zeros(64, 64));
testImages{5}(1:32, :) = 50;
testImages{5}(33:64, :) = 200;

fprintf('  ✓ Created %d test images (64x64)\n\n', nTestImages);

%% Define SNR profiles
fprintf('[4/5] Defining SNR profiles...\n');
snrProfiles = {
    % Urban (highly variable)
    [25 20 15 12 10 8 10 12 15 18 20 22 25 23 20 18 15 12 10 15 20 25],
    
    % Suburban (moderate variation)
    [28 26 24 22 20 18 16 15 16 18 20 22 24 26 28],
    
    % Indoor (stable with sudden drops)
    [25 25 25 25 10 8 25 25 25 25 25 12 10 25 25],
    
    % Highway (slow fade)
    [30 29 28 27 26 25 24 23 22 21 20 21 22 23 24 25],
    
    % Extreme degradation
    [30 25 20 15 10 8 6 5 6 8 10 12 15 18 20 25 30],
    
    % High quality stable
    [30 30 30 28 28 28 30 30 30 28 28 30 30],
    
    % Low quality stable
    [10 10 10 12 12 10 10 10 12 10 10]
};

fprintf('  ✓ Created %d SNR profiles\n\n', length(snrProfiles));

%% Data collection
fprintf('[5/5] Running simulations...\n');
fprintf('      This may take 5-15 minutes...\n\n');

allData = [];
totalFrames = 0;

for imageIdx = 1:nTestImages
    testImage = testImages{imageIdx};
    params.refImage = testImage;
    
    fprintf('  Image %d/%d:\n', imageIdx, nTestImages);
    
    for profileIdx = 1:length(snrProfiles)
        snrProfile = snrProfiles{profileIdx};
        nFrames = length(snrProfile);
        
        fprintf('    Profile %d/%d: %d frames ... ', ...
            profileIdx, length(snrProfiles), nFrames);
        
        % Reset parameters for each profile
        currentParams = params;
        currentParams.baseMod = 4;           % Start with QPSK
        currentParams.enhancementMod = 64;   % Start with 64-QAM
        
        for frame = 1:nFrames
            try
                % Transmit
                [txSignal, ~] = two_layer_tx_pipeline(testImage, currentParams);
                
                % Channel
                channelParams = struct();
                channelParams.channelType = 'AWGN';
                channelParams.SNR_dB = snrProfile(frame);
                [rxSignal, ~] = channel_model(txSignal, channelParams);
                
                % Receive
                [rxImage, feedback, ~] = two_layer_rx_pipeline(rxSignal, currentParams);
                
                % Calculate quality
                ssim_val = calculate_ssim(testImage, rxImage);
                psnr_val = calculate_psnr(testImage, rxImage);
                
                % Extract EVM (with error checking)
                base_evm = 15.0;  % Default values if not available
                enh_evm = 20.0;
                urgent_flag = 0;
                quality_met = 0;
                
                if isstruct(feedback)
                    if isfield(feedback, 'bandAnalysis') && isstruct(feedback.bandAnalysis)
                        if isfield(feedback.bandAnalysis, 'critical')
                            if isfield(feedback.bandAnalysis.critical, 'evm')
                                base_evm = feedback.bandAnalysis.critical.evm;
                                if isnan(base_evm)
                                    base_evm = 15.0;
                                end
                            end
                        end
                        if isfield(feedback.bandAnalysis, 'noncritical')
                            if isfield(feedback.bandAnalysis.noncritical, 'evm')
                                enh_evm = feedback.bandAnalysis.noncritical.evm;
                                if isnan(enh_evm)
                                    enh_evm = 20.0;
                                end
                            end
                        end
                    end
                    if isfield(feedback, 'urgentAdapt')
                        urgent_flag = double(feedback.urgentAdapt);
                    end
                    if isfield(feedback, 'qualityMet')
                        quality_met = double(feedback.qualityMet);
                    end
                end
                
                % Log data row
                dataRow = [
                    totalFrames + frame, ...           % frame_id
                    imageIdx, ...                      % image_id
                    profileIdx, ...                    % profile_id
                    snrProfile(frame), ...             % snr_db
                    currentParams.baseMod, ...         % base_mod
                    currentParams.enhancementMod, ...  % enh_mod
                    ssim_val, ...                      % ssim
                    psnr_val, ...                      % psnr
                    base_evm, ...                      % base_evm
                    enh_evm, ...                       % enh_evm
                    urgent_flag, ...                   % urgent_flag
                    quality_met                        % quality_met
                ];
                
                allData = [allData; dataRow];
                
                % Simple adaptation logic for next frame
                if ssim_val < 0.8
                    % Decrease modulation if quality poor
                    currentParams.enhancementMod = max(4, currentParams.enhancementMod / 2);
                    if ssim_val < 0.7
                        currentParams.baseMod = max(2, currentParams.baseMod / 2);
                    end
                elseif ssim_val > 0.9 && snrProfile(frame) > 20
                    % Increase modulation if quality excellent and SNR good
                    currentParams.enhancementMod = min(256, currentParams.enhancementMod * 2);
                end
                
                fprintf('.');
                
            catch ME
                warning('Error in frame %d: %s', frame, ME.message);
                fprintf('x');
            end
        end
        
        totalFrames = totalFrames + nFrames;
        fprintf(' Done!\n');
    end
    fprintf('\n');
end

%% Save to CSV
fprintf('Saving data to CSV...\n');

header = {'frame_id', 'image_id', 'profile_id', 'snr_db', 'base_mod', ...
          'enh_mod', 'ssim', 'psnr', 'base_evm', 'enh_evm', ...
          'urgent_flag', 'quality_met'};

% Write header
fid = fopen('ai_training_data.csv', 'w');
fprintf(fid, '%s\n', strjoin(header, ','));
fclose(fid);

% Write data
dlmwrite('ai_training_data.csv', allData, '-append', 'precision', '%.6f');

fprintf('  ✓ Saved: ai_training_data.csv\n');
fprintf('  ✓ Total samples: %d\n', size(allData, 1));
fprintf('  ✓ SNR range: %.1f - %.1f dB\n', min(allData(:,4)), max(allData(:,4)));
fprintf('  ✓ SSIM range: %.3f - %.3f\n\n', min(allData(:,7)), max(allData(:,7)));

%% Summary
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║              Data Generation Complete!                        ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('Dataset Summary:\n');
fprintf('  Samples: %d\n', size(allData, 1));
fprintf('  Images: %d\n', nTestImages);
fprintf('  Profiles: %d\n', length(snrProfiles));
fprintf('  SNR range: %.0f - %.0f dB\n', min(allData(:,4)), max(allData(:,4)));
fprintf('  Mean SSIM: %.3f\n', mean(allData(:,7)));
fprintf('  Mean PSNR: %.1f dB\n\n', mean(allData(:,8)));

fprintf('Next Steps:\n');
fprintf('  1. Activate Python: source venv/bin/activate\n');
fprintf('  2. Train AI: python feedback_agent.py\n');
fprintf('  3. Test: python matlab_ai_integration.py\n\n');

%% Helper Functions

function ssim_val = calculate_ssim(ref, test)
    ref = double(ref); 
    test = double(test);
    K1 = 0.01; 
    K2 = 0.03; 
    L = 255;
    C1 = (K1*L)^2; 
    C2 = (K2*L)^2;
    mu1 = mean(ref(:)); 
    mu2 = mean(test(:));
    s1 = std(ref(:)); 
    s2 = std(test(:));
    s12 = mean((ref(:)-mu1).*(test(:)-mu2));
    ssim_val = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1^2+s2^2+C2));
end

function psnr_val = calculate_psnr(ref, test)
    mse = mean((double(ref(:)) - double(test(:))).^2);
    if mse == 0
        psnr_val = 100;
    else
        psnr_val = 10*log10(255^2/mse);
    end
end
