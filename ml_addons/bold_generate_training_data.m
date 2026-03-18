function data = bold_generate_training_data(opts)
% BOLD_GENERATE_TRAINING_DATA  Generate training data for all 3 ML models
% Requires: Image Processing Toolbox, WLAN Toolbox
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 1, opts = struct(); end
    if ~isfield(opts, 'nProfiles'),  opts.nProfiles = 8; end
    if ~isfield(opts, 'verbose'),    opts.verbose = true; end
    if ~isfield(opts, 'savePath'),   opts.savePath = fullfile(fileparts(mfilename('fullpath')), 'bold_training_data.mat'); end

    thisDir = fileparts(mfilename('fullpath'));
    root = fullfile(thisDir, '..');
    addpath(fullfile(root, 'tx', 'src'));
    addpath(fullfile(root, 'rx'));
    addpath(fullfile(root, 'channel'));
    addpath(fullfile(root, 'ml_addons'));

    fprintf('\n======================================================================\n');
    fprintf('  Bold ML Training Data Generator - 3 Model Pipeline\n');
    fprintf('======================================================================\n\n');

    %% Test images
    fprintf('[1/5] Creating test images...\n');
    testImg = imread(fullfile(root, 'tx', 'images', 'Testing examples.jpg'));
    testImg = im2double(testImg);
    if size(testImg,3)==3, testImg = rgb2gray(testImg); end
    testImg = testImg(1:64, 1:64);

    images = cell(1, 4);
    images{1} = testImg;
    [X,Y] = meshgrid(linspace(0,1,64), linspace(0,1,64));
    images{2} = X;                                          % gradient
    images{3} = double(sqrt((X-0.5).^2+(Y-0.5).^2)<0.3);  % circle
    images{4} = double(mod(floor(X*8)+floor(Y*8),2));       % checkerboard
    fprintf('  Created %d test images (64x64)\n', length(images));

    %% SNR profiles (shorter for speed)
    fprintf('[2/5] Generating SNR profiles...\n');
    profiles = generate_snr_profiles(opts.nProfiles);
    fprintf('  Generated %d profiles\n', length(profiles));

    % Fewer policy variations for speed
    policies = generate_policy_variations();
    fprintf('  Generated %d policy variations\n', length(policies));

    %% Storage
    quality_features = [];
    quality_targets  = [];
    snr_sequences    = {};

    %% Estimate total work
    totalWork = 0;
    for p = 1:length(profiles), totalWork = totalWork + length(profiles{p}); end
    totalWork = totalWork * length(images) * length(policies);
    fprintf('  Total blocks to process: ~%d (this may take a few minutes)\n', totalWork * 64);

    %% Main loop
    fprintf('[3/5] Running simulations...\n');
    totalSamples = 0;
    w = warning('off', 'all');  % suppress channel warnings during data gen

    for imgIdx = 1:length(images)
        img = images{imgIdx};
        fprintf('  Image %d/%d ', imgIdx, length(images));

        for profIdx = 1:length(profiles)
            snrProfile = profiles{profIdx};
            snr_sequences{end+1} = snrProfile;

            for polIdx = 1:length(policies)
                policy = policies{polIdx};
                N = 8;
                [dctCoeffs, dctInfo] = DCT_Block(double(img), N);

                % Only sample a few SNR points per profile (speed)
                snrSample = snrProfile(1:min(3, length(snrProfile)));

                for snrIdx = 1:length(snrSample)
                    snr = snrSample(snrIdx);
                    % Use RICIAN — channel_model_v1 only supports this
                    param = struct('channelType','RICIAN','SNR_dB',snr,'seed',snrIdx+imgIdx*100);

                    dctReco = zeros(dctInfo.Hp, dctInfo.Wp);
                    nBlocksW = dctInfo.Wp / N;
                    nBlocksH = dctInfo.Hp / N;

                    for blockId = 0:(nBlocksH*nBlocksW - 1)
                        bRow = floor(double(blockId)/double(nBlocksW));
                        bCol = mod(double(blockId), double(nBlocksW));
                        rS = bRow*N+1; cS = bCol*N+1;
                        block = dctCoeffs(rS:rS+N-1, cS:cS+N-1);
                        try
                            [txWave, txInfo] = two_layer_tx_pipeline(block, policy, 0, blockId, 0);
                            [rxWave, ~] = channel_model_v1(txWave, param);
                            % Pad to handle multipath sample loss
                            rxWave = [rxWave; zeros(10,1)];
                            dctBlockHat = two_layer_rx_v2(rxWave, txInfo);
                            dctReco(rS:rS+N-1, cS:cS+N-1) = dctBlockHat;
                        catch
                            % Skip failed blocks (channel edge case)
                        end
                    end

                    recon = IDCT(dctReco, dctInfo);

                    actual_ssim = ssim(recon, img);
                    actual_psnr = psnr(recon, img);

                    feat = bold_quality_predictor(dctReco, double(policy.K), policy, 'extract_features');

                    quality_features = [quality_features, feat(:)];
                    quality_targets  = [quality_targets, [actual_ssim; actual_psnr]];
                    totalSamples = totalSamples + 1;
                end
            end
            fprintf('.');
        end
        fprintf(' done (%d samples so far)\n', totalSamples);
    end

    warning(w);  % restore warnings

    %% Symbol denoiser data
    fprintf('[4/5] Generating symbol denoiser data...\n');
    symbol_data = generate_symbol_denoiser_data();

    %% Save
    fprintf('[5/5] Saving training data...\n');
    data = struct();
    data.quality_features = quality_features;
    data.quality_targets  = quality_targets;
    data.snr_sequences    = snr_sequences;
    data.symbol_data      = symbol_data;
    data.nQualitySamples  = size(quality_features, 2);
    data.nSNRSequences    = length(snr_sequences);

    save(opts.savePath, '-struct', 'data', '-v7.3');

    fprintf('\n  Data Generation Complete!\n');
    fprintf('  Quality samples: %d\n', data.nQualitySamples);
    fprintf('  SNR sequences:   %d\n', data.nSNRSequences);
    fprintf('  Saved to: %s\n\n', opts.savePath);
end

function profiles = generate_snr_profiles(nProfiles)
    profiles = {};
    profiles{end+1} = [31 29 27 24 21 18 15 12 10 8 10 12 15 18 22 26 30];
    profiles{end+1} = [10 10 10 10 10 10];
    profiles{end+1} = [30 30 30 30 30 30];
    profiles{end+1} = [25 25 8 8 25 25];
    profiles{end+1} = [8 16 24 32 40];
    profiles{end+1} = [40 32 24 16 8];
    for i = 1:(nProfiles - length(profiles))
        len = randi([5, 12]);
        base = 15 + 10*rand();
        profile = base + cumsum(randn(1, len) * 3);
        profile = max(5, min(45, profile));
        profiles{end+1} = profile;
    end
end

function policies = generate_policy_variations()
    policies = {};
    K_vals = [10 20 35];
    mcs_base_vals = [0 2];
    mcs_enh_vals = [4 6];
    for k = K_vals
        for mb = mcs_base_vals
            for me = mcs_enh_vals
                p = PolicyController.defaultPolicy();
                p.K = uint8(k);
                p.mcs_base = uint8(mb);
                p.mcs_enh = uint8(me);
                p.tx_mode = "combined";
                policies{end+1} = p;
            end
        end
    end
end

function sd = generate_symbol_denoiser_data()
    modOrders = [4 16 64];
    snrRange = 5:5:35;
    nSymPerCondition = 300;
    sd = struct();
    for mi = 1:length(modOrders)
        M = modOrders(mi);
        constellation = gen_const(M);
        nConst = length(constellation);
        all_noisy = []; all_clean = []; all_snr = [];
        for snr = snrRange
            idx = randi(nConst, nSymPerCondition, 1);
            clean = constellation(idx);
            snrLin = 10^(snr/10);
            sigPow = mean(abs(clean).^2);
            noisePow = sigPow / snrLin;
            noise = sqrt(noisePow/2) * (randn(nSymPerCondition,1) + 1j*randn(nSymPerCondition,1));
            noisy = clean + noise;
            all_noisy = [all_noisy; noisy];
            all_clean = [all_clean; clean];
            all_snr   = [all_snr; snr * ones(nSymPerCondition, 1)];
        end
        fieldName = sprintf('mod%d', M);
        sd.(fieldName).noisy = all_noisy;
        sd.(fieldName).clean = all_clean;
        sd.(fieldName).snr   = all_snr;
    end
end

function C = gen_const(M)
    switch M
        case 4,  C = [1+1j; 1-1j; -1+1j; -1-1j] / sqrt(2);
        case 16, [I,Q] = meshgrid(-3:2:3, -3:2:3); C = (I(:)+1j*Q(:))/sqrt(10);
        case 64, [I,Q] = meshgrid(-7:2:7, -7:2:7); C = (I(:)+1j*Q(:))/sqrt(42);
    end
end
