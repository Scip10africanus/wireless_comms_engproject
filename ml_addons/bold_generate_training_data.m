function data = bold_generate_training_data(opts)
% BOLD_GENERATE_TRAINING_DATA  Generate training data for all 3 ML models
%
% CHANGES v2 (Bold - Apr 2026):
%   - Fixed broken TX pipeline: replaced deleted two_layer_tx_pipeline
%     with two_layer_tx_DCT + two_layer_rx_testing_dct (split mode)
%   - Added noiseVar from LLTF channel estimation (Jie Du, Apr 2026)
%     as feature 16 — integrates Jie's channel estimation work
%   - Training data now reflects actual TX/RX behaviour (not zeros)
%
% Bold (Unubold) - EEME30003 Major Project

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nProfiles'), opts.nProfiles = 8; end
    if ~isfield(opts,'verbose'),   opts.verbose   = true; end
    if ~isfield(opts,'savePath')
        opts.savePath = fullfile(fileparts(mfilename('fullpath')), 'bold_training_data.mat');
    end

    thisDir = fileparts(mfilename('fullpath'));
    root    = fullfile(thisDir, '..');
    addpath(fullfile(root,'tx','src'));
    addpath(fullfile(root,'rx'));
    addpath(fullfile(root,'channel'));
    addpath(fullfile(root,'ml_addons'));

    fprintf('\n======================================================================\n');
    fprintf('  Bold ML Training Data Generator v2 (fixed TX pipeline)\n');
    fprintf('======================================================================\n\n');

    %% Test images
    fprintf('[1/5] Creating test images...\n');
    testImg = imread(fullfile(root,'tx','images','Testing examples.jpg'));
    testImg = im2double(testImg);
    if size(testImg,3)==3, testImg = rgb2gray(testImg); end
    testImg = testImg(1:64, 1:64);

    images    = cell(1,4);
    images{1} = testImg;
    [X,Y]     = meshgrid(linspace(0,1,64), linspace(0,1,64));
    images{2} = X;
    images{3} = double(sqrt((X-0.5).^2+(Y-0.5).^2)<0.3);
    images{4} = double(mod(floor(X*8)+floor(Y*8), 2));
    fprintf('  Created %d test images (64x64)\n', length(images));

    %% SNR profiles
    fprintf('[2/5] Generating SNR profiles...\n');
    profiles = generate_snr_profiles(opts.nProfiles);
    policies = generate_policy_variations();
    fprintf('  Profiles: %d  |  Policy variations: %d\n', length(profiles), length(policies));

    %% Storage
    quality_features = [];
    quality_targets  = [];
    snr_sequences    = {};

    %% Main loop
    fprintf('[3/5] Running simulations...\n');
    totalSamples = 0;
    w = warning('off','all');

    N = 8;

    for imgIdx = 1:length(images)
        img = images{imgIdx};
        fprintf('  Image %d/%d ', imgIdx, length(images));

        [dctCoeffs, dctInfo] = DCT_Block(double(img), N);
        nBlocksW = dctInfo.Wp / N;
        nBlocksH = dctInfo.Hp / N;
        nBlocks  = nBlocksH * nBlocksW;

        % Pack all blocks into 3D array once per image (matches bold_closed_loop_ml.m)
        blocks3D = zeros(N, N, nBlocks);
        for bi = 0:(nBlocks-1)
            bRow = floor(bi/nBlocksW); bCol = mod(bi, nBlocksW);
            rS = bRow*N+1; cS = bCol*N+1;
            blocks3D(:,:,bi+1) = dctCoeffs(rS:rS+N-1, cS:cS+N-1);
        end

        for profIdx = 1:length(profiles)
            snrProfile = profiles{profIdx};
            snr_sequences{end+1} = snrProfile;

            for polIdx = 1:length(policies)
                policy = policies{polIdx};

                snrSample = snrProfile(1:min(3, length(snrProfile)));

                for snrIdx = 1:length(snrSample)
                    snr = snrSample(snrIdx);
                    param = struct('channelType','RICIAN','SNR_dB',snr,...
                                   'seed', snrIdx + imgIdx*100 + polIdx*10);

                    dctReco   = zeros(dctInfo.Hp, dctInfo.Wp);
                    noiseVar_sample = 10^(-snr/10);  % prior: SNR-based estimate

                    try
                        %% Base layer TX → channel → RX  (split mode, layer_type=1)
                        [txWaveBase, txInfoBase] = two_layer_tx_DCT(blocks3D, policy, 0, 0, 1);
                        [rxWaveBase, ~]          = channel_model_v1(txWaveBase, param);
                        rxWaveBase = [rxWaveBase; zeros(10,1)];

                        % Capture noiseVar from LLTF — Jie Du's channel estimation
                        % (same estimator used inside demodem.m, line 40)
                        try
                            ind       = wlanFieldIndices(txInfoBase.cfgBase);
                            rxLLTF    = rxWaveBase(ind.LLTF(1):ind.LLTF(2));
                            demodLLTF = wlanLLTFDemodulate(rxLLTF, txInfoBase.cfgBase.ChannelBandwidth);
                            noiseVar_sample = wlanLLTFNoiseEstimate(demodLLTF);
                        catch, end

                        [DCTbase, rxinfoBase] = two_layer_rx_testing_dct(rxWaveBase, txInfoBase);
                        if rxinfoBase.blockOK
                            dctReco = bold_assemble_dct_gen(DCTbase, dctInfo, N, nBlocksH, nBlocksW);
                        end

                        %% Enhancement layer TX → channel → RX  (layer_type=2)
                        [txWaveEnh, txInfoEnh] = two_layer_tx_DCT(blocks3D, policy, 0, 0, 2);
                        [rxWaveEnh, ~]         = channel_model_v1(txWaveEnh, param);
                        rxWaveEnh = [rxWaveEnh; zeros(10,1)];
                        [DCTenh, rxinfoEnh]    = two_layer_rx_testing_dct(rxWaveEnh, txInfoEnh);

                        enhOK = rxinfoEnh.blockOK && rxinfoEnh.sanityOK && ...
                                ~any(isnan(DCTenh(:))) && ~any(isinf(DCTenh(:)));
                        if enhOK
                            dctReco = dctReco + bold_assemble_dct_gen(DCTenh, dctInfo, N, nBlocksH, nBlocksW);
                        end

                    catch
                        % Pipeline failed — use zero dctReco (records a failure sample)
                    end

                    recon = IDCT(dctReco, dctInfo);
                    actual_ssim = ssim(recon, img);
                    actual_psnr = psnr(recon, img);

                    % Extract all 16 features at once — pass noiseVar so
                    % bold_quality_predictor handles feat 16 internally.
                    % Do NOT append noiseVar again afterwards.
                    feat = bold_quality_predictor(dctReco, double(policy.K), policy, ...
                        'extract_features', [], struct(), noiseVar_sample);

                    quality_features = [quality_features, feat(:)];
                    quality_targets  = [quality_targets,  [actual_ssim; actual_psnr]];
                    totalSamples     = totalSamples + 1;
                end
            end
            fprintf('.');
        end
        fprintf(' done (%d samples)\n', totalSamples);
    end

    warning(w);

    %% Symbol denoiser data (unchanged)
    fprintf('[4/5] Generating symbol denoiser data...\n');
    symbol_data = generate_symbol_denoiser_data();

    %% Save
    fprintf('[5/5] Saving...\n');
    data = struct();
    data.quality_features = quality_features;   % [16 x N]
    data.quality_targets  = quality_targets;
    data.snr_sequences    = snr_sequences;
    data.symbol_data      = symbol_data;
    data.nQualitySamples  = size(quality_features, 2);
    data.nSNRSequences    = length(snr_sequences);
    data.nFeatures        = size(quality_features, 1);  % 16
    save(opts.savePath, '-struct', 'data', '-v7.3');

    fprintf('\n  Done! Samples: %d  |  Features: %d  |  SNR seqs: %d\n\n', ...
        data.nQualitySamples, data.nFeatures, data.nSNRSequences);
end

%% -----------------------------------------------------------------------
function dctMat = bold_assemble_dct_gen(DCTcoe, dctInfo, N, nBlocksH, nBlocksW)
    dctMat = zeros(dctInfo.Hp, dctInfo.Wp);
    for bi = 0:(nBlocksH*nBlocksW-1)
        bRow = floor(bi/nBlocksW); bCol = mod(bi, nBlocksW);
        rS = bRow*N+1; cS = bCol*N+1;
        dctMat(rS:rS+N-1, cS:cS+N-1) = DCTcoe(:,:,bi+1);
    end
end

%% -----------------------------------------------------------------------
function profiles = generate_snr_profiles(nProfiles)
    profiles = {};
    profiles{end+1} = [31 29 27 24 21 18 15 12 10 8 10 12 15 18 22 26 30];
    profiles{end+1} = [10 10 10 10 10 10];
    profiles{end+1} = [30 30 30 30 30 30];
    profiles{end+1} = [25 25 8 8 25 25];
    profiles{end+1} = [8 16 24 32 40];
    profiles{end+1} = [40 32 24 16 8];
    % Forced low-SNR profiles for training data balance (added v1)
    profiles{end+1} = [5 5 5 5 5 5];
    profiles{end+1} = [6 6 7 7 6 6];
    profiles{end+1} = [3 4 5 4 3 4];
    profiles{end+1} = [5 8 12 8 5 3];
    for i = 1:(nProfiles - length(profiles))
        len     = randi([5,12]);
        base    = 15 + 10*rand();
        profile = base + cumsum(randn(1,len)*3);
        profile = max(5, min(45, profile));
        profiles{end+1} = profile;
    end
end

function policies = generate_policy_variations()
    policies = {};
    K_vals       = [10 20 35];
    mcs_base_vals = [0 1];
    mcs_enh_vals  = [3 5];
    for k = K_vals
        for mb = mcs_base_vals
            for me = mcs_enh_vals
                p          = PolicyController.defaultPolicy();
                p.K        = uint8(k);
                p.mcs_base = uint8(mb);
                p.mcs_enh  = uint8(me);
                p.tx_mode  = "split";  % must be split for layer_type=1/2 calls
                policies{end+1} = p;
            end
        end
    end
end

function sd = generate_symbol_denoiser_data()
    modOrders = [4 16 64];
    snrRange  = 5:5:35;
    nSym      = 300;
    sd        = struct();
    for mi = 1:length(modOrders)
        M    = modOrders(mi);
        C    = gen_const(M);
        nC   = length(C);
        all_noisy = []; all_clean = []; all_snr = [];
        for snr = snrRange
            idx   = randi(nC, nSym, 1);
            clean = C(idx);
            snrLin   = 10^(snr/10);
            sigPow   = mean(abs(clean).^2);
            noisePow = sigPow / snrLin;
            noise    = sqrt(noisePow/2) * (randn(nSym,1) + 1j*randn(nSym,1));
            all_noisy = [all_noisy; clean+noise];
            all_clean = [all_clean; clean];
            all_snr   = [all_snr;   snr*ones(nSym,1)];
        end
        fn = sprintf('mod%d', M);
        sd.(fn).noisy = all_noisy;
        sd.(fn).clean = all_clean;
        sd.(fn).snr   = all_snr;
    end
end

function C = gen_const(M)
    switch M
        case 4,  C = [1+1j;1-1j;-1+1j;-1-1j]/sqrt(2);
        case 16, [I,Q]=meshgrid(-3:2:3,-3:2:3); C=(I(:)+1j*Q(:))/sqrt(10);
        case 64, [I,Q]=meshgrid(-7:2:7,-7:2:7); C=(I(:)+1j*Q(:))/sqrt(42);
    end
end
