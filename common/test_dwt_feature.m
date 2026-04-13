function test_dwt_feature()
    fprintf('\n================================================================\n');
    fprintf('TEST: DWT feature extractor + offline generator\n');
    fprintf('================================================================\n');

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    rootDir = fileparts(thisDir);
    addpath(fullfile(rootDir,'common'));
    addpath(fullfile(rootDir,'tx','src'));
    addpath(fullfile(rootDir,'rx'));
    addpath(fullfile(rootDir,'channel'));

    validate_extractor(rootDir);
    validate_generator_output();
    validate_semantic_stability();
    validate_path_consistency(rootDir);

    fprintf('PASS: DWT extractor + generator validations passed\n');
    fprintf('================================================================\n');
end

function validate_extractor(rootDir)
    fprintf('\n[1/4] Extractor-level validation\n');

    N = 64;
    img_const = uint8(128 * ones(N,N));
    img_grad = uint8(repmat(uint8(linspace(0,255,N)), N, 1));
    [X,Y] = meshgrid(0:(N-1), 0:(N-1));
    img_chk = uint8(255 * mod(X + Y, 2));

    policy = struct();
    policy.Qb_idx = uint8(0);
    policy.Qe_idx = uint8(0);
    policy.tx_mode = 'split';
    policy.mux_gap_us = 0;
    policy.mcs_base = uint8(0);
    policy.mcs_enh  = uint8(0);

    [~, txInfo] = two_layer_tx_wavelet(double(img_const), policy, uint8(0));
    base_share_expected = double(txInfo.numBaseBlocks * txInfo.blk_size^2 * 2) / ...
        double(txInfo.numBaseBlocks * txInfo.blk_size^2 * 2 + txInfo.numEnhBlocks * txInfo.blk_size^2 * 1);

    feats = struct();
    feats.const = extract_one(img_const);
    feats.grad  = extract_one(img_grad);
    feats.chk   = extract_one(img_chk);

    fprintf('  constant:   E_LL_ratio=%.6f  E_HH_ratio=%.6f  base_share=%.6f\n', feats.const.E_LL_ratio, feats.const.E_HH_ratio, feats.const.base_share);
    fprintf('  gradient:   E_LL_ratio=%.6f  E_HH_ratio=%.6f  base_share=%.6f\n', feats.grad.E_LL_ratio, feats.grad.E_HH_ratio, feats.grad.base_share);
    fprintf('  checkerbd:  E_LL_ratio=%.6f  E_HH_ratio=%.6f  base_share=%.6f\n', feats.chk.E_LL_ratio, feats.chk.E_HH_ratio, feats.chk.base_share);

    check_ratio(feats.const.E_LL_ratio, 'const.E_LL_ratio');
    check_ratio(feats.const.E_HH_ratio, 'const.E_HH_ratio');
    check_ratio(feats.grad.E_LL_ratio,  'grad.E_LL_ratio');
    check_ratio(feats.grad.E_HH_ratio,  'grad.E_HH_ratio');
    check_ratio(feats.chk.E_LL_ratio,   'chk.E_LL_ratio');
    check_ratio(feats.chk.E_HH_ratio,   'chk.E_HH_ratio');

    assert(isfinite(feats.const.base_share) && feats.const.base_share >= 0 && feats.const.base_share <= 1, 'base_share out of [0,1]');
    assert(abs(feats.const.base_share - base_share_expected) < 1e-12, 'base_share mismatch vs modeled txInfo');

    assert(feats.const.E_LL_ratio > 0.98, 'constant image E_LL_ratio not close to 1');
    assert(feats.const.E_HH_ratio < 1e-6, 'constant image E_HH_ratio not close to 0');

    assert(feats.chk.E_HH_ratio > feats.grad.E_HH_ratio, 'checkerboard E_HH_ratio not higher than gradient');
    assert(feats.chk.E_HH_ratio > feats.const.E_HH_ratio, 'checkerboard E_HH_ratio not higher than constant');

    function feat = extract_one(img)
        info = struct();
        [res, info] = DWT(double(img), info, 1);
        feat = dwt_extract_features(res, txInfo, struct('featureVersion',uint32(1),'wavelet',string(info.wavelet),'level',double(info.level)));
        assert(feat.feature_version == 1);
        assert(feat.wavelet == "haar");
        assert(feat.level == 1);
    end
end

function validate_generator_output()
    fprintf('\n[2/4] Generator output validation\n');

    opts = struct();
    opts.verbose = false;
    opts.nProfiles = 1;
    opts.snrSamplesPerProfile = 1;
    opts.level = 1;
    opts.wavelet = "haar";
    opts.featureVersion = uint32(1);

    d = dwt_generate_training_data(opts);
    T = d.samples;

    nImages = 4;
    nProfiles = 1;
    nPolicies = 12;
    nSnr = 1;
    assert(height(T) == nImages*nProfiles*nPolicies*nSnr, 'row count mismatch');

    required = [ ...
        "feature_version","wavelet","level", ...
        "E_LL_ratio","E_HH_ratio","base_share_model", ...
        "noiseVar_base_mean","noiseVar_enh_mean", ...
        "log10NoiseVar_base_mean","log10NoiseVar_enh_mean", ...
        "basePER","enhPER","ssim","psnr","mse" ...
    ];
    for i = 1:numel(required)
        assert(any(strcmp(T.Properties.VariableNames, required(i))), "missing column: %s", required(i));
    end

    assert(all(T.feature_version == double(opts.featureVersion)), 'feature_version inconsistent');
    assert(all(string(T.wavelet) == "haar"), 'wavelet inconsistent');
    assert(all(T.level == 1), 'level inconsistent');

    check_ratio_vec(T.E_LL_ratio, 'E_LL_ratio');
    check_ratio_vec(T.E_HH_ratio, 'E_HH_ratio');
    check_ratio_vec(T.base_share_model, 'base_share_model');
    check_ratio_vec(1 - T.basePER, 'baseOK_ratio');
    check_ratio_vec(1 - T.enhPER, 'enhOK_ratio');

    fprintf('  rows=%d, required columns present, ratios in [0,1]\n', height(T));
end

function validate_semantic_stability()
    fprintf('\n[3/4] Semantic stability validation\n');

    opts = struct();
    opts.verbose = false;
    opts.nProfiles = 1;
    opts.snrSamplesPerProfile = 2;
    opts.level = 1;
    opts.wavelet = "haar";
    opts.featureVersion = uint32(1);

    d = dwt_generate_training_data(opts);
    T = d.samples;

    imgIds = unique(T.image_id);
    for i = 1:numel(imgIds)
        id = imgIds(i);
        mask = (T.image_id == id);
        r1 = T.E_LL_ratio(mask);
        r2 = T.E_HH_ratio(mask);
        assert(max(r1) - min(r1) < 1e-10, 'E_LL_ratio not stable for image_id=%d', id);
        assert(max(r2) - min(r2) < 1e-10, 'E_HH_ratio not stable for image_id=%d', id);
    end

    snrs = unique(T.snr_dB);
    assert(numel(snrs) == 2, 'expected exactly 2 SNR points');
    hi = max(snrs);
    lo = min(snrs);

    nvB_hi = median(T.noiseVar_base_mean(T.snr_dB == hi));
    nvB_lo = median(T.noiseVar_base_mean(T.snr_dB == lo));
    nvE_hi = median(T.noiseVar_enh_mean(T.snr_dB == hi));
    nvE_lo = median(T.noiseVar_enh_mean(T.snr_dB == lo));
    assert(nvB_hi <= nvB_lo, 'noiseVar_base_mean median not lower at higher SNR');
    assert(nvE_hi <= nvE_lo, 'noiseVar_enh_mean median not lower at higher SNR');

    perB_hi = median(T.basePER(T.snr_dB == hi));
    perB_lo = median(T.basePER(T.snr_dB == lo));
    assert(perB_hi <= perB_lo + 1e-12, 'basePER median worse at higher SNR');

    idx = isfinite(T.noiseVar_base_mean) & (T.noiseVar_base_mean > 0);
    assert(all(abs(T.log10NoiseVar_base_mean(idx) - log10(T.noiseVar_base_mean(idx))) < 1e-10), 'log10NoiseVar_base_mean mismatch');

    idx = isfinite(T.noiseVar_enh_mean) & (T.noiseVar_enh_mean > 0);
    assert(all(abs(T.log10NoiseVar_enh_mean(idx) - log10(T.noiseVar_enh_mean(idx))) < 1e-10), 'log10NoiseVar_enh_mean mismatch');

    fprintf('  stability OK; noiseVar trends OK; log10 consistency OK\n');
end

function validate_path_consistency(rootDir)
    fprintf('\n[4/4] Path-consistency validation\n');

    addpath(fullfile(rootDir,'common'));
    addpath(fullfile(rootDir,'tx','src'));
    addpath(fullfile(rootDir,'rx'));
    addpath(fullfile(rootDir,'channel'));

    opts = struct();
    opts.verbose = false;
    opts.nProfiles = 1;
    opts.snrSamplesPerProfile = 1;
    opts.level = 1;
    opts.wavelet = "haar";
    opts.featureVersion = uint32(1);
    opts.ricianK = 3;

    d = dwt_generate_training_data(opts);
    T = d.samples;

    row = T(T.image_id==1 & T.profile_id==1 & T.policy_id==1 & T.snr_dB==31, :);
    assert(height(row) == 1, 'expected exactly one row for fixed case');

    img = load_image1(rootDir);
    policy = struct('Qb_idx',uint8(0),'Qe_idx',uint8(0),'tx_mode','split','mux_gap_us',0,'mcs_base',uint8(row.mcs_base),'mcs_enh',uint8(row.mcs_enh));

    param = struct('channelType','RICIAN','SNR_dB',31,'ricianK',double(opts.ricianK),'seed',1111);
    [txWaves, txInfo] = two_layer_tx_wavelet(double(img), policy, uint8(0));
    rxWaves = struct('base', [], 'enh', {{}});
    for i = 1:txInfo.numBaseBlocks
        [rx, ~] = channel_model_v1(txWaves.base{i}, param);
        rxWaves.base{i} = rx;
    end
    for i = 1:txInfo.numEnhBlocks
        [rx, ~] = channel_model_v1(txWaves.enh{i}, param);
        rxWaves.enh{i} = rx;
    end

    [res_hat, rxinfo_ref] = two_layer_rx_testing_dwt(rxWaves, txInfo, 1, 1);
    recon = IDWT(res_hat, txInfo.level);
    ssim_ref = ssim(recon, double(img));
    psnr_ref = psnr(recon, double(img), 255);
    mse_ref  = mean((double(img(:)) - double(recon(:))).^2);

    assert(abs(ssim_ref - row.ssim) < 1e-9, 'SSIM mismatch vs reference path');
    assert(abs(psnr_ref - row.psnr) < 1e-6, 'PSNR mismatch vs reference path');
    assert(abs(mse_ref - row.mse) < 1e-6,  'MSE mismatch vs reference path');
    assert(abs(rxinfo_ref.basePER - row.basePER) < 1e-12, 'basePER mismatch vs reference path');
    assert(abs(rxinfo_ref.enhPER - row.enhPER) < 1e-12, 'enhPER mismatch vs reference path');

    fprintf('  fixed-case metrics match reference path (SSIM/PSNR/MSE/PER)\n');
end

function img = load_image1(rootDir)
    p = fullfile(rootDir,'tx','images','Testing examples.jpg');
    raw = imread(p);
    if size(raw,3) == 3, raw = rgb2gray(raw); end
    img = raw(1:64,1:64);
end

function check_ratio(v, name)
    assert(isfinite(v), '%s not finite', name);
    assert(v >= 0 && v <= 1, '%s not in [0,1]', name);
end

function check_ratio_vec(v, name)
    assert(all(isfinite(v)), '%s contains non-finite', name);
    assert(all(v >= 0 & v <= 1), '%s not in [0,1]', name);
end

