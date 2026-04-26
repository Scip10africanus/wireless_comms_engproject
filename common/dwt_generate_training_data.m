function data = dwt_generate_training_data(opts)
    if nargin < 1 || ~isstruct(opts), opts = struct(); end
    if ~isfield(opts,'verbose'), opts.verbose = true; end
    if ~isfield(opts,'nProfiles'), opts.nProfiles = 6; end
    if ~isfield(opts,'snrSamplesPerProfile'), opts.snrSamplesPerProfile = 3; end
    if ~isfield(opts,'ricianK'), opts.ricianK = 3; end
    if ~isfield(opts,'channelType'), opts.channelType = 'RICIAN'; end
    if ~isfield(opts,'savePath'), opts.savePath = ''; end
    if ~isfield(opts,'rx_Qb'), opts.rx_Qb = 1; end
    if ~isfield(opts,'rx_Qe'), opts.rx_Qe = 1; end
    if ~isfield(opts,'level'), opts.level = 1; end
    if ~isfield(opts,'wavelet'), opts.wavelet = "haar"; end
    if ~isfield(opts,'featureVersion'), opts.featureVersion = uint32(1); end

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    rootDir = fileparts(thisDir);

    addpath(fullfile(rootDir,'common'));
    addpath(fullfile(rootDir,'tx','src'));
    addpath(fullfile(rootDir,'rx'));
    addpath(fullfile(rootDir,'channel'));

    images = make_default_images(rootDir);
    profiles = make_snr_profiles(opts.nProfiles);
    policies = make_policy_variations();

    rows = {};
    if opts.verbose
        fprintf('\n================================================================\n');
        fprintf('DWT TRAINING DATA GENERATOR | images=%d | profiles=%d | policies=%d\n', ...
            numel(images), numel(profiles), numel(policies));
        fprintf('================================================================\n');
    end

    sampleId = 0;
    for imgIdx = 1:numel(images)
        img = images{imgIdx};
        img = double(img);

        for profIdx = 1:numel(profiles)
            snrProfile = profiles{profIdx};
            snrProfile = snrProfile(1:min(numel(snrProfile), max(1, opts.snrSamplesPerProfile)));

            for polIdx = 1:numel(policies)
                policy = policies{polIdx};

                for snrIdx = 1:numel(snrProfile)
                    snr = double(snrProfile(snrIdx));

                    param = struct();
                    param.channelType = opts.channelType;
                    param.SNR_dB = snr;
                    param.ricianK = double(opts.ricianK);
                    param.seed = 1000 + imgIdx*100 + profIdx*10 + snrIdx;

                    info = struct();
                    [res, info] = DWT(img, info, double(opts.level));

                    [txWaves, txInfo] = two_layer_tx_wavelet(img, policy, uint8(0));
                    txInfo.DWTInfo = info;

                    rxWaves = apply_channel_to_waves(txWaves, param);

                    [res_hat, rxinfo, noiseStats] = decode_wavelet_with_noisevar(rxWaves, txInfo, opts.rx_Qb, opts.rx_Qe);
                    recon = IDWT(res_hat, txInfo.level);

                    ssimVal = ssim(recon, img);
                    psnrVal = psnr(recon, img, 255);
                    mseVal  = mean((img(:) - recon(:)).^2);

                    featOpts = struct('featureVersion', uint32(opts.featureVersion), ...
                        'wavelet', string(get_field_or(info,'wavelet',opts.wavelet)), ...
                        'level', double(get_field_or(info,'level',txInfo.level)));
                    feat = dwt_extract_features(res, txInfo, featOpts);

                    sampleId = sampleId + 1;
                    rows{end+1,1} = make_row(sampleId, imgIdx, profIdx, polIdx, snr, param, policy, feat, txInfo, rxinfo, noiseStats, ssimVal, psnrVal, mseVal); %#ok<AGROW>
                end
            end
        end
    end

    T = vertcat(rows{:});
    data = struct();
    data.samples = T;
    data.opts = opts;

    if isfield(opts,'savePath') && strlength(string(opts.savePath)) > 0
        save(string(opts.savePath), '-struct', 'data', '-v7.3');
        if opts.verbose
            fprintf('Saved: %s\n', string(opts.savePath));
        end
    end
end

function v = get_field_or(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = defaultValue;
    end
end

function imgs = make_default_images(rootDir)
    imgs = {};
    p = fullfile(rootDir,'tx','images','Testing examples.jpg');
    raw = imread(p);
    if size(raw,3) == 3, raw = rgb2gray(raw); end
    raw = raw(1:64,1:64);
    imgs{end+1} = raw;

    [X,Y] = meshgrid(linspace(0,255,64), linspace(0,255,64));
    imgs{end+1} = uint8(X);
    imgs{end+1} = uint8(255 * (sqrt((X-128).^2 + (Y-128).^2) < 80));
    imgs{end+1} = uint8(255 * mod(floor(X/32) + floor(Y/32), 2));
end

function profiles = make_snr_profiles(nProfiles)
    base = {};
    base{end+1} = [31 29 27 24 21 18 15 12 10 8 10 12 15 18 22 26 30];
    base{end+1} = [10 10 10 10 10 10];
    base{end+1} = [30 30 30 30 30 30];
    base{end+1} = [25 25 8 8 25 25];
    base{end+1} = [8 16 24 32 40];
    base{end+1} = [40 32 24 16 8];

    profiles = {};
    nBase = min(numel(base), max(0, nProfiles));
    for i = 1:nBase
        profiles{end+1} = base{i};
    end

    for i = (nBase+1):nProfiles
        len = randi([5, 12]);
        b = 15 + 10*rand();
        profile = b + cumsum(randn(1, len) * 3);
        profile = max(5, min(45, profile));
        profiles{end+1} = profile;
    end
end

function policies = make_policy_variations()
    policies = {};
    p = struct();
    p.Qb_idx = uint8(0);
    p.Qe_idx = uint8(0);
    p.tx_mode = 'split';
    p.mux_gap_us = 0;

    mcs_base_vals = [0 1 2];
    mcs_enh_vals = [0 2 4 5];
    for mb = mcs_base_vals
        for me = mcs_enh_vals
            pp = p;
            pp.mcs_base = uint8(mb);
            pp.mcs_enh  = uint8(me);
            policies{end+1} = pp; %#ok<AGROW>
        end
    end
end

function rxWaves = apply_channel_to_waves(txWaves, param)
    rxWaves = struct('base', [], 'enh', {{}});
    if isfield(txWaves,'base') && ~isempty(txWaves.base)
        rxWaves.base = cell(size(txWaves.base));
        for i = 1:numel(txWaves.base)
            [rx, ~] = channel_model_v1(txWaves.base{i}, param);
            rxWaves.base{i} = rx;
        end
    else
        rxWaves.base = {};
    end
    if isfield(txWaves,'enh') && ~isempty(txWaves.enh)
        rxWaves.enh = cell(size(txWaves.enh));
        for i = 1:numel(txWaves.enh)
            [rx, ~] = channel_model_v1(txWaves.enh{i}, param);
            rxWaves.enh{i} = rx;
        end
    else
        rxWaves.enh = {};
    end
end

function [res_hat, rxinfo, noiseStats] = decode_wavelet_with_noisevar(rxWaves, txInfo, rx_Qb, rx_Qe)
    rxinfo = struct();
    rxinfo.txInfo = txInfo;
    rxinfo.numBasePkts = txInfo.numBaseBlocks;
    rxinfo.numEnhPkts  = txInfo.numEnhBlocks;
    rxinfo.numBaseOk = 0;
    rxinfo.numEnhOk  = 0;
    rxinfo.baseFailReasons = {};
    rxinfo.enhFailReasons  = {};

    noiseBase = [];
    noiseEnh  = [];

    baseBlocks = cell(txInfo.numBaseBlocks, 1);
    for i = 1:txInfo.numBaseBlocks
        [rxBaseBytes, ok, reason, nVar] = demodem(rxWaves.base{i}, txInfo.cfgBase);
        if ~isnan(nVar), noiseBase(end+1) = double(nVar); end %#ok<AGROW>
        if ~ok
            rxinfo.baseFailReasons{end+1} = sprintf('base block %d: %s', i, reason); %#ok<AGROW>
            baseBlocks{i} = zeros(txInfo.blk_size, txInfo.blk_size);
            continue;
        end
        try
            rxBaseInt16 = typecast(uint8(rxBaseBytes(:)), 'int16');
            [blk_hat, ~] = dequant(rxBaseInt16, [], rx_Qb, rx_Qe);
            baseBlocks{i} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
            rxinfo.numBaseOk = rxinfo.numBaseOk + 1;
        catch ME
            rxinfo.baseFailReasons{end+1} = sprintf('base block %d: base_unpack_fail:%s', i, ME.identifier); %#ok<AGROW>
            baseBlocks{i} = zeros(txInfo.blk_size, txInfo.blk_size);
        end
    end
    base_hat_coeffs = untile_coeffs(baseBlocks, txInfo.basePadInfo, txInfo.blk_size);

    res_hat = struct();
    res_hat.finalLL = base_hat_coeffs;
    res_hat.subbands = struct('LH', {}, 'HL', {}, 'HH', {}, 'inputSize', {});
    for k = 1:txInfo.level
        res_hat.subbands(k).LH = [];
        res_hat.subbands(k).HL = [];
        res_hat.subbands(k).HH = [];
        res_hat.subbands(k).inputSize = [];
    end

    enhIdx = 1;
    for i = 1:numel(txInfo.enhSubbandMeta)
        meta = txInfo.enhSubbandMeta(i);
        sbBlocks = cell(meta.numBlocks, 1);
        for b = 1:meta.numBlocks
            [rxEnhBytes, ok, reason, nVar] = demodem(rxWaves.enh{enhIdx}, txInfo.cfgEnh);
            if ~isnan(nVar), noiseEnh(end+1) = double(nVar); end %#ok<AGROW>
            if ~ok
                rxinfo.enhFailReasons{end+1} = sprintf('enh block %d (%s%d #%d): %s', enhIdx, meta.type, meta.level, b, reason); %#ok<AGROW>
                sbBlocks{b} = zeros(txInfo.blk_size, txInfo.blk_size);
                enhIdx = enhIdx + 1;
                continue;
            end
            try
                rxEnhInt8 = typecast(uint8(rxEnhBytes(:)), 'int8');
                [~, blk_hat] = dequant([], rxEnhInt8, rx_Qb, rx_Qe);
                sbBlocks{b} = reshape(blk_hat, txInfo.blk_size, txInfo.blk_size);
                rxinfo.numEnhOk = rxinfo.numEnhOk + 1;
            catch ME
                rxinfo.enhFailReasons{end+1} = sprintf('enh block %d (%s%d #%d): enh_unpack_fail:%s', enhIdx, meta.type, meta.level, b, ME.identifier); %#ok<AGROW>
                sbBlocks{b} = zeros(txInfo.blk_size, txInfo.blk_size);
            end
            enhIdx = enhIdx + 1;
        end

        sbMat = untile_coeffs(sbBlocks, meta.origSize, txInfo.blk_size);
        res_hat.subbands(meta.level).(meta.type) = sbMat;
        if isempty(res_hat.subbands(meta.level).inputSize)
            res_hat.subbands(meta.level).inputSize = meta.origSize * 2;
        end
    end

    if (enhIdx - 1) ~= double(txInfo.numEnhBlocks)
        rxinfo.enhFailReasons{end+1} = sprintf('enh_block_count_mismatch: decoded=%d expected=%d', enhIdx-1, double(txInfo.numEnhBlocks)); %#ok<AGROW>
    end

    rxinfo.basePER = 1 - rxinfo.numBaseOk / max(rxinfo.numBasePkts, 1);
    rxinfo.enhPER  = 1 - rxinfo.numEnhOk  / max(rxinfo.numEnhPkts, 1);

    noiseStats = struct();
    noiseStats.noiseVar_base_mean = mean_or_nan(noiseBase);
    noiseStats.noiseVar_base_median = median_or_nan(noiseBase);
    noiseStats.noiseVar_enh_mean = mean_or_nan(noiseEnh);
    noiseStats.noiseVar_enh_median = median_or_nan(noiseEnh);
    noiseStats.log10NoiseVar_base_mean = log10_or_nan(noiseStats.noiseVar_base_mean);
    noiseStats.log10NoiseVar_enh_mean = log10_or_nan(noiseStats.noiseVar_enh_mean);
end

function v = mean_or_nan(x)
    if isempty(x), v = NaN; else, v = mean(x); end
end

function v = median_or_nan(x)
    if isempty(x), v = NaN; else, v = median(x); end
end

function v = log10_or_nan(x)
    if ~isfinite(x) || x <= 0
        v = NaN;
    else
        v = log10(x);
    end
end

function T = make_row(sampleId, imgIdx, profIdx, polIdx, snr, param, policy, feat, txInfo, rxinfo, noiseStats, ssimVal, psnrVal, mseVal)
    base_payload_bytes = double(txInfo.numBaseBlocks) * double(txInfo.blk_size)^2 * 2;
    enh_payload_bytes  = double(txInfo.numEnhBlocks)  * double(txInfo.blk_size)^2 * 1;

    T = table();
    T.sample_id = sampleId;
    T.image_id = imgIdx;
    T.profile_id = profIdx;
    T.policy_id = polIdx;

    T.feature_version = double(feat.feature_version);
    T.wavelet = string(feat.wavelet);
    T.level = double(feat.level);

    T.snr_dB = snr;
    T.ricianK_dB = double(param.ricianK);
    T.seed = double(param.seed);

    T.mcs_base = double(policy.mcs_base);
    T.mcs_enh  = double(policy.mcs_enh);
    T.Qb_idx = double(policy.Qb_idx);
    T.Qe_idx = double(policy.Qe_idx);
    T.tx_mode = string(policy.tx_mode);
    T.mux_gap_us = double(policy.mux_gap_us);

    T.dwt_blk_size = double(txInfo.blk_size);
    T.numBaseBlocks = double(txInfo.numBaseBlocks);
    T.numEnhBlocks  = double(txInfo.numEnhBlocks);
    T.base_payload_bytes_model = base_payload_bytes;
    T.enh_payload_bytes_model  = enh_payload_bytes;

    T.E_LL_ratio = feat.E_LL_ratio;
    T.E_HH_ratio = feat.E_HH_ratio;
    T.base_share_model = feat.base_share;

    T.basePER = double(rxinfo.basePER);
    T.enhPER  = double(rxinfo.enhPER);

    T.noiseVar_base_mean = double(noiseStats.noiseVar_base_mean);
    T.noiseVar_base_median = double(noiseStats.noiseVar_base_median);
    T.noiseVar_enh_mean = double(noiseStats.noiseVar_enh_mean);
    T.noiseVar_enh_median = double(noiseStats.noiseVar_enh_median);
    T.log10NoiseVar_base_mean = double(noiseStats.log10NoiseVar_base_mean);
    T.log10NoiseVar_enh_mean = double(noiseStats.log10NoiseVar_enh_mean);

    T.ssim = double(ssimVal);
    T.psnr = double(psnrVal);
    T.mse  = double(mseVal);
end
