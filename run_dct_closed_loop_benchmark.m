function results = run_dct_closed_loop_benchmark(opts)
    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),         opts.nFrames = 20; end
    if ~isfield(opts,'snrProfile'),      opts.snrProfile = 25 * ones(1, opts.nFrames); end
    if ~isfield(opts,'ricianK'),         opts.ricianK = 3; end
    if ~isfield(opts,'policyCsvPath'),   opts.policyCsvPath = fullfile(tempdir(), 'policy_dct_benchmark.csv'); end
    if ~isfield(opts,'verbose'),         opts.verbose = true; end
    if ~isfield(opts,'doPlot'),          opts.doPlot = true; end
    if ~isfield(opts,'concealmentMode'), opts.concealmentMode = 'zero'; end
    if ~isfield(opts,'targetSsim'),      opts.targetSsim = 0.9; end
    if ~isfield(opts,'targetPsnr'),      opts.targetPsnr = 30; end
    if ~isfield(opts,'useCrc16'),        opts.useCrc16 = true; end

    opts.snrProfile = opts.snrProfile(:).';
    if numel(opts.snrProfile) < opts.nFrames
        opts.snrProfile(end+1:opts.nFrames) = opts.snrProfile(end);
    end
    opts.snrProfile = opts.snrProfile(1:opts.nFrames);

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end

    addpath(fullfile(thisDir,'common'));
    addpath(fullfile(thisDir,'tx','src'));
    addpath(fullfile(thisDir,'rx'));
    addpath(fullfile(thisDir,'channel'));

    pTx = string(which('two_layer_tx_DCT'));
    pRx = string(which('two_layer_rx_testing_dct'));
    pCh = string(which('channel_model_v1'));
    pFb = string(which('rx_feedback'));
    pPc = string(which('PolicyController'));
    assert(strlength(pTx) > 0, 'two_layer_tx_DCT not found on path');
    assert(strlength(pRx) > 0, 'two_layer_rx_testing_dct not found on path');
    assert(strlength(pCh) > 0, 'channel_model_v1 not found on path');
    assert(strlength(pFb) > 0, 'rx_feedback not found on path');
    assert(strlength(pPc) > 0, 'PolicyController not found on path');
    assert(contains(pTx, fullfile(thisDir,'tx','src')), "Wrong two_layer_tx_DCT resolved: %s", pTx);
    assert(contains(pRx, fullfile(thisDir,'rx')), "Wrong two_layer_rx_testing_dct resolved: %s", pRx);
    assert(contains(pCh, fullfile(thisDir,'channel')), "Wrong channel_model_v1 resolved: %s", pCh);

    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end
    pc = PolicyController(opts.policyCsvPath);
    pc.Verbose = logical(opts.verbose);

    testImg = im2double(imread(fullfile(thisDir,'tx','images','Testing examples.jpg')));
    if size(testImg,3) == 3, testImg = rgb2gray(testImg); end
    testImg = testImg(1:64, 1:64);

    N = 8;
    [dctCoeffs, dctInfo] = DCT_Block(double(testImg), N);
    nBlocksW = dctInfo.Wp / N;
    nBlocksH = dctInfo.Hp / N;
    nBlocks = nBlocksW * nBlocksH;

    channelHistory = struct('snr', []);
    chanParam = struct('channelType','RICIAN','SNR_dB',20,'ricianK',double(opts.ricianK));

    results = init_results(opts.nFrames);
    results.nFrames = opts.nFrames;
    results.snrProfile = opts.snrProfile;

    fprintf('\n================================================================\n');
    fprintf('DCT CLOSED-LOOP BENCHMARK | %dx%d | blocks=%d | frames=%d | RICIAN(K=%g)\n', ...
        size(testImg,1), size(testImg,2), nBlocks, opts.nFrames, double(opts.ricianK));
    fprintf('================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %7s | %9s | %10s | %10s | %10s\n', ...
        'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR','blkOK/N','failBase','failEnh','catast');
    fprintf('%s\n', repmat('-',1,112));

    for f = 1:opts.nFrames
        frameId = uint8(mod(f-1, 256));
        snr = double(opts.snrProfile(f));
        channelHistory.snr(end+1) = snr;
        chanParam.SNR_dB = snr;
        chanParam.seed = double(frameId);

        policy = pc.tick(frameId);
        policy.use_crc16 = logical(opts.useCrc16);
        results.policyTrace{f} = policy;

        dctReco = zeros(dctInfo.Hp, dctInfo.Wp);
        frameDiag = init_frame_diag(frameId, policy, nBlocks);

        counts = init_counts();
        for blockId = 0:(nBlocks-1)
            [rStart, cStart] = block_pos(blockId, nBlocksW, N);
            block = dctCoeffs(rStart:rStart+N-1, cStart:cStart+N-1);

            [blkHat, blkRes] = run_single_dct_block_txrx(block, policy, frameId, blockId, chanParam, opts.concealmentMode);
            dctReco(rStart:rStart+N-1, cStart:cStart+N-1) = blkHat;

            counts = merge_counts(counts, blkRes);
            if ~isempty(blkRes.events)
                frameDiag.events = [frameDiag.events, blkRes.events]; %#ok<AGROW>
            end
            if blkRes.concealed
                frameDiag.concealedBlocks(end+1) = double(blockId); %#ok<AGROW>
            end
        end

        recon = IDCT(dctReco, dctInfo);
        ssimVal = ssim(recon, testImg);
        psnrVal = psnr(recon, testImg);
        mseVal  = mean((testImg(:) - recon(:)).^2);

        quality = struct('hasMetrics', true, 'ssim', ssimVal, 'psnr', psnrVal, 'mse', mseVal);

        frameSuccess = (counts.catastrophicBlocks == 0);
        rxRes = struct();
        rxRes.success = frameSuccess;
        rxRes.blkOK = counts.blkOK;
        rxRes.nBlocks = nBlocks;
        rxRes.enhFailRatio = counts.blkFailEnh / max(1, nBlocks);
        rxRes.baseOnlyRatio = counts.baseOnlyBlocks / max(1, nBlocks);

        rxSettings = struct();
        rxSettings.policyCsvPath = opts.policyCsvPath;
        rxSettings.verbose = logical(opts.verbose);
        rxSettings.useGaOptimizer = true;
        rxSettings.useChannelPredictor = false;
        rxSettings.channelHistory = channelHistory;
        rxSettings.targetSsim = double(opts.targetSsim);
        rxSettings.targetPsnr = double(opts.targetPsnr);
        rxSettings.oracleSNR = snr;
        rxSettings.oracleSNRNote = "This benchmark currently uses known simulation SNR rather than receiver-estimated SNR.";

        feedback = rx_feedback(quality, rxRes, policy, rxSettings);

        results.snr(f) = snr;
        results.ssim(f) = ssimVal;
        results.psnr(f) = psnrVal;
        results.mse(f) = mseVal;
        results.K(f) = double(policy.K);
        results.mcs_base(f) = double(policy.mcs_base);
        results.mcs_enh(f) = double(policy.mcs_enh);
        results.success(f) = frameSuccess;
        results.blkOK(f) = counts.blkOK;
        results.nBlocks(f) = nBlocks;
        results.blkFailBase(f) = counts.blkFailBase;
        results.blkFailEnh(f) = counts.blkFailEnh;
        results.blkFailCombined(f) = counts.blkFailCombined;
        results.baseOnlyBlocks(f) = counts.baseOnlyBlocks;
        results.catastrophicBlocks(f) = counts.catastrophicBlocks;

        frameDiag.blkOK = counts.blkOK;
        frameDiag.blkFailBase = counts.blkFailBase;
        frameDiag.blkFailEnh = counts.blkFailEnh;
        frameDiag.blkFailCombined = counts.blkFailCombined;
        frameDiag.baseOnlyBlocks = counts.baseOnlyBlocks;
        frameDiag.catastrophicBlocks = counts.catastrophicBlocks;
        frameDiag.enhFailRatio = rxRes.enhFailRatio;
        frameDiag.baseOnlyRatio = rxRes.baseOnlyRatio;
        frameDiag.success = frameSuccess;
        frameDiag.quality = quality;
        frameDiag.feedback = summarize_feedback(feedback);
        results.frameDiagnostics{f} = frameDiag;

        print_frame_log(f, snr, policy, ssimVal, psnrVal, counts, nBlocks);
    end

    print_summary(results);
    print_enh_diagnostics(results);
    if opts.doPlot
        plot_results(results);
    end
end

function results = init_results(nFrames)
    results = struct();
    results.nFrames = nFrames;
    results.snrProfile = [];
    results.snr = zeros(1,nFrames);
    results.ssim = zeros(1,nFrames);
    results.psnr = zeros(1,nFrames);
    results.mse = zeros(1,nFrames);
    results.K = zeros(1,nFrames);
    results.mcs_base = zeros(1,nFrames);
    results.mcs_enh = zeros(1,nFrames);
    results.success = false(1,nFrames);
    results.blkOK = zeros(1,nFrames);
    results.nBlocks = zeros(1,nFrames);
    results.blkFailBase = zeros(1,nFrames);
    results.blkFailEnh = zeros(1,nFrames);
    results.blkFailCombined = zeros(1,nFrames);
    results.baseOnlyBlocks = zeros(1,nFrames);
    results.catastrophicBlocks = zeros(1,nFrames);
    results.policyTrace = cell(1,nFrames);
    results.frameDiagnostics = cell(1,nFrames);
end

function d = init_frame_diag(frameId, policy, nBlocks)
    d = struct();
    d.frameId = double(frameId);
    d.tx_mode = string(policy.tx_mode);
    d.nBlocks = double(nBlocks);
    d.policy = policy;
    d.events = {};
    d.concealedBlocks = [];
    d.blkOK = 0;
    d.blkFailBase = 0;
    d.blkFailEnh = 0;
    d.blkFailCombined = 0;
    d.baseOnlyBlocks = 0;
    d.catastrophicBlocks = 0;
    d.success = false;
    d.quality = struct();
    d.feedback = struct();
end

function c = init_counts()
    c = struct('blkOK',0,'blkFailBase',0,'blkFailEnh',0,'blkFailCombined',0,'baseOnlyBlocks',0,'catastrophicBlocks',0);
end

function c = merge_counts(c, blkRes)
    c.blkOK = c.blkOK + blkRes.blkOK;
    c.blkFailBase = c.blkFailBase + blkRes.blkFailBase;
    c.blkFailEnh = c.blkFailEnh + blkRes.blkFailEnh;
    c.blkFailCombined = c.blkFailCombined + blkRes.blkFailCombined;
    c.baseOnlyBlocks = c.baseOnlyBlocks + blkRes.baseOnlyBlocks;
    c.catastrophicBlocks = c.catastrophicBlocks + blkRes.catastrophicBlocks;
end

function [blkHat, blkRes] = run_single_dct_block_txrx(block, policy, frameId, blockId, chanParam, concealmentMode)
    N = 8;
    blocks = reshape(block, N, N, 1);
    [expBaseBytes, expEnhBytes, expCombinedBytes] = expected_payload_bytes(block, policy);

    blkRes = struct();
    blkRes.blkOK = 0;
    blkRes.blkFailBase = 0;
    blkRes.blkFailEnh = 0;
    blkRes.blkFailCombined = 0;
    blkRes.baseOnlyBlocks = 0;
    blkRes.catastrophicBlocks = 0;
    blkRes.concealed = false;
    blkRes.events = {};

    txMode = string(policy.tx_mode);
    if txMode == "combined"
        try
            [txWave, txInfo] = two_layer_tx_DCT(blocks, policy, frameId, blockId, 0);
            [rxWave, ~] = channel_model_v1(txWave, chanParam);
            [dctOut, rxInfo] = two_layer_rx_testing_dct(rxWave, txInfo);
            ok = isfield(rxInfo,'blockOK') && logical(rxInfo.blockOK);
            got = uint8([]);
            if isfield(rxInfo,'combinedBytes'), got = uint8(rxInfo.combinedBytes(:)); end
            payloadMatchCombined = bytes_equal(got, expCombinedBytes);
            if ok
                blkHat = dctOut(:,:,1);
                blkRes.blkOK = 1;
            else
                blkHat = conceal_block(concealmentMode);
                blkRes.blkFailCombined = 1;
                blkRes.catastrophicBlocks = 1;
                blkRes.concealed = true;
                extra = struct('expectedLen',numel(expCombinedBytes),'gotLen',numel(got),'payloadMatchCombined',payloadMatchCombined);
                blkRes.events{end+1} = make_event(blockId, "combined", rxInfo, extra);
            end
        catch ME
            blkHat = conceal_block(concealmentMode);
            blkRes.blkFailCombined = 1;
            blkRes.catastrophicBlocks = 1;
            blkRes.concealed = true;
            blkRes.events{end+1} = make_exception(blockId, "combined", ME);
        end
        return;
    end

    baseOK = false; enhOK = false;
    baseBlk = zeros(N,N); enhBlk = zeros(N,N);
    rxInfoBase = struct(); rxInfoEnh = struct();

    try
        [txWaveB, txInfoB] = two_layer_tx_DCT(blocks, policy, frameId, blockId, 1);
        [rxWaveB, ~] = channel_model_v1(txWaveB, chanParam);
        [baseOut, rxInfoBase] = two_layer_rx_testing_dct(rxWaveB, txInfoB);
        baseOK = isfield(rxInfoBase,'blockOK') && logical(rxInfoBase.blockOK);
        gotB = uint8([]);
        if isfield(rxInfoBase,'baseBytes'), gotB = uint8(rxInfoBase.baseBytes(:)); end
        payloadOkB = bytes_equal(gotB, expBaseBytes);
        baseOK = baseOK && payloadOkB;
        if baseOK, baseBlk = baseOut(:,:,1); end
    catch ME
        baseOK = false;
        rxInfoBase = struct('blockOK',false,'baseFailReason','exception','exceptionId',ME.identifier,'exceptionMsg',ME.message);
    end

    try
        [txWaveE, txInfoE] = two_layer_tx_DCT(blocks, policy, frameId, blockId, 2);
        [rxWaveE, ~] = channel_model_v1(txWaveE, chanParam);
        [enhOut, rxInfoEnh] = two_layer_rx_testing_dct(rxWaveE, txInfoE);
        enhOK = isfield(rxInfoEnh,'blockOK') && logical(rxInfoEnh.blockOK);
        gotE = uint8([]);
        if isfield(rxInfoEnh,'enhBytes'), gotE = uint8(rxInfoEnh.enhBytes(:)); end
        payloadMatchEnh = bytes_equal(gotE, expEnhBytes);
        if enhOK, enhBlk = enhOut(:,:,1); end
    catch ME
        enhOK = false;
        rxInfoEnh = struct('blockOK',false,'enhFailReason','exception','exceptionId',ME.identifier,'exceptionMsg',ME.message);
    end

    if enhOK && exist('payloadMatchEnh','var') && ~payloadMatchEnh
        blkRes.events{end+1} = make_diag_event( ...
            'blockId', double(blockId), ...
            'stage', "split_enh", ...
            'type', "enh_payload_mismatch", ...
            'payloadMatchEnh', payloadMatchEnh);
    end
    if enhOK && isstruct(rxInfoEnh) && isfield(rxInfoEnh,'sanityWarning') && logical(rxInfoEnh.sanityWarning)
        blkRes.events{end+1} = make_diag_event( ...
            'blockId', double(blockId), ...
            'stage', "split_enh", ...
            'type', "enh_sanity_warning", ...
            'sanityWarning', true);
    end

    if ~baseOK
        blkHat = conceal_block(concealmentMode);
        blkRes.blkFailBase = 1;
        blkRes.catastrophicBlocks = 1;
        blkRes.concealed = true;
        extraB = struct('expectedLen',numel(expBaseBytes));
        if exist('gotB','var'), extraB.gotLen = numel(gotB); end
        if exist('payloadOkB','var'), extraB.payloadOk = payloadOkB; end
        blkRes.events{end+1} = make_event(blockId, "split_base", rxInfoBase, extraB);
        if ~enhOK
            blkRes.blkFailEnh = 1;
            extraE = struct('expectedLen',numel(expEnhBytes));
            if exist('gotE','var'), extraE.gotLen = numel(gotE); end
            if exist('payloadMatchEnh','var'), extraE.payloadMatchEnh = payloadMatchEnh; end
            blkRes.events{end+1} = make_event(blockId, "split_enh", rxInfoEnh, extraE);
        end
        return;
    end

    blkRes.blkOK = 1;
    if enhOK
        blkHat = baseBlk + enhBlk;
    else
        blkHat = baseBlk;
        blkRes.blkFailEnh = 1;
        blkRes.baseOnlyBlocks = 1;
        extraE = struct('expectedLen',numel(expEnhBytes));
        if exist('gotE','var'), extraE.gotLen = numel(gotE); end
        if exist('payloadMatchEnh','var'), extraE.payloadMatchEnh = payloadMatchEnh; end
        blkRes.events{end+1} = make_event(blockId, "split_enh", rxInfoEnh, extraE);
    end
end

function [baseBytes, enhBytes, combinedBytes] = expected_payload_bytes(block, policy)
    N = 8;
    K = double(policy.K);
    Qb_list = [0.02 0.05 0.1 0.2];
    Qe_list = [0.02 0.05 0.1 0.2];

    qbIdx = max(0, min(numel(Qb_list)-1, double(policy.Qb_idx)));
    qeIdx = max(0, min(numel(Qe_list)-1, double(policy.Qe_idx)));
    Qb = Qb_list(qbIdx + 1);
    Qe = Qe_list(qeIdx + 1);

    [baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);
    [base_q, enh_q, ~, ~] = quant(baseCoeffs, enhCoeffs, Qb, Qe);

    baseBytes = uint8(typecast(int16(base_q(:)), 'uint8'));
    enhBytes  = uint8(typecast(int8(enh_q(:)), 'uint8'));
    baseBytes = baseBytes(:);
    enhBytes = enhBytes(:);
    combinedBytes = [baseBytes; enhBytes];
end

function ok = bytes_equal(a, b)
    a = uint8(a(:));
    b = uint8(b(:));
    ok = (numel(a) == numel(b)) && all(a == b);
end

function blk = conceal_block(mode)
    N = 8;
    mode = string(mode);
    if mode == "zero"
        blk = zeros(N,N);
    end
end

function e = make_exception(blockId, stage, ME)
    e = struct();
    e.blockId = double(blockId);
    e.stage = string(stage);
    e.type = "exception";
    e.identifier = string(ME.identifier);
    e.message = string(ME.message);
end

function e = make_event(blockId, stage, rxInfo, extra)
    e = struct();
    e.blockId = double(blockId);
    e.stage = string(stage);
    e.type = "decode_fail";
    if isstruct(rxInfo)
        if isfield(rxInfo,'baseFailReason'), e.baseFailReason = string(rxInfo.baseFailReason); end
        if isfield(rxInfo,'enhFailReason'), e.enhFailReason = string(rxInfo.enhFailReason); end
        if isfield(rxInfo,'combinedFailReason'), e.combinedFailReason = string(rxInfo.combinedFailReason); end
        if isfield(rxInfo,'sanityOK'), e.sanityOK = logical(rxInfo.sanityOK); end
        if isfield(rxInfo,'sanityWarning'), e.sanityWarning = logical(rxInfo.sanityWarning); end
        if isfield(rxInfo,'sanityLimit'), e.sanityLimit = double(rxInfo.sanityLimit); end
        if isfield(rxInfo,'sanityFailBlocks'), e.sanityFailBlocks = double(rxInfo.sanityFailBlocks); end
    end
    if e.stage == "split_enh"
        [e.enhFailReasonNorm, e.enhFailClass] = normalize_enh_reason(e);
    end
    if nargin >= 4 && ~isempty(extra)
        e.extra = extra;
    end
end

function e = make_diag_event(varargin)
    e = struct();
    for i = 1:2:numel(varargin)
        e.(varargin{i}) = varargin{i+1};
    end
    if isfield(e,'stage') && string(e.stage) == "split_enh"
        [e.enhFailReasonNorm, e.enhFailClass] = normalize_enh_reason(e);
    end
end

function [r, c] = normalize_enh_reason(e)
    c = "post_decode";
    t = "";
    if isfield(e,'type'), t = string(e.type); end
    if t == "enh_payload_mismatch"
        r = "payload_mismatch";
        return;
    end
    if t == "enh_sanity_warning"
        r = "sanity_warning";
        return;
    end
    c = "packet";
    raw = "";
    if isfield(e,'enhFailReason'), raw = string(e.enhFailReason); end
    raw = lower(strtrim(raw));
    if raw == ""
        r = "unknown";
        return;
    end
    if contains(raw,"crc")
        r = "crc_fail";
    elseif contains(raw,"input_too_short") || contains(raw,"too_short") || contains(raw,"pkt_too_short")
        r = "pkt_too_short";
    elseif contains(raw,"recover_fail") || contains(raw,"recovered_bits_too_short")
        r = "demodem_fail";
    elseif contains(raw,"exception") || contains(raw,"assert")
        r = "fatal_exception";
    else
        r = raw;
    end
end

function print_enh_diagnostics(r)
    enhFail = containers.Map('KeyType','char','ValueType','double');
    enhPost = containers.Map('KeyType','char','ValueType','double');
    enhRaw  = containers.Map('KeyType','char','ValueType','double');
    totalFailEvents = 0;
    totalPostEvents = 0;
    for f = 1:r.nFrames
        d = r.frameDiagnostics{f};
        if ~isstruct(d) || ~isfield(d,'events') || isempty(d.events), continue; end
        evs = d.events;
        if isstruct(evs), evs = num2cell(evs); end
        for i = 1:numel(evs)
            e = evs{i};
            if ~isstruct(e) || ~isfield(e,'stage') || string(e.stage) ~= "split_enh"
                continue;
            end
            if isfield(e,'enhFailReason') && strlength(string(e.enhFailReason)) > 0
                enhRaw = inc_map(enhRaw, char(string(e.enhFailReason)));
            end
            if isfield(e,'enhFailReasonNorm') && isfield(e,'enhFailClass')
                key = char(string(e.enhFailReasonNorm));
                cls = string(e.enhFailClass);
                if cls == "packet"
                    enhFail = inc_map(enhFail, key);
                    totalFailEvents = totalFailEvents + 1;
                else
                    enhPost = inc_map(enhPost, key);
                    totalPostEvents = totalPostEvents + 1;
                end
            else
                [key, cls] = normalize_enh_reason(e);
                if cls == "packet"
                    enhFail = inc_map(enhFail, char(key));
                    totalFailEvents = totalFailEvents + 1;
                else
                    enhPost = inc_map(enhPost, char(key));
                    totalPostEvents = totalPostEvents + 1;
                end
            end
        end
    end

    fprintf('\n================================================================\n');
    fprintf('ENHANCEMENT FAILURE DIAGNOSTICS\n');
    fprintf('================================================================\n');
    fprintf('Enh fail blocks (from counters): %d\n', sum(r.blkFailEnh));
    fprintf('Enh failure events counted:      %d\n', totalFailEvents);
    fprintf('Enh post-decode diag events:     %d\n', totalPostEvents);
    fprintf('\nEnh fail reason histogram (normalized):\n');
    print_hist(enhFail);
    fprintf('\nEnh post-decode diagnostics (not counted as failure):\n');
    print_hist(enhPost);
    fprintf('\nEnh raw reason histogram (as reported by rx):\n');
    print_hist(enhRaw);
    fprintf('================================================================\n');
end

function m = inc_map(m, key)
    if isempty(key), key = 'unknown'; end
    if isKey(m, key)
        m(key) = m(key) + 1;
    else
        m(key) = 1;
    end
end

function print_hist(m)
    ks = keys(m);
    if isempty(ks)
        fprintf('  (none)\n');
        return;
    end
    vals = cellfun(@(k) m(k), ks);
    [vals, idx] = sort(vals, 'descend');
    ks = ks(idx);
    for i = 1:numel(ks)
        fprintf('  %-24s %6d\n', ks{i}, vals(i));
    end
end

function s = summarize_feedback(feedback)
    s = struct();
    if ~isstruct(feedback), return; end
    if isfield(feedback,'meetsQualityTargets'), s.meetsQualityTargets = feedback.meetsQualityTargets; end
    if isfield(feedback,'needsUrgentChange'), s.needsUrgentChange = feedback.needsUrgentChange; end
    if isfield(feedback,'urgentReason'), s.urgentReason = feedback.urgentReason; end
    if isfield(feedback,'baseLayerStatus'), s.baseLayerStatus = feedback.baseLayerStatus; end
    if isfield(feedback,'detailLayerStatus'), s.detailLayerStatus = feedback.detailLayerStatus; end
end

function print_frame_log(frameIdx, snr, policy, ssimVal, psnrVal, counts, nBlocks)
    fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %7.2f | %4d/%-4d | %10d | %10d | %10d\n', ...
        frameIdx, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
        ssimVal, psnrVal, counts.blkOK, nBlocks, counts.blkFailBase, counts.blkFailEnh, counts.catastrophicBlocks);
end

function print_summary(r)
    ok = r.success;
    if any(ok)
        avgSsimOk = mean(r.ssim(ok));
        avgPsnrOk = mean(r.psnr(ok));
    else
        avgSsimOk = NaN;
        avgPsnrOk = NaN;
    end

    blkRatio = r.blkOK ./ max(1, r.nBlocks);
    fprintf('\n================================================================\n');
    fprintf('SUMMARY (DCT BENCHMARK)\n');
    fprintf('================================================================\n');
    fprintf('Avg SSIM (all frames):        %.4f\n', mean(r.ssim));
    fprintf('Avg PSNR (all frames):        %.2f dB\n', mean(r.psnr));
    fprintf('Avg SSIM (success frames):    %.4f\n', avgSsimOk);
    fprintf('Avg PSNR (success frames):    %.2f dB\n', avgPsnrOk);
    fprintf('Frame success rate:           %d/%d (%.1f%%)\n', sum(ok), r.nFrames, 100*sum(ok)/r.nFrames);
    fprintf('Avg block success ratio:      %.3f\n', mean(blkRatio));
    fprintf('Total base fail count:        %d\n', sum(r.blkFailBase));
    fprintf('Total enhancement fail count: %d\n', sum(r.blkFailEnh));
    fprintf('Total combined fail count:    %d\n', sum(r.blkFailCombined));
    fprintf('Total catastrophic blocks:    %d\n', sum(r.catastrophicBlocks));
    fprintf('================================================================\n');
end

function plot_results(r)
    frames = 1:r.nFrames;
    figure('Position',[80 80 1100 750],'Name','DCT Closed-Loop Benchmark');

    subplot(3,2,1);
    plot(frames, r.snr, 'b-o','MarkerSize',3);
    xlabel('Frame'); ylabel('SNR (dB)'); title('Channel SNR'); grid on;

    subplot(3,2,2);
    plot(frames, r.ssim, 'g-o','MarkerSize',3);
    xlabel('Frame'); ylabel('SSIM'); title('SSIM'); ylim([0 1]); grid on;

    subplot(3,2,3);
    plot(frames, r.psnr, 'b-o','MarkerSize',3);
    xlabel('Frame'); ylabel('PSNR (dB)'); title('PSNR'); grid on;

    subplot(3,2,4);
    plot(frames, r.mcs_base, 'b-s','MarkerSize',3); hold on;
    plot(frames, r.mcs_enh, 'r-o','MarkerSize',3); hold off;
    legend('Base','Enh'); xlabel('Frame'); ylabel('MCS'); title('MCS'); grid on;

    subplot(3,2,5);
    plot(frames, r.K, 'k-d','MarkerSize',3);
    xlabel('Frame'); ylabel('K'); title('K'); ylim([0 64]); grid on;

    subplot(3,2,6);
    plot(frames, r.blkFailBase, 'r-','LineWidth',1.5); hold on;
    plot(frames, r.blkFailEnh, 'm-','LineWidth',1.5);
    plot(frames, r.catastrophicBlocks, 'k-','LineWidth',1.5); hold off;
    legend('Base fail','Enh fail','Catastrophic');
    xlabel('Frame'); ylabel('# Blocks'); title('Block Failures'); grid on;
end

function [rStart, cStart] = block_pos(blockId, nBlocksW, N)
    bRow = floor(double(blockId) / double(nBlocksW));
    bCol = mod(double(blockId), double(nBlocksW));
    rStart = bRow*N + 1;
    cStart = bCol*N + 1;
end
