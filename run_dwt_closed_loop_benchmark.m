function results = run_dwt_closed_loop_benchmark(opts)
% RUN_DWT_CLOSED_LOOP_BENCHMARK  DWT pipeline closed-loop benchmark
%
% Mirrors run_dct_closed_loop_benchmark.m but drives the DWT pipeline
% (two_layer_tx_wavelet + two_layer_rx_testing_dwt). Produces the same
% results-struct shape as the DCT benchmark so downstream analysis,
% plotting, and test harnesses are compatible.
%
% Unubold (Bold) - Apr 2026. Added to address Zibo's concern that the
% closed-loop pipeline had no DWT equivalent to the DCT benchmark. Zibo
% can use this for synthesis / DWT testbench validation.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'nFrames'),       opts.nFrames = 20; end
    if ~isfield(opts,'snrProfile'),    opts.snrProfile = 25 * ones(1, opts.nFrames); end
    if ~isfield(opts,'ricianK'),       opts.ricianK = 3; end
    if ~isfield(opts,'policyCsvPath'), opts.policyCsvPath = fullfile(tempdir(), 'policy_dwt_benchmark.csv'); end
    if ~isfield(opts,'verbose'),       opts.verbose = true; end
    if ~isfield(opts,'doPlot'),        opts.doPlot = true; end
    if ~isfield(opts,'targetSsim'),    opts.targetSsim = 0.9; end
    if ~isfield(opts,'targetPsnr'),    opts.targetPsnr = 30; end
    if ~isfield(opts,'level'),         opts.level = 1; end
    if ~isfield(opts,'rx_Qb'),         opts.rx_Qb = 1; end
    if ~isfield(opts,'rx_Qe'),         opts.rx_Qe = 1; end
    if ~isfield(opts,'useGaOptimizer'), opts.useGaOptimizer = true; end

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

    % Path sanity - same pattern as DCT benchmark so we fail loudly if an
    % old duplicate is shadowing the canonical rx/ versions.
    pTx = string(which('two_layer_tx_wavelet'));
    pRx = string(which('two_layer_rx_testing_dwt'));
    pCh = string(which('channel_model_v1'));
    pFb = string(which('rx_feedback'));
    pPc = string(which('PolicyController'));
    pDw = string(which('DWT'));
    pID = string(which('IDWT'));
    assert(strlength(pTx) > 0, 'two_layer_tx_wavelet not found on path');
    assert(strlength(pRx) > 0, 'two_layer_rx_testing_dwt not found on path');
    assert(strlength(pCh) > 0, 'channel_model_v1 not found on path');
    assert(strlength(pFb) > 0, 'rx_feedback not found on path');
    assert(strlength(pPc) > 0, 'PolicyController not found on path');
    assert(strlength(pDw) > 0, 'DWT not found on path');
    assert(strlength(pID) > 0, 'IDWT not found on path');
    assert(contains(pTx, fullfile(thisDir,'tx','src')), "Wrong two_layer_tx_wavelet resolved: %s", pTx);
    assert(contains(pRx, fullfile(thisDir,'rx')),       "Wrong two_layer_rx_testing_dwt resolved: %s", pRx);
    assert(contains(pCh, fullfile(thisDir,'channel')),  "Wrong channel_model_v1 resolved: %s", pCh);

    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end
    pc = PolicyController(opts.policyCsvPath);
    pc.Verbose = logical(opts.verbose);
    % Force split mode from frame 1 - wavelet pipeline is packet-per-block,
    % combined mode is not defined for it.
    pc.Policy.tx_mode = "split";

    testImg = im2double(imread(fullfile(thisDir,'tx','images','Testing examples.jpg')));
    if size(testImg,3) == 3, testImg = rgb2gray(testImg); end
    testImg = testImg(1:64, 1:64);

    % Scale to uint8 range - two_layer_tx_wavelet and quant() expect the
    % same numerical range the baseline DWT simulation uses (img = double
    % of uint8 image in close_sim_Tx_DWT.m). Keep the reference in the
    % same space for SSIM / PSNR comparison.
    testImgScaled = testImg * 255;

    channelHistory = struct('snr', []);
    chanParam = struct('channelType','RICIAN','SNR_dB',20,'ricianK',double(opts.ricianK));

    results = init_results(opts.nFrames);
    results.nFrames    = opts.nFrames;
    results.snrProfile = opts.snrProfile;
    results.pipeline   = "DWT";
    results.level      = double(opts.level);

    fprintf('\n================================================================\n');
    fprintf('DWT CLOSED-LOOP BENCHMARK | %dx%d | level=%d | frames=%d | RICIAN(K=%g)\n', ...
        size(testImg,1), size(testImg,2), double(opts.level), opts.nFrames, double(opts.ricianK));
    fprintf('================================================================\n');
    fprintf('%5s | %6s | %3s | %4s | %4s | %6s | %7s | %10s | %10s | %7s | %7s\n', ...
        'Frame','SNR','K','mcsB','mcsE','SSIM','PSNR','basePkt', 'enhPkt', 'basePER','enhPER');
    fprintf('%s\n', repmat('-',1,110));

    for f = 1:opts.nFrames
        frameId = uint8(mod(f-1, 256));
        snr = double(opts.snrProfile(f));
        channelHistory.snr(end+1) = snr;
        chanParam.SNR_dB = snr;
        chanParam.seed = double(frameId);

        policy = pc.tick(frameId);
        policy.use_crc16 = false;  % wavelet RX does not currently implement CRC
        results.policyTrace{f} = policy;

        frameDiag = init_frame_diag(frameId, policy);
        recon = zeros(size(testImgScaled));
        rxinfo = struct('basePER', 1, 'enhPER', 1, 'numBaseOk', 0, 'numEnhOk', 0, ...
                        'numBasePkts', 0, 'numEnhPkts', 0, ...
                        'baseFailReasons', {{}}, 'enhFailReasons', {{}});

        frameErr = "";
        try
            % TX
            [txWaves, txInfo] = two_layer_tx_wavelet(testImgScaled, policy, frameId);

            % Channel
            rxWaves = apply_channel_to_waves(txWaves, chanParam);

            % RX
            [res_hat, rxinfo] = two_layer_rx_testing_dwt(rxWaves, txInfo, opts.rx_Qb, opts.rx_Qe);

            % Reconstruct
            recon = IDWT(res_hat, txInfo.level);

        catch ME
            frameErr = string(ME.identifier) + ": " + string(ME.message);
            % Always surface the first-frame error — silent catches hide real bugs.
            % After that, throttle so a runaway loop doesn't flood the console.
            if opts.verbose || f <= 3
                fprintf('[Frame %d] TX/RX error: %s (%s)\n', f, ME.message, ME.identifier);
                if f == 1 && ~isempty(ME.stack)
                    fprintf('           at %s line %d\n', ME.stack(1).name, ME.stack(1).line);
                end
            end
        end

        % Clip to valid intensity range before measuring quality
        recon = max(0, min(255, recon));

        ssimVal = safe_ssim(recon, testImgScaled);
        psnrVal = safe_psnr(recon, testImgScaled);
        mseVal  = mean((testImgScaled(:) - recon(:)).^2);

        quality = struct('hasMetrics', true, 'ssim', ssimVal, 'psnr', psnrVal, 'mse', mseVal);

        % The DWT pipeline doesn't have block-level OK/fail counters in the
        % same shape as DCT, so we synthesise an equivalent from PER.
        basePktOk = double(rxinfo.numBaseOk);
        enhPktOk  = double(rxinfo.numEnhOk);
        basePktN  = double(rxinfo.numBasePkts);
        enhPktN   = double(rxinfo.numEnhPkts);
        totalOk   = basePktOk + enhPktOk;
        totalN    = basePktN + enhPktN;

        frameSuccess = (basePktOk >= basePktN) && (strlength(frameErr) == 0);
        rxRes = struct();
        rxRes.success       = frameSuccess;
        rxRes.blkOK         = totalOk;
        rxRes.nBlocks       = max(1, totalN);
        rxRes.enhFailRatio  = double(rxinfo.enhPER);
        rxRes.baseOnlyRatio = double(basePktOk) / max(1, basePktN) * double(enhPktOk < enhPktN);

        rxSettings = struct();
        rxSettings.isDWT = true;
        rxSettings.policyCsvPath       = opts.policyCsvPath;
        rxSettings.verbose             = logical(opts.verbose);
        rxSettings.useGaOptimizer = opts.useGaOptimizer;
        rxSettings.useChannelPredictor = false;
        rxSettings.channelHistory      = channelHistory;
        rxSettings.targetSsim          = double(opts.targetSsim);
        rxSettings.targetPsnr          = double(opts.targetPsnr);
        rxSettings.oracleSNR           = snr;
        rxSettings.oracleSNRNote       = "DWT benchmark uses simulation SNR rather than receiver-estimated SNR.";

        feedback = rx_feedback(quality, rxRes, policy, rxSettings);

        results.snr(f)         = snr;
        results.ssim(f)        = ssimVal;
        results.psnr(f)        = psnrVal;
        results.mse(f)         = mseVal;
        results.K(f)           = double(policy.K);
        results.mcs_base(f)    = double(policy.mcs_base);
        results.mcs_enh(f)     = double(policy.mcs_enh);
        results.success(f)     = frameSuccess;
        results.blkOK(f)       = totalOk;
        results.nBlocks(f)     = max(1, totalN);
        results.basePER(f)     = double(rxinfo.basePER);
        results.enhPER(f)      = double(rxinfo.enhPER);
        results.numBaseOk(f)   = basePktOk;
        results.numEnhOk(f)    = enhPktOk;
        results.numBasePkts(f) = basePktN;
        results.numEnhPkts(f)  = enhPktN;

        frameDiag.numBaseOk    = basePktOk;
        frameDiag.numEnhOk     = enhPktOk;
        frameDiag.numBasePkts  = basePktN;
        frameDiag.numEnhPkts   = enhPktN;
        frameDiag.basePER      = double(rxinfo.basePER);
        frameDiag.enhPER       = double(rxinfo.enhPER);
        frameDiag.baseFailReasons = rxinfo.baseFailReasons;
        frameDiag.enhFailReasons  = rxinfo.enhFailReasons;
        frameDiag.success      = frameSuccess;
        frameDiag.quality      = quality;
        frameDiag.feedback     = summarize_feedback(feedback);
        frameDiag.frameErr     = frameErr;
        results.frameDiagnostics{f} = frameDiag;

        print_frame_log(f, snr, policy, ssimVal, psnrVal, basePktOk, basePktN, enhPktOk, enhPktN, rxinfo.basePER, rxinfo.enhPER);
    end

    print_summary(results);
    print_dwt_failure_histogram(results);
    if opts.doPlot
        plot_results(results);
    end
end


% -----------------------------------------------------------------------
% Helpers
% -----------------------------------------------------------------------

function results = init_results(nFrames)
    results = struct();
    results.nFrames        = nFrames;
    results.snrProfile     = [];
    results.snr            = zeros(1,nFrames);
    results.ssim           = zeros(1,nFrames);
    results.psnr           = zeros(1,nFrames);
    results.mse            = zeros(1,nFrames);
    results.K              = zeros(1,nFrames);
    results.mcs_base       = zeros(1,nFrames);
    results.mcs_enh        = zeros(1,nFrames);
    results.success        = false(1,nFrames);
    results.blkOK          = zeros(1,nFrames);
    results.nBlocks        = zeros(1,nFrames);
    results.basePER        = zeros(1,nFrames);
    results.enhPER         = zeros(1,nFrames);
    results.numBaseOk      = zeros(1,nFrames);
    results.numEnhOk       = zeros(1,nFrames);
    results.numBasePkts    = zeros(1,nFrames);
    results.numEnhPkts     = zeros(1,nFrames);
    results.policyTrace    = cell(1,nFrames);
    results.frameDiagnostics = cell(1,nFrames);
end

function d = init_frame_diag(frameId, policy)
    d = struct();
    d.frameId = double(frameId);
    d.tx_mode = string(policy.tx_mode);
    d.policy = policy;
    d.numBaseOk = 0; d.numEnhOk = 0;
    d.numBasePkts = 0; d.numEnhPkts = 0;
    d.basePER = 1; d.enhPER = 1;
    d.baseFailReasons = {};
    d.enhFailReasons = {};
    d.success = false;
    d.quality = struct();
    d.feedback = struct();
    d.frameErr = "";
end

function rxWaves = apply_channel_to_waves(txWaves, chanParam)
    rxWaves = struct('base', [], 'enh', {{}});
    if isfield(txWaves,'base') && ~isempty(txWaves.base)
        rxWaves.base = cell(size(txWaves.base));
        for i = 1:numel(txWaves.base)
            [rx, ~] = channel_model_v1(txWaves.base{i}, chanParam);
            % Pad a couple of samples so timing estimator has room at the tail,
            % matching the same trick used in bold_closed_loop_ml.m
            rxWaves.base{i} = [rx; zeros(10,1)];
        end
    else
        rxWaves.base = {};
    end
    if isfield(txWaves,'enh') && ~isempty(txWaves.enh)
        rxWaves.enh = cell(size(txWaves.enh));
        for i = 1:numel(txWaves.enh)
            [rx, ~] = channel_model_v1(txWaves.enh{i}, chanParam);
            rxWaves.enh{i} = [rx; zeros(10,1)];
        end
    else
        rxWaves.enh = {};
    end
end

function s = safe_ssim(a, b)
    try
        s = ssim(a, b);
    catch
        s = 0;
    end
    if ~isfinite(s), s = 0; end
end

function p = safe_psnr(a, b)
    try
        p = psnr(a, b, 255);
    catch
        p = 0;
    end
    if ~isfinite(p), p = 0; end
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

function print_frame_log(f, snr, policy, ssimVal, psnrVal, basePktOk, basePktN, enhPktOk, enhPktN, basePER, enhPER)
    fprintf('%5d | %6.1f | %3d | %4d | %4d | %6.4f | %7.2f | %4d/%-4d | %4d/%-4d | %6.1f%% | %6.1f%%\n', ...
        f, snr, double(policy.K), double(policy.mcs_base), double(policy.mcs_enh), ...
        ssimVal, psnrVal, basePktOk, basePktN, enhPktOk, enhPktN, 100*basePER, 100*enhPER);
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

    fprintf('\n================================================================\n');
    fprintf('SUMMARY (DWT BENCHMARK)\n');
    fprintf('================================================================\n');
    fprintf('Avg SSIM (all frames):        %.4f\n', mean(r.ssim));
    fprintf('Avg PSNR (all frames):        %.2f dB\n', mean(r.psnr));
    fprintf('Avg SSIM (success frames):    %.4f\n', avgSsimOk);
    fprintf('Avg PSNR (success frames):    %.2f dB\n', avgPsnrOk);
    fprintf('Frame success rate:           %d/%d (%.1f%%)\n', sum(ok), r.nFrames, 100*sum(ok)/r.nFrames);
    fprintf('Avg base PER:                 %.2f%%\n', 100*mean(r.basePER));
    fprintf('Avg enh PER:                  %.2f%%\n', 100*mean(r.enhPER));
    fprintf('Total base pkt OK:            %d / %d\n', sum(r.numBaseOk), sum(r.numBasePkts));
    fprintf('Total enh pkt OK:             %d / %d\n', sum(r.numEnhOk),  sum(r.numEnhPkts));
    fprintf('================================================================\n');
end

function print_dwt_failure_histogram(r)
    baseHist = containers.Map('KeyType','char','ValueType','double');
    enhHist  = containers.Map('KeyType','char','ValueType','double');
    totalBaseFail = 0; totalEnhFail = 0;

    for f = 1:r.nFrames
        d = r.frameDiagnostics{f};
        if ~isstruct(d), continue; end
        if isfield(d,'baseFailReasons') && ~isempty(d.baseFailReasons)
            for i = 1:numel(d.baseFailReasons)
                key = norm_dwt_reason(d.baseFailReasons{i});
                baseHist = inc_map(baseHist, key);
                totalBaseFail = totalBaseFail + 1;
            end
        end
        if isfield(d,'enhFailReasons') && ~isempty(d.enhFailReasons)
            for i = 1:numel(d.enhFailReasons)
                key = norm_dwt_reason(d.enhFailReasons{i});
                enhHist = inc_map(enhHist, key);
                totalEnhFail = totalEnhFail + 1;
            end
        end
    end

    fprintf('\n================================================================\n');
    fprintf('DWT FAILURE DIAGNOSTICS\n');
    fprintf('================================================================\n');
    fprintf('Total base pkt failures: %d\n', totalBaseFail);
    fprintf('Total enh  pkt failures: %d\n', totalEnhFail);
    fprintf('\nBase pkt failure reasons:\n');
    print_hist(baseHist);
    fprintf('\nEnh pkt failure reasons:\n');
    print_hist(enhHist);
    fprintf('================================================================\n');
end

function key = norm_dwt_reason(raw)
    % Fail reasons from two_layer_rx_testing_dwt have the form:
    %   "base block 3: recover_fail:someId"
    %   "enh block 12 (LH1 #3): input_too_short"
    % Strip the block index noise and bucket by the demodem reason tail.
    s = char(string(raw));
    colons = strfind(s, ':');
    if isempty(colons)
        key = strtrim(s);
    else
        % take everything after the first colon, then bucket on keyword
        tail = strtrim(s(colons(1)+1:end));
        tail = lower(tail);
        if contains(tail, 'input_too_short') || contains(tail, 'too_short')
            key = 'pkt_too_short';
        elseif contains(tail, 'recovered_bits_too_short')
            key = 'recovered_bits_too_short';
        elseif contains(tail, 'recover_fail') || contains(tail, 'exception') || contains(tail, 'assert')
            key = 'recover_fail';
        elseif contains(tail, 'unpack_fail')
            key = 'unpack_fail';
        else
            key = tail;
        end
    end
    if isempty(key), key = 'unknown'; end
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
        fprintf('  %-28s %6d\n', ks{i}, vals(i));
    end
end

function plot_results(r)
    frames = 1:r.nFrames;
    figure('Position',[80 80 1100 750],'Name','DWT Closed-Loop Benchmark');

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
    plot(frames, r.mcs_enh,  'r-o','MarkerSize',3); hold off;
    legend('Base','Enh'); xlabel('Frame'); ylabel('MCS'); title('MCS'); grid on;

    subplot(3,2,5);
    plot(frames, r.K, 'k-d','MarkerSize',3);
    xlabel('Frame'); ylabel('K'); title('K (policy)'); grid on;

    subplot(3,2,6);
    plot(frames, 100*r.basePER, 'r-','LineWidth',1.5); hold on;
    plot(frames, 100*r.enhPER,  'm-','LineWidth',1.5); hold off;
    legend('Base PER','Enh PER');
    xlabel('Frame'); ylabel('PER (%)'); title('Packet Error Rate'); grid on;

    sgtitle(sprintf('DWT Closed-Loop Benchmark | Avg SSIM=%.4f | Avg PSNR=%.2f dB', ...
        mean(r.ssim), mean(r.psnr)));
end
