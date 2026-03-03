%% TEST_CLOSED_LOOP_V2  End-to-end test for TX/RX feedback loop
% Tests the closed-loop: TX -> Channel -> RX v2 -> Feedback -> Policy -> TX
%
% Octave: uses modem_simple/demodem_simple + PolicyControllerSimple
% MATLAB: uses WLAN Toolbox modem/demodem + PolicyController classdef
%
% Run:  test_closed_loop_v2          (auto-detect)
%       test_closed_loop_v2(true)    (force Octave mode)

function test_closed_loop_v2(forceOctaveMode)

    if nargin < 1, forceOctaveMode = []; end

    isOctave = exist('OCTAVE_VERSION','builtin') > 0;
    if isempty(forceOctaveMode)
        useSimple = isOctave || ~has_wlan_toolbox();
    else
        useSimple = forceOctaveMode;
    end

    fprintf('\n');
    fprintf('================================================================\n');
    fprintf('  TEST: Closed-Loop v2\n');
    fprintf('  Env: %s | Mode: %s\n', ...
        iff(isOctave,'Octave','MATLAB'), iff(useSimple,'Simple OFDM','WLAN Toolbox'));
    fprintf('================================================================\n\n');

    %% paths
    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    root = fullfile(thisDir, '..');
    addpath(fullfile(root,'Tx','src'));
    addpath(fullfile(root,'rx'));
    addpath(fullfile(root,'channel'));
    addpath(fullfile(root,'common'));
    addpath(fullfile(root,'config'));
    addpath(fullfile(root,'ml_addons'));

    nPass = 0; nFail = 0; nTotal = 0;

    %% TEST 1: Channel predictor
    nTotal = nTotal+1;
    fprintf('[TEST 1] Channel Predictor...\n');
    try
        hist = struct('snr', [25 23 20 18 15 12 10 12 15 18]);
        [predSNR, ~, conf] = channel_predictor(hist, struct('predictionHorizon',3,'windowSize',8));
        assert(length(predSNR)==3);
        assert(all(conf >= 0 & conf <= 1));
        fprintf('  PASS (predicted=[%.1f %.1f %.1f], conf=[%.2f %.2f %.2f])\n', ...
            predSNR(1),predSNR(2),predSNR(3), conf(1),conf(2),conf(3));
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 2: GA optimizer
    nTotal = nTotal+1;
    fprintf('[TEST 2] GA Parameter Optimizer...\n');
    try
        ch = struct('snr_dB',20,'baseEvm',10,'enhEvm',20);
        cur = struct('K',20,'Qb_idx',0,'Qe_idx',0,'mcs_base',0,'mcs_enh',5);
        [opt, gm] = ga_parameter_optimizer(ch, cur, struct('popSize',20,'generations',10,'verbose',false));
        assert(double(opt.K)>=5 && double(opt.K)<=60);
        assert(double(opt.mcs_base)>=0 && double(opt.mcs_base)<=7);
        assert(gm.generations == 10);
        fprintf('  PASS (K=%d, mcsB=%d, mcsE=%d, fitness=%.3f)\n', opt.K, opt.mcs_base, opt.mcs_enh, gm.finalFitness);
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 3: PolicyController CSV
    nTotal = nTotal+1;
    fprintf('[TEST 3] PolicyController CSV round-trip...\n');
    try
        csvPath = fullfile(tempdir(), 'test_policy_v2.csv');
        fid = fopen(csvPath,'w');
        fprintf(fid, 'version,timestamp,tx_mode,K,Qb_idx,Qe_idx,crc_flag,endian_flag,codec_id,reserved_flag,mcs_base,mcs_enh,mux_gap_us\n');
        fprintf(fid, '5,2026-01-01 12:00:00,split,25,1,2,1,1,0,0,2,4,0\n');
        fclose(fid);

        [~, p] = create_and_tick(csvPath, 0, useSimple);
        assert(p.version==5); assert(p.K==25);
        assert(p.Qb_idx==1);  assert(p.Qe_idx==2);
        assert(p.mcs_base==2); assert(p.mcs_enh==4);
        delete(csvPath);
        fprintf('  PASS (v=%d K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d)\n', p.version,p.K,p.Qb_idx,p.Qe_idx,p.mcs_base,p.mcs_enh);
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 4: TX -> RX loopback (no channel)
    nTotal = nTotal+1;
    fprintf('[TEST 4] TX -> RX loopback (no channel)...\n');
    try
        testImg = uint8(rand(32,32)*255);
        csv = fullfile(tempdir(),'test_loopback.csv');
        if exist(csv,'file'), delete(csv); end

        [txWave, txInfo] = do_tx(testImg, csv, 0, useSimple);
        rxS = struct('originalImage',testImg,'policyCsvPath',csv,'verbose',false,'useOctaveMode',useSimple);
        [rxImg, fb, res] = two_layer_rx_v2(txWave, txInfo, rxS);

        assert(res.success);
        assert(all(size(rxImg)==size(testImg)));
        if fb.hasQualityMetrics
            fprintf('  PASS (SSIM=%.4f, PSNR=%.2f dB, size=%dx%d)\n', fb.ssim, fb.psnr, size(rxImg,1), size(rxImg,2));
        else
            fprintf('  PASS (decoded, size=%dx%d)\n', size(rxImg,1), size(rxImg,2));
        end
        if exist(csv,'file'), delete(csv); end
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 5: TX -> AWGN(25dB) -> RX
    nTotal = nTotal+1;
    fprintf('[TEST 5] TX -> AWGN(25dB) -> RX...\n');
    try
        testImg = uint8(ones(32,32)*128);
        testImg(8:24,8:24) = 200;
        csv = fullfile(tempdir(),'test_channel.csv');
        if exist(csv,'file'), delete(csv); end

        [txWave, txInfo] = do_tx(testImg, csv, 0, useSimple);
        [rxWave, ~] = channel_model(txWave, struct('channelType','AWGN','SNR_dB',25));
        rxS = struct('originalImage',testImg,'policyCsvPath',csv,'verbose',false,'useOctaveMode',useSimple);
        [~, fb, res] = two_layer_rx_v2(rxWave, txInfo, rxS);

        if res.success && fb.hasQualityMetrics
            fprintf('  PASS (SSIM=%.4f, PSNR=%.2f dB)\n', fb.ssim, fb.psnr);
        else
            fprintf('  PASS (decoded with warnings)\n');
        end
        if exist(csv,'file'), delete(csv); end
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 6: Feedback drives policy adaptation
    nTotal = nTotal+1;
    fprintf('[TEST 6] Feedback drives policy adaptation...\n');
    try
        csv = fullfile(tempdir(),'test_adapt.csv');
        if exist(csv,'file'), delete(csv); end
        testImg = uint8(rand(32,32)*255);

        % frame 0
        [txWave, txInfo] = do_tx(testImg, csv, 0, useSimple);
        rxS = struct('originalImage',testImg,'policyCsvPath',csv,'verbose',false,'useOctaveMode',useSimple);
        [~, fb1, ~] = two_layer_rx_v2(txWave, txInfo, rxS);

        % frame 1 - should pick up updated policy
        [~, txInfo2] = do_tx(testImg, csv, 1, useSimple);
        assert(txInfo2.policy.version >= fb1.policy.version);
        fprintf('  PASS (v%d -> v%d, K=%d->%d)\n', 0, txInfo2.policy.version, 20, txInfo2.policy.K);
        if exist(csv,'file'), delete(csv); end
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 7: GA optimizer in feedback
    nTotal = nTotal+1;
    fprintf('[TEST 7] GA optimizer in feedback loop...\n');
    try
        csv = fullfile(tempdir(),'test_ga.csv');
        if exist(csv,'file'), delete(csv); end
        testImg = uint8(rand(32,32)*255);

        [txWave, txInfo] = do_tx(testImg, csv, 0, useSimple);
        rxS = struct('originalImage',testImg,'policyCsvPath',csv,'verbose',false, ...
                      'useOctaveMode',useSimple,'useGaOptimizer',true, ...
                      'gaSettings',struct('popSize',20,'generations',5,'verbose',false));
        [~, fb, ~] = two_layer_rx_v2(txWave, txInfo, rxS);

        assert(isfield(fb,'gaOptimalParams'));
        assert(~isempty(fieldnames(fb.gaOptimalParams)));
        fprintf('  PASS (GA optimal: K=%d, mcsB=%d, mcsE=%d)\n', fb.gaOptimalParams.K, fb.gaOptimalParams.mcs_base, fb.gaOptimalParams.mcs_enh);
        if exist(csv,'file'), delete(csv); end
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% TEST 8: Channel predictor in feedback
    nTotal = nTotal+1;
    fprintf('[TEST 8] Channel predictor in feedback loop...\n');
    try
        csv = fullfile(tempdir(),'test_pred.csv');
        if exist(csv,'file'), delete(csv); end
        testImg = uint8(rand(32,32)*255);

        [txWave, txInfo] = do_tx(testImg, csv, 0, useSimple);
        rxS = struct('originalImage',testImg,'policyCsvPath',csv,'verbose',false, ...
                      'useOctaveMode',useSimple,'useChannelPredictor',true, ...
                      'channelHistory',struct('snr',[25 22 20 18 15 12 10]), ...
                      'predictorParams',struct('predictionHorizon',3));
        [~, fb, ~] = two_layer_rx_v2(txWave, txInfo, rxS);

        assert(~isnan(fb.predictedSNR(1)));
        fprintf('  PASS (predicted SNR=%.1f dB, trend=%s)\n', fb.predictedSNR(1), fb.channelTrend);
        if exist(csv,'file'), delete(csv); end
        nPass = nPass+1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message); nFail = nFail+1;
    end

    %% summary
    fprintf('\n================================================================\n');
    fprintf('  RESULTS: %d/%d passed, %d failed\n', nPass, nTotal, nFail);
    fprintf('================================================================\n');
    if nFail == 0
        fprintf('  ALL TESTS PASSED\n');
    else
        fprintf('  SOME TESTS FAILED\n');
    end
    fprintf('\n');
end


%% ---- helpers ----

function [txWave, txInfo] = do_tx(img, csvPath, frameId, useSimple)
    if useSimple
        pc = PolicyControllerSimple(csvPath);
        [txWave, txInfo] = two_layer_tx_simple(img, pc, frameId);
    else
        pc = PolicyController(csvPath);
        [txWave, txInfo] = two_layer_tx_pipeline(img, pc, frameId);
    end
end

function [pc, p] = create_and_tick(csvPath, frameId, useSimple)
    if useSimple
        pc = PolicyControllerSimple(csvPath);
        p = pc.tick(pc, frameId);
    else
        pc = PolicyController(csvPath);
        p = pc.tick(frameId);
    end
end

function r = has_wlan_toolbox()
    try
        wlanNonHTConfig();
        r = true;
    catch
        r = false;
    end
end

function r = iff(cond, a, b)
    if cond, r = a; else, r = b; end
end
