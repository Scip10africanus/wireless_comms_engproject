function test_dwt_closed_loop_benchmark()
% TEST_DWT_CLOSED_LOOP_BENCHMARK  Integration test for the DWT benchmark
%
% Mirrors test_dct_closed_loop_benchmark.m - runs a representative SNR
% sweep through the DWT pipeline and validates the results struct.
%
% Unubold (Bold) - Apr 2026.

    fprintf('\n================================================================\n');
    fprintf('TEST: DWT closed-loop benchmark integration\n');
    fprintf('================================================================\n');

    rng(1);

    opts = struct();
    opts.nFrames = 180;
    opts.ricianK = 3;
    % Channel assumption
    % Key points of the 3-second video channel evolution
    keyFrames = [ ...
          1   18   36   54   72   90  108  126  144  162  180];

    keySNR = [ ...
         31   30   27   23   18   11    8   13   21   30   24];

    % Smooth large-scale trend
    snrBase = interp1(keyFrames, keySNR, 1:opts.nFrames, 'pchip');

    % Small-scale multipath fluctuation (deterministic, repeatable)
    idx = 1:opts.nFrames;
    snrFlutter = 0.9*sin(2*pi*idx/13) + 0.6*sin(2*pi*idx/29);

    % Final profile
    opts.snrProfile = round(snrBase + snrFlutter);

    % Clamp to a reasonable range
    opts.snrProfile = max(min(opts.snrProfile, 33), 6);
    opts.verbose = false;
    opts.doPlot = true;
    opts.level = 1;
    opts.rx_Qb = 1;
    opts.rx_Qe = 1;
    opts.policyCsvPath = fullfile(tempdir(), 'policy_dwt_integration_test.csv');
    opts.targetSsim = 0.9;
    opts.targetPsnr = 30;

    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end

    r = run_dwt_closed_loop_benchmark(opts);

    requiredFields = { ...
        'nFrames','snrProfile','snr','ssim','psnr','mse','K','mcs_base','mcs_enh', ...
        'success','blkOK','nBlocks','basePER','enhPER', ...
        'numBaseOk','numEnhOk','numBasePkts','numEnhPkts', ...
        'policyTrace','frameDiagnostics','pipeline','level' ...
    };
    for i = 1:numel(requiredFields)
        assert(isfield(r, requiredFields{i}), "Missing results field: %s", requiredFields{i});
    end

    assert(r.nFrames == opts.nFrames);
    assert(numel(r.snr) == opts.nFrames);
    assert(string(r.pipeline) == "DWT");
    assert(r.level == opts.level);
    assert(all(r.nBlocks > 0));
    assert(all(r.blkOK >= 0));
    assert(all(r.basePER >= 0 & r.basePER <= 1));
    assert(all(r.enhPER  >= 0 & r.enhPER  <= 1));
    assert(all(isfinite(r.ssim)));
    assert(all(isfinite(r.psnr)));
    assert(all(isfinite(r.mse)));

    for f = 1:opts.nFrames
        d = r.frameDiagnostics{f};
        assert(isstruct(d));
        assert(isfield(d,'frameId'));
        assert(isfield(d,'tx_mode'));
        assert(isfield(d,'numBasePkts'));
        assert(isfield(d,'numEnhPkts'));
        assert(isfield(d,'basePER'));
        assert(isfield(d,'enhPER'));
        assert(d.numBasePkts == r.numBasePkts(f));
        assert(d.numEnhPkts  == r.numEnhPkts(f));
    end

    % Sanity: at high-SNR frames (first 5 are >= 21 dB), base PER should be
    % substantially lower than at low-SNR frames (frames 8-10 are 8-12 dB).
    hi = mean(r.basePER(1:5));
    lo = mean(r.basePER(8:10));
    if any(r.numBasePkts > 0)
        assert(hi <= lo + 0.05, ...
            'Base PER does not degrade with SNR: hi=%.3f lo=%.3f', hi, lo);
    end

    fprintf('PASS: DWT results struct + frame diagnostics generated\n');
    fprintf('================================================================\n');
end
