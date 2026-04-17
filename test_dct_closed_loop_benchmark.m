function test_dct_closed_loop_benchmark()
    fprintf('\n================================================================\n');
    fprintf('TEST: DCT closed-loop benchmark integration\n');
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
    opts.concealmentMode = 'zero';
    opts.policyCsvPath = fullfile(tempdir(), 'policy_dct_integration_test.csv');
    opts.targetSsim = 0.9;
    opts.targetPsnr = 30;

    if exist(opts.policyCsvPath,'file'), delete(opts.policyCsvPath); end

    r = run_dct_closed_loop_benchmark(opts);

    requiredFields = { ...
        'nFrames','snrProfile','snr','ssim','psnr','mse','K','mcs_base','mcs_enh', ...
        'success','blkOK','nBlocks','blkFailBase','blkFailEnh','blkFailCombined', ...
        'baseOnlyBlocks','catastrophicBlocks','policyTrace','frameDiagnostics' ...
    };
    for i = 1:numel(requiredFields)
        assert(isfield(r, requiredFields{i}), "Missing results field: %s", requiredFields{i});
    end

    assert(r.nFrames == opts.nFrames);
    assert(numel(r.snr) == opts.nFrames);
    assert(all(r.nBlocks > 0));
    assert(all(r.blkOK >= 0 & r.blkOK <= r.nBlocks));
    assert(all(r.catastrophicBlocks >= 0 & r.catastrophicBlocks <= r.nBlocks));
    assert(all(r.baseOnlyBlocks >= 0 & r.baseOnlyBlocks <= r.nBlocks));
    assert(all(isfinite(r.ssim)));
    assert(all(isfinite(r.psnr)));
    assert(all(isfinite(r.mse)));

    for f = 1:opts.nFrames
        d = r.frameDiagnostics{f};
        assert(isstruct(d));
        assert(isfield(d,'frameId'));
        assert(isfield(d,'tx_mode'));
        assert(isfield(d,'nBlocks'));
        assert(isfield(d,'blkOK'));
        assert(isfield(d,'catastrophicBlocks'));
        assert(d.nBlocks == r.nBlocks(f));
        assert(d.blkOK == r.blkOK(f));
        assert(d.catastrophicBlocks == r.catastrophicBlocks(f));
    end

    fprintf('PASS: results struct + frame diagnostics generated\n');
    fprintf('================================================================\n');
end

