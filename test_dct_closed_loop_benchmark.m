function test_dct_closed_loop_benchmark()
    fprintf('\n================================================================\n');
    fprintf('TEST: DCT closed-loop benchmark integration\n');
    fprintf('================================================================\n');

    rng(1);

    opts = struct();
    opts.nFrames = 50;
    opts.snrProfile = [ ...
          31 29 27 24 21 18 15 12 10  8 ...
          10 12 15 18 22 26 30 33 28 24 ...
          20 16 13 11  9  8  9  9  9  9 ...
           9  9  9 34 38 41 36 30 24 24 ...
          24 24 24 10 10 10 10 10 10 10];
    opts.ricianK = 3;
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

