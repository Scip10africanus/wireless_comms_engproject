function [outputImage, feedback, results] = two_layer_rx_v2(rxWave, txInfo, settings)
% TWO_LAYER_RX_V2  Receiver for new TX pipeline
% rxWave    - received waveform (complex column)
% txInfo    - metadata struct from TX (lenBaseWave, cfgBase, policy, etc.)
% settings  - struct with .originalImage, .policyCsvPath, .useOctaveMode, etc.

    if nargin < 3, settings = struct(); end
    if ~isfield(settings, 'policyCsvPath'),      settings.policyCsvPath = ''; end
    if ~isfield(settings, 'verbose'),            settings.verbose = true; end
    if ~isfield(settings, 'useOctaveMode'),      settings.useOctaveMode = false; end
    if ~isfield(settings, 'useGaOptimizer'),     settings.useGaOptimizer = false; end
    if ~isfield(settings, 'gaSettings'),         settings.gaSettings = struct(); end
    if ~isfield(settings, 'useChannelPredictor'),settings.useChannelPredictor = false; end
    if ~isfield(settings, 'predictorParams'),    settings.predictorParams = struct(); end
    if ~isfield(settings, 'targetSsim'),         settings.targetSsim = 0.9; end
    if ~isfield(settings, 'targetPsnr'),         settings.targetPsnr = 30; end

    policy = txInfo.policy;
    K = double(policy.K);
    N = 8;
    nCoeffs = N*N;
    nBlocks = txInfo.nBlocks;
    imgH = txInfo.imgHeight;
    imgW = txInfo.imgWidth;

    Qb = max(1, double(policy.Qb_idx));
    Qe = max(1, double(policy.Qe_idx));
    if policy.Qb_idx == 0, Qb = 1; end
    if policy.Qe_idx == 0, Qe = 1; end

    results = struct('success', true, 'errors', {{}}, 'warnings', {{}});

    if settings.verbose
        fprintf('[RX] Policy v%d: K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d | %dx%d (%d blocks)\n', ...
            policy.version, K, policy.Qb_idx, policy.Qe_idx, ...
            policy.mcs_base, policy.mcs_enh, imgH, imgW, nBlocks);
    end

    %% 1 - Demux base and enhancement waveforms
    if settings.useOctaveMode
        fs = 20e6;
    else
        fs = wlanSampleRate(txInfo.cfgBase);
    end
    [rxBase, rxEnh] = layer_demux_bylen(rxWave, ...
        txInfo.lenBaseWave, txInfo.lenEnhWave, double(policy.mux_gap_us), fs);

    %% 2 - Demodulate
    if settings.useOctaveMode
        basePktBytes = demodem_simple(rxBase, txInfo.cfgBase);
        if txInfo.lenEnhWave > 0
            enhPktBytes = demodem_simple(rxEnh, txInfo.cfgEnh);
        else
            enhPktBytes = uint8([]);
        end
    else
        basePktBytes = demodem(rxBase, txInfo.cfgBase);
        enhPktBytes  = demodem(rxEnh,  txInfo.cfgEnh);
    end

    %% 3 - Parse packets -> extract coefficients
    baseParsed = parse_packets(basePktBytes, nBlocks);
    enhParsed  = parse_packets(enhPktBytes,  nBlocks);

    allBase = [];
    allEnh  = [];
    for b = 1:length(baseParsed)
        allBase = [allBase; baseParsed{b}.base_coeff_int16(:)];
    end
    for b = 1:length(enhParsed)
        allEnh = [allEnh; enhParsed{b}.enh_coeff_int8(:)];
    end

    %% 4 - Dequantize + IDCT -> reconstruct image
    dctInfo  = txInfo.dctInfo;
    Hp = dctInfo.Hp;  Wp = dctInfo.Wp;
    blockLen = dctInfo.block_length;
    nBlocksH = Hp / blockLen;
    nBlocksW = Wp / blockLen;

    reconDCT  = zeros(Hp, Wp);
    zigzagIdx = Zigzag_index(blockLen);
    bPtr = 1; ePtr = 1;

    for bRow = 1:nBlocksH
        for bCol = 1:nBlocksW
            % base coeffs
            if bPtr+K-1 <= length(allBase)
                bCoeff = allBase(bPtr:bPtr+K-1);
            else
                bCoeff = int16(zeros(K,1));
            end
            bPtr = bPtr + K;

            % enh coeffs
            eLen = nCoeffs - K;
            if ePtr+eLen-1 <= length(allEnh)
                eCoeff = allEnh(ePtr:ePtr+eLen-1);
            else
                eCoeff = int8(zeros(eLen,1));
            end
            ePtr = ePtr + eLen;

            % dequant -> zigzag -> block -> IDCT
            [baseHat, enhHat] = dequant(double(bCoeff), double(eCoeff), Qb, Qe);

            block = zeros(blockLen);
            block(zigzagIdx) = [baseHat; enhHat];

            rS = (bRow-1)*blockLen + 1;
            cS = (bCol-1)*blockLen + 1;
            reconDCT(rS:rS+blockLen-1, cS:cS+blockLen-1) = idct2(block);
        end
    end

    outputImage = uint8(max(0, min(255, round(reconDCT(1:imgH, 1:imgW)))));

    %% 5 - Quality metrics
    quality = struct('hasMetrics', false, 'ssim', NaN, 'psnr', NaN, 'mse', NaN);
    if isfield(settings, 'originalImage') && ~isempty(settings.originalImage)
        orig = double(settings.originalImage);
        rec  = double(outputImage);
        quality.mse  = mean((orig(:) - rec(:)).^2);
        quality.psnr = 10*log10(255^2 / (quality.mse + eps));
        quality.ssim = simple_ssim(orig, rec);
        quality.hasMetrics = true;
    end
    results.quality = quality;

    if settings.verbose && quality.hasMetrics
        fprintf('[RX] SSIM=%.4f  PSNR=%.2f dB\n', quality.ssim, quality.psnr);
    end

    %% 6 - Feedback + policy update
    feedback = rx_feedback(quality, results, policy, settings);
    results.feedback = feedback;

    if settings.verbose
        fprintf('[RX] Policy v%d -> v%d\n', policy.version, feedback.policy.version);
    end
end


%% ---- helpers ----

function ssimVal = simple_ssim(img1, img2)
% basic global SSIM (not windowed, good enough for testing)
    C1 = (0.01*255)^2;  C2 = (0.03*255)^2;
    mu1 = mean(img1(:));  mu2 = mean(img2(:));
    s1  = var(img1(:));   s2  = var(img2(:));
    s12 = mean((img1(:)-mu1).*(img2(:)-mu2));
    ssimVal = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1+s2+C2));
end

function parsed = parse_packets(pktBytes, expectedBlocks)
% try parsing as one big packet first, else find sync words
    parsed = {};
    pktBytes = uint8(pktBytes(:).');

    try
        parsed{1} = bitStream_parsing(pktBytes);
        return;
    catch
    end

    % find individual packets by sync word
    sync = uint8([hex2dec('AA') hex2dec('AA') hex2dec('AA') hex2dec('AA')]);
    pos = 1;  cnt = 0;

    while pos <= length(pktBytes)-12 && cnt < expectedBlocks
        % scan for sync
        found = false;
        for s = pos:length(pktBytes)-3
            if pktBytes(s)==sync(1) && pktBytes(s+1)==sync(2) && ...
               pktBytes(s+2)==sync(3) && pktBytes(s+3)==sync(4)
                pos = s; found = true; break;
            end
        end
        if ~found, break; end

        if pos+11 > length(pktBytes), break; end
        payloadLen = double(pktBytes(pos+11));
        crcFlag = bitand(bitshift(pktBytes(pos+7),-4), uint8(1));

        pktLen = 4 + 8 + payloadLen;
        if crcFlag, pktLen = pktLen + 2; end
        if pos+pktLen-1 > length(pktBytes), break; end

        try
            cnt = cnt + 1;
            parsed{cnt} = bitStream_parsing(pktBytes(pos:pos+pktLen-1));
        catch
            cnt = cnt - 1; % failed, undo
        end
        pos = pos + pktLen;
    end
end
