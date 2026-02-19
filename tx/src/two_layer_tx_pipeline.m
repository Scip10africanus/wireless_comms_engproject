function [txWave, txInfo] = two_layer_tx_pipeline(img, policyCtrl, frameId)
% TWO_LAYER_TX_PIPELINE - New TX pipeline using modular components
%
% Wraps DCT_Block, Layer_Split, quant, bitStream_packet, modem, and
% layer_mux into a complete transmit chain driven by PolicyController.
%
% The PolicyController reads a policy CSV (written by RX feedback) to
% adapt K, quantization, and MCS each frame.
%
% Inputs:
%   img        - Input grayscale image (uint8, H x W)
%   policyCtrl - PolicyController handle (reads policy.csv)
%   frameId    - Current frame number (uint8, 0-255)
%
% Outputs:
%   txWave - Transmitted waveform (complex column vector)
%   txInfo - Metadata struct for the receiver:
%       .lenBaseWave, .lenEnhWave  - Waveform lengths for demuxing
%       .cfgBase, .cfgEnh          - wlanNonHTConfig objects
%       .nBlocks                   - Number of DCT blocks
%       .imgHeight, .imgWidth      - Image dimensions
%       .policy                    - Policy snapshot used
%       .dctInfo                   - DCT processing info
%       .basePktLen, .enhPktLen    - Packet byte lengths
%
% Example:
%   policyCtrl = PolicyController('policy.csv');
%   img = uint8(rand(64,64)*255);
%   [txWave, txInfo] = two_layer_tx_pipeline(img, policyCtrl, 0);

    %% Step 1: Get current policy
    policy = policyCtrl.tick(frameId);

    K = double(policy.K);
    N = 8;  % DCT block size (always 8x8)
    nCoeffs = N * N;  % 64 coefficients per block

    % Quantization step sizes
    Qb = max(1, double(policy.Qb_idx));
    Qe = max(1, double(policy.Qe_idx));
    if policy.Qb_idx == 0, Qb = 1; end
    if policy.Qe_idx == 0, Qe = 1; end

    mcs_base = double(policy.mcs_base);
    mcs_enh  = double(policy.mcs_enh);
    mux_gap  = double(policy.mux_gap_us);

    fprintf('[TX] Frame %d: Policy v%d | K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d\n', ...
        frameId, policy.version, K, policy.Qb_idx, policy.Qe_idx, mcs_base, mcs_enh);

    %% Step 2: DCT transform the image
    imgDouble = double(img);
    [dctCoeffs, dctInfo] = DCT_Block(imgDouble, N);

    Hp = dctInfo.Hp;
    Wp = dctInfo.Wp;
    nBlocksH = Hp / N;
    nBlocksW = Wp / N;
    nBlocks = nBlocksH * nBlocksW;

    %% Step 3: Process each block - split, quantize, packetize
    allBasePktBytes = [];
    allEnhPktBytes  = [];
    zigzagIdx = Zigzag_index(N);

    totalClipCount = 0;
    blockId = 0;

    for bRow = 1:nBlocksH
        for bCol = 1:nBlocksW
            % Extract this block
            rStart = (bRow-1)*N + 1;
            cStart = (bCol-1)*N + 1;
            block = dctCoeffs(rStart:rStart+N-1, cStart:cStart+N-1);

            % Split into base and enhancement via zigzag ordering
            [baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);

            % Quantize
            [base_q, enh_q, ~, clipCnt] = quant(baseCoeffs, enhCoeffs, Qb, Qe);
            totalClipCount = totalClipCount + clipCnt;

            % Build packet parameters
            if strcmp(policy.tx_mode, 'split')
                % Split mode: separate base and enhancement packets

                % Base packet
                baseParams = policyCtrl.makeParams(frameId, blockId, 1, N);  % layer_type=1 (base)
                basePkt = bitStream_packet(baseParams, base_q, int8([]));
                allBasePktBytes = [allBasePktBytes, basePkt];

                % Enhancement packet
                enhParams = policyCtrl.makeParams(frameId, blockId, 3, N);  % layer_type=3 (enh)
                enhPkt = bitStream_packet(enhParams, int16([]), enh_q);
                allEnhPktBytes = [allEnhPktBytes, enhPkt];
            else
                % Combined mode: single packet with both layers
                combParams = policyCtrl.makeParams(frameId, blockId, 0, N);
                combPkt = bitStream_packet(combParams, base_q, enh_q);
                % In combined mode, put everything in base waveform
                allBasePktBytes = [allBasePktBytes, combPkt];
            end

            blockId = blockId + 1;
        end
    end

    if totalClipCount > 0
        fprintf('[TX] Warning: %d enhancement coefficients clipped to int8 range\n', totalClipCount);
    end

    %% Step 4: Modulate each layer using WLAN Non-HT
    fprintf('[TX] Modulating: base=%d bytes (MCS %d), enh=%d bytes (MCS %d)\n', ...
        length(allBasePktBytes), mcs_base, length(allEnhPktBytes), mcs_enh);

    [waveBase, cfgBase] = modem(allBasePktBytes, mcs_base);

    if ~isempty(allEnhPktBytes)
        [waveEnh, cfgEnh] = modem(allEnhPktBytes, mcs_enh);
    else
        waveEnh = zeros(0, 1);
        cfgEnh = [];
    end

    %% Step 5: Multiplex base and enhancement waveforms
    fs = wlanSampleRate(cfgBase);
    txWave = layer_mux(waveBase, waveEnh, mux_gap, fs);

    fprintf('[TX] Output: %d samples (base=%d, enh=%d, gap=%g us)\n', ...
        length(txWave), length(waveBase), length(waveEnh), mux_gap);

    %% Step 6: Build txInfo for the receiver
    txInfo = struct();
    txInfo.lenBaseWave = length(waveBase);
    txInfo.lenEnhWave  = length(waveEnh);
    txInfo.cfgBase     = cfgBase;
    txInfo.cfgEnh      = cfgEnh;
    txInfo.nBlocks     = nBlocks;
    txInfo.imgHeight   = dctInfo.H;
    txInfo.imgWidth    = dctInfo.W;
    txInfo.policy      = policy;
    txInfo.dctInfo     = dctInfo;
    txInfo.basePktLen  = length(allBasePktBytes);
    txInfo.enhPktLen   = length(allEnhPktBytes);
    txInfo.frameId     = frameId;
end
