function [txWave, txInfo] = two_layer_tx_simple(img, pc, frameId)
% TWO_LAYER_TX_SIMPLE - Octave-compatible TX pipeline
%
% Same as two_layer_tx_pipeline but uses PolicyControllerSimple (struct)
% and modem_simple instead of WLAN Toolbox.
%
% Inputs:
%   img     - Input grayscale image (uint8, H x W)
%   pc      - PolicyControllerSimple struct (from PolicyControllerSimple())
%   frameId - Current frame number (0-255)
%
% Outputs:
%   txWave - Transmitted OFDM waveform
%   txInfo - Metadata struct for receiver

    %% Step 1: Get current policy
    policy = pc.tick(pc, frameId);

    K = double(policy.K);
    N = 8;
    nCoeffs = N * N;

    Qb = max(1, double(policy.Qb_idx));
    Qe = max(1, double(policy.Qe_idx));
    if policy.Qb_idx == 0, Qb = 1; end
    if policy.Qe_idx == 0, Qe = 1; end

    mcs_base = double(policy.mcs_base);
    mcs_enh  = double(policy.mcs_enh);
    mux_gap  = double(policy.mux_gap_us);

    fprintf('[TX] Frame %d: Policy v%d | K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d\n', ...
        frameId, policy.version, K, policy.Qb_idx, policy.Qe_idx, mcs_base, mcs_enh);

    %% Step 2: DCT transform
    imgDouble = double(img);
    [dctCoeffs, dctInfo] = DCT_Block_simple(imgDouble, N);

    Hp = dctInfo.Hp;
    Wp = dctInfo.Wp;
    nBlocksH = Hp / N;
    nBlocksW = Wp / N;
    nBlocks = nBlocksH * nBlocksW;

    %% Step 3: Process each block
    allBasePktBytes = [];
    allEnhPktBytes  = [];
    zigzagIdx = Zigzag_index(N);
    totalClipCount = 0;
    blockId = 0;

    for bRow = 1:nBlocksH
        for bCol = 1:nBlocksW
            rStart = (bRow-1)*N + 1;
            cStart = (bCol-1)*N + 1;
            block = dctCoeffs(rStart:rStart+N-1, cStart:cStart+N-1);

            [baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);
            [base_q, enh_q, ~, clipCnt] = quant(baseCoeffs, enhCoeffs, Qb, Qe);
            totalClipCount = totalClipCount + clipCnt;

            % Base packet - build params from policy (not pc.makeParams,
            % which reads stale pc.Policy due to struct pass-by-value)
            baseParams = make_params_from_policy(policy, frameId, blockId, 1, N);
            basePkt = bitStream_packet(baseParams, base_q, int8([]));
            allBasePktBytes = [allBasePktBytes, basePkt];

            % Enhancement packet
            enhParams = make_params_from_policy(policy, frameId, blockId, 3, N);
            enhPkt = bitStream_packet(enhParams, int16([]), enh_q);
            allEnhPktBytes = [allEnhPktBytes, enhPkt];

            blockId = blockId + 1;
        end
    end

    if totalClipCount > 0
        fprintf('[TX] Warning: %d coefficients clipped\n', totalClipCount);
    end

    %% Step 4: Modulate with simple OFDM
    [waveBase, cfgBase] = modem_simple(allBasePktBytes, mcs_base);

    if ~isempty(allEnhPktBytes)
        [waveEnh, cfgEnh] = modem_simple(allEnhPktBytes, mcs_enh);
    else
        waveEnh = zeros(0, 1);
        cfgEnh = struct();
    end

    %% Step 5: Multiplex
    fs = 20e6;
    gapSamp = round(mux_gap * 1e-6 * fs);
    txWave = [waveBase; zeros(gapSamp, 1); waveEnh];

    fprintf('[TX] Output: %d samples (base=%d, enh=%d)\n', ...
        length(txWave), length(waveBase), length(waveEnh));

    %% Step 6: txInfo
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

function params = make_params_from_policy(policy, frame_id, block_id, layer_type, N)
% Build params struct directly from policy (avoids struct pass-by-value stale data)
    params = struct();
    params.Qb_idx = uint8(policy.Qb_idx);
    params.Qe_idx = uint8(policy.Qe_idx);
    params.K      = uint8(policy.K);
    params.N      = uint8(N);
    params.codec_id = uint8(policy.codec_id);
    params.crc_flag = uint8(policy.crc_flag);
    params.endian_flag = uint8(policy.endian_flag);
    params.reserved_flag = uint8(policy.reserved_flag);

    if strcmp(policy.tx_mode, 'split')
        params.packet_mode = uint8(1);
        params.layer_type  = uint8(layer_type);
    else
        params.packet_mode = uint8(0);
        params.layer_type  = uint8(0);
    end

    params.frame_id = uint8(frame_id);
    params.block_id = uint16(block_id);
end
