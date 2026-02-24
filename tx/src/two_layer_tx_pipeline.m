function [txWave, txInfo] = two_layer_tx_pipeline(block, pc ,policy, frameId,blockId)
    N = 8;
    K = double(policy.K);
    Qb_list = [0.01 0.02 0.05 0.1];
    Qe_list = [0.01 0.02 0.05 0.1];
    Qb =  Qb_list(policy.Qb_idx + 1); 
    Qe =  Qe_list(policy.Qe_idx + 1);
    
    % split + quant
    [baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);
    [base_q, enh_q, ~, ~] = quant(baseCoeffs, enhCoeffs, Qb, Qe);

    % packetize
    if strcmp(policy.tx_mode,'split')
        baseParams = pc.makeParams(frameId, blockId, 1, N); % layer_type=1
        enhParams  = pc.makeParams(frameId, blockId, 3, N); % layer_type=3
        basePkt = bitStream_packet(baseParams, base_q, int8([]));
        enhPkt  = bitStream_packet(enhParams, int16([]), enh_q);
    else
        combParams = policy.makeParams(frameId, blockId, Policy.tx_mode, N);
        basePkt = bitStream_packet(combParams, base_q, enh_q);
        enhPkt  = uint8([]);
    end

    % 1 packet -> 1 PPDU
    [waveBase, cfgBase] = modem(basePkt, double(policy.mcs_base));

    if ~isempty(enhPkt) && double(policy.mcs_enh) > 0
        [waveEnh, cfgEnh] = modem(enhPkt, double(policy.mcs_enh));
    else
        waveEnh = complex(zeros(0,1));
        cfgEnh = [];
    end
        fs = wlanSampleRate(cfgBase);
    txWave = layer_mux(waveBase, waveEnh, policy.mux_gap_us, fs);
        % fprintf('[TX] Output: %d samples (base=%d, enh=%d, gap=%g us)\n', ...
        % length(txWave), length(waveBase), length(waveEnh), policy.mux_gap_us);
        % fprintf('[TX] Policy v%d: K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d \n', ...
        %     policy.version, K, policy.Qb_idx, policy.Qe_idx, ...
        %     policy.mcs_base, policy.mcs_enh);

    txInfo = struct();
    txInfo.waveBase = waveBase;
    txInfo.lenBaseWave = length(waveBase);
    txInfo.lenEnhWave = length(waveEnh);
    txInfo.cfgBase  = cfgBase;
    txInfo.basePktLen = numel(basePkt);
    txInfo.waveEnh = waveEnh;
    txInfo.cfgEnh  = cfgEnh;
    txInfo.enhPktLen = numel(enhPkt);
    txInfo.Qb = Qb; txInfo.Qe = Qe;
    txInfo.K = K;
    txInfo.frameId = frameId;
    txInfo.blockId = blockId;
    txInfo.policy = policy;
end
