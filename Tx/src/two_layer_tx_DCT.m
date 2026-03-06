function [txWave, txInfo] = two_layer_tx_DCT(block, policy, frameId, blockId, layer_type)
N = 8;
K = double(policy.K);
Qb_list = [0.01 0.02 0.05 0.1];
Qe_list = [0.01 0.02 0.05 0.1];
Qb =  Qb_list(policy.Qb_idx + 1);
Qe =  Qe_list(policy.Qe_idx + 1);

txInfo = struct();
waveBase = complex(zeros(0,1));
waveEnh  = complex(zeros(0,1));
cfgBase  = [];
cfgEnh   = [];
basePkt  = uint8([]);
enhPkt   = uint8([]);

% split + quant
[baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);
[base_q, enh_q, ~, ~] = quant(baseCoeffs, enhCoeffs, Qb, Qe);
baseBytes = typecast(int16(base_q(:)), 'uint8');   % 2*K bytes
enhBytes  = typecast(int8(enh_q(:)),  'uint8');    % (64-K) bytes
% fprintf("[TX] baseBytes=%d enhBytes=%d |\n", numel(baseBytes), numel(enhBytes));
% packetize
if strcmp(policy.tx_mode,'split')
    Tx_mode = 1;
    txInfo.Tx_mode = Tx_mode;
else
    Tx_mode = 0;
    txInfo.Tx_mode = Tx_mode;
end

basePkt = baseBytes;
enhPkt  = enhBytes;

% 1 packet -> 1 PPDU
if Tx_mode == 1
    txInfo.layer_type = layer_type;
    if layer_type == 2 % Enhance layer
        [waveEnhance, cfgEnhance] = modem(enhPkt, double(policy.mcs_enh));
        waveEnh = waveEnhance;
        txInfo.lenEnhWave = length(waveEnh);
        cfgEnh = cfgEnhance;
        txInfo.enhPktLen = numel(enhPkt);
        txInfo.cfgEnh  = cfgEnh;
        fs = wlanSampleRate(cfgEnh);
        txInfo.Qb_idx = policy.Qb_idx;
        txInfo.Qe_idx = policy.Qe_idx;
    else % Base layer
        [waveBase, cfgBase] = modem(basePkt, double(policy.mcs_base));
        txInfo.lenBaseWave = length(waveBase);
        waveEnh = complex(zeros(0,1));
        txInfo.cfgBase  = cfgBase;
        fs = wlanSampleRate(cfgBase);
        txInfo.basePktLen = numel(basePkt);
        txInfo.Qb_idx = policy.Qb_idx;
        txInfo.Qe_idx = policy.Qe_idx;
    end
else
    combinedPkt = [basePkt; enhPkt];
    [waveBase, cfgBase] = modem(combinedPkt, double(policy.mcs_base));
    txInfo.lenBaseWave = length(waveBase);
    waveEnh = complex(zeros(0,1));
    fs = wlanSampleRate(cfgBase);
    txInfo.cfgBase  = cfgBase;
    txInfo.basePktLen = numel(basePkt);
    txInfo.enhPktLen = numel(enhPkt);
    txInfo.Qb_idx = policy.Qb_idx;
    txInfo.Qe_idx = policy.Qe_idx;
end

txWave = layer_mux(waveBase, waveEnh, policy.mux_gap_us, fs);
% fprintf('[TX] Output: %d samples (base=%d, enh=%d, gap=%g us)\n', ...
% length(txWave), length(waveBase), length(waveEnh), policy.mux_gap_us);
% fprintf('[TX] Policy v%d: K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d \n', ...
%     policy.version, K, policy.Qb_idx, policy.Qe_idx, ...
%     policy.mcs_base, policy.mcs_enh);

txInfo.K = K;
txInfo.N = N;
txInfo.mux_gap_us= policy.mux_gap_us;
txInfo.frameId = frameId;
txInfo.blockId = blockId;
txInfo.policy = policy;
end
