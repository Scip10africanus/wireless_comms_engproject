function [txWave, txInfo] = two_layer_tx_DCT(blocks, policy, frameId, blockId, layer_type)
% blocks is N x N x numBlocks
N = 8;
K = double(policy.K);
Qb_list = [0.02 0.05 0.1 0.2];
Qe_list = [0.02 0.05 0.1 0.2];
Qb =  Qb_list(policy.Qb_idx + 1);
Qe =  Qe_list(policy.Qe_idx + 1);

numBlocks = size(blocks, 3);
txInfo = struct();
txInfo.numBlocks = numBlocks;

waveBase = complex(zeros(0,1));
waveEnh  = complex(zeros(0,1));
cfgBase  = [];
cfgEnh   = [];
basePkt  = uint8([]);
enhPkt   = uint8([]);

baseBytesAll = uint8(zeros(2*K*numBlocks, 1));
enhBytesAll  = uint8(zeros((N*N-K)*numBlocks, 1));

for i = 1:numBlocks
    block = blocks(:,:,i);
    [baseCoeffs, enhCoeffs] = Layer_Split(K, N, block);
    [base_q, enh_q, ~, ~] = quant(baseCoeffs, enhCoeffs, Qb, Qe);
    
    b_idx = (i-1)*2*K + 1;
    e_idx = (i-1)*(N*N-K) + 1;
    
    baseBytesAll(b_idx : b_idx + 2*K - 1) = typecast(int16(base_q(:)), 'uint8');
    enhBytesAll(e_idx : e_idx + (N*N-K) - 1) = typecast(int8(enh_q(:)), 'uint8');
end

if strcmp(policy.tx_mode,'split')
    Tx_mode = 1;
    txInfo.Tx_mode = Tx_mode;
else
    Tx_mode = 0;
    txInfo.Tx_mode = Tx_mode;
end

basePkt = baseBytesAll;
enhPkt  = enhBytesAll;

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
