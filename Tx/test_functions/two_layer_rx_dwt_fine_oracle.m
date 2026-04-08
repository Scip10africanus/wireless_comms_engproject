function [img_hat, rxInfo] = two_layer_rx_dwt_fine_oracle(rxWaves, txInfo)
    
    rxInfo = struct();
    rxInfo.baseOk = false;
    rxInfo.enhOk = false;
    rxInfo.baseFailReason = '';
    rxInfo.enhFailReason = '';
    
    rxInfo.baseFailCount = 0;
    rxInfo.enhFailCount = 0;
    
    % LL payload
    sb_name = 'LL';
    len = txInfo.subbands.(sb_name).payloadBaseLen;
    sb_payloads_base_LL = zeros(len, 1, 'int16');
    
    % LH_HL_pool
    lh_hl_num_base = txInfo.subbands.LH_HL_pool.num_base;
    lh_hl_num_enh = txInfo.subbands.LH_HL_pool.num_enh;
    H = txInfo.subbands.LH_HL_pool.origH;
    W = txInfo.subbands.LH_HL_pool.origW;
    total_lh_hl = 2 * H * W;
    
    lh_hl_base_data = zeros(lh_hl_num_base, 1, 'int16');
    lh_hl_base_idx = zeros(lh_hl_num_base, 1);
    
    lh_hl_enh_data = zeros(lh_hl_num_enh, 1, 'int8');
    lh_hl_enh_idx = zeros(lh_hl_num_enh, 1);
    
    % HH_pool
    hh_num_enh = txInfo.subbands.HH_pool.num_enh;
    hh_enh_data = zeros(hh_num_enh, 1, 'int8');
    hh_enh_idx = zeros(hh_num_enh, 1);
    
    % Base Layer
    if txInfo.numBasePkts > 0
        for i = 1:txInfo.numBasePkts
            expected_len = txInfo.basePktLens(i);
            meta = txInfo.basePktMeta{i};
            
            pkt_ok = false;
            
            if i <= length(rxWaves.base)
                try
                    [rxBytes, ok, reason] = demodem(rxWaves.base{i}, txInfo.cfgBase{i});
                    if ok && length(rxBytes) == expected_len
                        rxData = typecast(rxBytes(:), 'int16');
                        
                        if strcmp(meta.subband, 'LL')
                            start_idx = (meta.startBlock - 1) * (meta.payloadLenBytes / meta.bytesPerElem / meta.numBlocks) + 1;
                            end_idx = start_idx + length(rxData) - 1;
                            sb_payloads_base_LL(start_idx:end_idx) = rxData;
                        elseif strcmp(meta.subband, 'LH_HL_pool')
                            start_idx = meta.startIdx;
                            end_idx = meta.endIdx;
                            lh_hl_base_data(start_idx:end_idx) = rxData;
                            lh_hl_base_idx(start_idx:end_idx) = meta.global_indices;
                        end
                        pkt_ok = true;
                    else
                        rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
                    end
                catch ME
                    rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
                end
            else
                rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
            end
        end
        rxInfo.baseOk = (rxInfo.baseFailCount < txInfo.numBasePkts); 
    end
    
    % Enhancement Layer
    if txInfo.numEnhPkts > 0
        for i = 1:txInfo.numEnhPkts
            expected_len = txInfo.enhPktLens(i);
            meta = txInfo.enhPktMeta{i};
            
            pkt_ok = false;
            
            if i <= length(rxWaves.enh)
                try
                    [rxBytes, ok, reason] = demodem(rxWaves.enh{i}, txInfo.cfgEnh{i});
                    if ok && length(rxBytes) == expected_len
                        rxData = typecast(rxBytes(:), 'int8');
                        
                        if strcmp(meta.subband, 'LH_HL_pool')
                            start_idx = meta.startIdx;
                            end_idx = meta.endIdx;
                            lh_hl_enh_data(start_idx:end_idx) = rxData;
                            lh_hl_enh_idx(start_idx:end_idx) = meta.global_indices;
                        elseif strcmp(meta.subband, 'HH_pool')
                            start_idx = meta.startIdx;
                            end_idx = meta.endIdx;
                            hh_enh_data(start_idx:end_idx) = rxData;
                            hh_enh_idx(start_idx:end_idx) = meta.global_indices;
                        end
                        pkt_ok = true;
                    else
                        rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
                    end
                catch ME
                    rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
                end
            else
                rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
            end
        end
        rxInfo.enhOk = (rxInfo.enhFailCount < txInfo.numEnhPkts); 
    end
    
    res_hat = struct();
    res_hat.subbands = struct('LH', {}, 'HL', {}, 'HH', {}, 'inputSize', {});
    for k = 1:txInfo.level
        res_hat.subbands(k).LH = [];
        res_hat.subbands(k).HL = [];
        res_hat.subbands(k).HH = [];
        res_hat.subbands(k).inputSize = [];
    end
    
    % Process LL
    Qb_list = [1 2 4 8];
    Qe_list = [1 2 4 8];
    Qb = Qb_list(min(txInfo.policy.Qb_idx + 1, length(Qb_list)));
    Qe = Qe_list(min(txInfo.policy.Qe_idx + 1, length(Qe_list)));
    
    sb_info_LL = txInfo.subbands.LL;
    res_hat.finalLL = decode_subband_base_only(sb_payloads_base_LL, sb_info_LL);
    
    % Reconstruct LH and HL from pool
    lh_hl_pool = zeros(total_lh_hl, 1);
    
    % Apply valid base indices
    valid_base = lh_hl_base_idx > 0;
    lh_hl_pool(lh_hl_base_idx(valid_base)) = double(lh_hl_base_data(valid_base)) * Qb;
    
    % Apply valid enh indices
    valid_enh = lh_hl_enh_idx > 0;
    lh_hl_pool(lh_hl_enh_idx(valid_enh)) = double(lh_hl_enh_data(valid_enh)) * Qe;
    
    % Split pool into LH and HL
    lh_flat = lh_hl_pool(1:(H*W));
    hl_flat = lh_hl_pool((H*W + 1):end);
    
    res_hat.subbands(1).LH = reshape(lh_flat, H, W);
    res_hat.subbands(1).HL = reshape(hl_flat, H, W);
    
    % Reconstruct HH from pool
    hh_pool = zeros(H*W, 1);
    valid_hh_enh = hh_enh_idx > 0;
    hh_pool(hh_enh_idx(valid_hh_enh)) = double(hh_enh_data(valid_hh_enh)) * Qe;
    
    res_hat.subbands(1).HH = reshape(hh_pool, H, W);
    
    % Infer inputSize for IDWT
    res_hat.subbands(1).inputSize = [sb_info_LL.origH, sb_info_LL.origW] * 2;
    
    img_hat = IDWT(res_hat, txInfo.level);
    rxInfo.recoveredSize = size(img_hat);
end

function sb_hat = decode_subband_base_only(payload_int16, sb_info)
    N = sb_info.N;
    sb_padded = zeros(sb_info.padH, sb_info.padW);
    
    idx = 1;
    for r = 1:sb_info.nBlocksH
        for c = 1:sb_info.nBlocksW
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            
            block_data = double(payload_int16(idx : idx+N*N-1));
            
            % No IDCT!
            block = reshape(block_data, N, N) * sb_info.Q;
            
            sb_padded(rStart:rStart+N-1, cStart:cStart+N-1) = block;
            idx = idx + N*N;
        end
    end
    
    sb_hat = sb_padded(1:sb_info.origH, 1:sb_info.origW);
end
