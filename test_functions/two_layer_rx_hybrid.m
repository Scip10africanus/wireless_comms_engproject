function [img_hat, rxInfo] = two_layer_rx_hybrid(rxWaves, txInfo) 
    rxInfo = struct();
    rxInfo.baseOk = false;
    rxInfo.enhOk = false;
    rxInfo.baseFailReason = '';
    rxInfo.enhFailReason = '';
    
    rxInfo.baseFailCount = 0;
    rxInfo.enhFailCount = 0;
    
    % Pre-allocate full subband payloads based on info
    sb_payloads_base = struct();
    sb_payloads_enh = struct();
    
    for i = 1:length(txInfo.subband_order_base)
        sb_name = txInfo.subband_order_base{i};
        len = txInfo.subbands.(sb_name).payloadBaseLen;
        sb_payloads_base.(sb_name) = zeros(len, 1, 'int16');
    end
    
    for i = 1:length(txInfo.subband_order_enh)
        sb_name = txInfo.subband_order_enh{i};
        len = txInfo.subbands.(sb_name).payloadEnhLen;
        sb_payloads_enh.(sb_name) = zeros(len, 1, 'int8');
    end
    
    % Base Layer
    if txInfo.numBasePkts > 0
        for i = 1:txInfo.numBasePkts
            expected_len = txInfo.basePktLens(i);
            meta = txInfo.basePktMeta{i};
            sb_name = meta.subband;
            
            pkt_ok = false;
            
            if i <= length(rxWaves.base)
                try
                    [rxBytes, ok, reason] = demodem(rxWaves.base{i}, txInfo.cfgBase{i});
                    if ok && length(rxBytes) == expected_len
                        % Reconstruct correct type
                        rxData = typecast(rxBytes(:), 'int16');
                        
                        % Place in correct position
                        start_idx = (meta.startBlock - 1) * (meta.payloadLenBytes / meta.bytesPerElem / meta.numBlocks) + 1;
                        end_idx = start_idx + length(rxData) - 1;
                        
                        start_idx = round(start_idx);
                        end_idx = round(end_idx);
                        
                        if start_idx >= 1 && end_idx <= length(sb_payloads_base.(sb_name))
                            sb_payloads_base.(sb_name)(start_idx:end_idx) = rxData;
                            pkt_ok = true;
                        else
                            rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
                            if isempty(rxInfo.baseFailReason), rxInfo.baseFailReason = 'Index out of bounds'; end
                        end
                    else
                        rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
                        if isempty(rxInfo.baseFailReason), rxInfo.baseFailReason = reason; end
                    end
                catch ME
                    rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
                    if isempty(rxInfo.baseFailReason), rxInfo.baseFailReason = ME.message; end
                end
            else
                rxInfo.baseFailCount = rxInfo.baseFailCount + 1;
            end
            
            if ~pkt_ok
                % Missing data remains zero due to pre-allocation
            end
        end
        rxInfo.baseOk = (rxInfo.baseFailCount < txInfo.numBasePkts); 
    end
    
    % Enhancement Layer
    if txInfo.numEnhPkts > 0
        for i = 1:txInfo.numEnhPkts
            expected_len = txInfo.enhPktLens(i);
            meta = txInfo.enhPktMeta{i};
            sb_name = meta.subband;
            
            pkt_ok = false;
            
            if i <= length(rxWaves.enh)
                try
                    [rxBytes, ok, reason] = demodem(rxWaves.enh{i}, txInfo.cfgEnh{i});
                    if ok && length(rxBytes) == expected_len
                        rxData = typecast(rxBytes(:), 'int8');
                        
                        start_idx = (meta.startBlock - 1) * (meta.payloadLenBytes / meta.bytesPerElem / meta.numBlocks) + 1;
                        end_idx = start_idx + length(rxData) - 1;
                        
                        start_idx = round(start_idx);
                        end_idx = round(end_idx);
                        
                        if start_idx >= 1 && end_idx <= length(sb_payloads_enh.(sb_name))
                            sb_payloads_enh.(sb_name)(start_idx:end_idx) = rxData;
                            pkt_ok = true;
                        else
                            rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
                            if isempty(rxInfo.enhFailReason), rxInfo.enhFailReason = 'Index out of bounds'; end
                        end
                    else
                        rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
                        if isempty(rxInfo.enhFailReason), rxInfo.enhFailReason = reason; end
                    end
                catch ME
                    rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
                    if isempty(rxInfo.enhFailReason), rxInfo.enhFailReason = ME.message; end
                end
            else
                rxInfo.enhFailCount = rxInfo.enhFailCount + 1;
            end
            
            if ~pkt_ok
                % Missing data remains zero
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
    
    % Process LL (Base)
    sb_name = 'LL';
    sb_info = txInfo.subbands.(sb_name);
    sb_hat = decode_subband_base_only(sb_payloads_base.(sb_name), sb_info);
    res_hat.finalLL = sb_hat;
    
    % Process LH (Split)
    sb_name = 'LH';
    sb_info = txInfo.subbands.(sb_name);
    sb_hat = decode_subband_split(sb_payloads_base.(sb_name), sb_payloads_enh.(sb_name), sb_info);
    res_hat.subbands(1).LH = sb_hat;
    
    % Process HL (Split)
    sb_name = 'HL';
    sb_info = txInfo.subbands.(sb_name);
    sb_hat = decode_subband_split(sb_payloads_base.(sb_name), sb_payloads_enh.(sb_name), sb_info);
    res_hat.subbands(1).HL = sb_hat;
    
    % Process HH (Enh only)
    sb_name = 'HH';
    sb_info = txInfo.subbands.(sb_name);
    sb_hat = decode_subband_enh_only(sb_payloads_enh.(sb_name), sb_info);
    res_hat.subbands(1).HH = sb_hat;
    
    % Infer inputSize for IDWT
    res_hat.subbands(1).inputSize = [txInfo.subbands.LL.origH, txInfo.subbands.LL.origW] * 2;
    
    img_hat = IDWT(res_hat, txInfo.level);
    
    % Metadata
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
            q_blk = reshape(block_data, N, N) * sb_info.Q;
            block = idct2(q_blk);
            
            sb_padded(rStart:rStart+N-1, cStart:cStart+N-1) = block;
            idx = idx + N*N;
        end
    end
    
    sb_hat = sb_padded(1:sb_info.origH, 1:sb_info.origW);
end

function sb_hat = decode_subband_split(payload_base, payload_enh, sb_info)
    N = sb_info.N;
    Kb = sb_info.Kb;
    Ke = sb_info.Ke;
    Qb = sb_info.Qb;
    Qe = sb_info.Qe;
    
    sb_padded = zeros(sb_info.padH, sb_info.padW);
    
    zz_idx = Zigzag_index(N);
    base_idx = zz_idx(1:Kb);
    enh_idx = zz_idx(Kb+1 : Kb+Ke);
    
    pb_idx = 1;
    pe_idx = 1;
    
    for r = 1:sb_info.nBlocksH
        for c = 1:sb_info.nBlocksW
            q_blk = zeros(N, N);
            
            b_coeffs = double(payload_base(pb_idx : pb_idx+Kb-1));
            q_blk(base_idx) = b_coeffs(:) .* Qb(:);
            pb_idx = pb_idx + Kb;
            
            e_coeffs = double(payload_enh(pe_idx : pe_idx+Ke-1));
            q_blk(enh_idx) = e_coeffs(:) .* Qe(:);
            pe_idx = pe_idx + Ke;
            
            block = idct2(q_blk);
            
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            sb_padded(rStart:rStart+N-1, cStart:cStart+N-1) = block;
        end
    end
    
    sb_hat = sb_padded(1:sb_info.origH, 1:sb_info.origW);
end

function sb_hat = decode_subband_enh_only(payload_enh, sb_info)
    N = sb_info.N;
    Ke = sb_info.Ke;
    Qe = sb_info.Qe;
    
    sb_padded = zeros(sb_info.padH, sb_info.padW);
    
    zz_idx = Zigzag_index(N);
    enh_idx = zz_idx(1:Ke);
    
    pe_idx = 1;
    
    for r = 1:sb_info.nBlocksH
        for c = 1:sb_info.nBlocksW
            q_blk = zeros(N, N);
            
            e_coeffs = double(payload_enh(pe_idx : pe_idx+Ke-1));
            q_blk(enh_idx) = e_coeffs(:) .* Qe(:);
            pe_idx = pe_idx + Ke;
            
            block = idct2(q_blk);
            
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            sb_padded(rStart:rStart+N-1, cStart:cStart+N-1) = block;
        end
    end
    
    sb_hat = sb_padded(1:sb_info.origH, 1:sb_info.origW);
end
