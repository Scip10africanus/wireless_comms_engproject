function [txwaves, txInfo] = two_layer_tx_dwt_fine_oracle(img, policy, frameId)
    level = 1; % DWT level 
    N = 8; % Block size used in Hybrid (for payload matching reference)
    
    Kb = double(policy.K);
    Kb = max(1, min(Kb, N*N));
    if isfield(policy, 'Ke')
        Ke = double(policy.Ke);
    else
        Ke = Kb;
    end
    Ke = max(1, min(Ke, N*N - Kb));
    Khh = Ke; % For HH
    
    Qb_list = [1 2 4 8];
    Qe_list = [1 2 4 8];
    Qb = Qb_list(min(policy.Qb_idx + 1, length(Qb_list)));
    Qe = Qe_list(min(policy.Qe_idx + 1, length(Qe_list)));

    txInfo = struct();
    txInfo.level = level;
    txInfo.frameId = frameId;
    txInfo.policy = policy;
    txInfo.basePktMeta = {};
    txInfo.enhPktMeta = {};
    
    info = struct();
    [dwt_res, dwt_info] = DWT(img, info, level); %% 1-level DWT
    txInfo.dwt_info = dwt_info;
    
    base_pkts_data = {};
    enh_pkts_data = {};
    
    % Calculate Hybrid Target Base Payload (in terms of number of int16 coefficients)
    % Hybrid puts entire LL in base.
    LL = dwt_res.finalLL;
    LL_num_coeffs = numel(LL);
    
    % Hybrid splits LH and HL into 8x8 blocks, taking Kb per block.
    LH = dwt_res.subbands(1).LH;
    HL = dwt_res.subbands(1).HL;
    HH = dwt_res.subbands(1).HH;
    
    [H, W] = size(LH);
    numBlocks = (ceil(H/N)*N / N) * (ceil(W/N)*N / N);
    
    hybrid_target_base_LH_HL = 2 * numBlocks * Kb;
    hybrid_target_base_total = LL_num_coeffs + hybrid_target_base_LH_HL;
    
    % 1. LL: Entirely in Base Layer
    % We packetize LL identically to Hybrid to maintain structure
    sb_name = 'LL';
    [payloads, pktMeta, sb_info] = encode_subband_base_only(LL, N, Qb, 16, sb_name);
    base_pkts_data = [base_pkts_data, payloads];
    txInfo.basePktMeta = [txInfo.basePktMeta, pktMeta];
    txInfo.subbands.(sb_name) = sb_info;
    
    % 2. LH and HL: Global magnitude selection
    % Pool LH and HL
    lh_flat = LH(:);
    hl_flat = HL(:);
    pool_coeffs = [lh_flat; hl_flat];
    [~, sort_idx] = sort(abs(pool_coeffs), 'descend');
    
    num_lh_hl_base = min(hybrid_target_base_LH_HL, length(pool_coeffs));
    base_idx = sort_idx(1:num_lh_hl_base);
    
    % The remaining coefficients go to enhancement
    % To match Hybrid enhancement payload size exactly:
    % Hybrid puts Ke * numBlocks from LH, and Ke * numBlocks from HL. Total 2 * Ke * numBlocks.
    hybrid_target_enh_LH_HL = 2 * numBlocks * Ke;
    num_lh_hl_enh = min(hybrid_target_enh_LH_HL, length(pool_coeffs) - num_lh_hl_base);
    enh_idx = sort_idx(num_lh_hl_base + 1 : num_lh_hl_base + num_lh_hl_enh);
    
    base_coeffs = pool_coeffs(base_idx);
    enh_coeffs = pool_coeffs(enh_idx);
    
    % Quantize
    q_base = round(base_coeffs / Qb);
    q_base = max(min(q_base, double(intmax('int16'))), double(intmin('int16')));
    
    q_enh = round(enh_coeffs / Qe);
    q_enh = max(min(q_enh, double(intmax('int8'))), double(intmin('int8')));
    
    % Packetize base coefficients into chunks of size 16 * Kb (to match Hybrid packet sizes)
    chunk_size_base = 16 * Kb;
    num_base_pkts = ceil(length(q_base) / chunk_size_base);
    
    for p = 1:num_base_pkts
        start_idx = (p-1)*chunk_size_base + 1;
        end_idx = min(p*chunk_size_base, length(q_base));
        chunk = int16(q_base(start_idx:end_idx));
        
        base_pkts_data{end+1} = chunk;
        
        meta = struct();
        meta.subband = 'LH_HL_pool';
        meta.layerType = 'base';
        meta.startIdx = start_idx;
        meta.endIdx = end_idx;
        meta.bytesPerElem = 2;
        meta.payloadLenBytes = length(chunk) * 2;
        meta.global_indices = base_idx(start_idx:end_idx); % Shared parameters
        txInfo.basePktMeta{end+1} = meta;
    end
    
    % Packetize enh coefficients into chunks of size 16 * Ke
    chunk_size_enh = 16 * Ke;
    num_enh_pkts = ceil(length(q_enh) / chunk_size_enh);
    
    for p = 1:num_enh_pkts
        start_idx = (p-1)*chunk_size_enh + 1;
        end_idx = min(p*chunk_size_enh, length(q_enh));
        chunk = int8(q_enh(start_idx:end_idx));
        
        enh_pkts_data{end+1} = chunk;
        
        meta = struct();
        meta.subband = 'LH_HL_pool';
        meta.layerType = 'enh';
        meta.startIdx = start_idx;
        meta.endIdx = end_idx;
        meta.bytesPerElem = 1;
        meta.payloadLenBytes = length(chunk) * 1;
        meta.global_indices = enh_idx(start_idx:end_idx); % Shared parameters
        txInfo.enhPktMeta{end+1} = meta;
    end
    
    txInfo.subbands.LH_HL_pool = struct('num_base', length(q_base), 'num_enh', length(q_enh), 'origH', H, 'origW', W);
    
    % 3. HH: Entirely in Enhancement Layer (Global magnitude selection to match Ke)
    % Hybrid puts Khh * numBlocks from HH into enhancement.
    hh_flat = HH(:);
    [~, sort_idx_hh] = sort(abs(hh_flat), 'descend');
    
    hybrid_target_enh_HH = numBlocks * Khh;
    num_hh_enh = min(hybrid_target_enh_HH, length(hh_flat));
    enh_hh_idx = sort_idx_hh(1:num_hh_enh);
    
    enh_hh_coeffs = hh_flat(enh_hh_idx);
    q_enh_hh = round(enh_hh_coeffs / Qe);
    q_enh_hh = max(min(q_enh_hh, double(intmax('int8'))), double(intmin('int8')));
    
    chunk_size_enh_hh = 16 * Khh;
    num_enh_hh_pkts = ceil(length(q_enh_hh) / chunk_size_enh_hh);
    
    for p = 1:num_enh_hh_pkts
        start_idx = (p-1)*chunk_size_enh_hh + 1;
        end_idx = min(p*chunk_size_enh_hh, length(q_enh_hh));
        chunk = int8(q_enh_hh(start_idx:end_idx));
        
        enh_pkts_data{end+1} = chunk;
        
        meta = struct();
        meta.subband = 'HH_pool';
        meta.layerType = 'enh';
        meta.startIdx = start_idx;
        meta.endIdx = end_idx;
        meta.bytesPerElem = 1;
        meta.payloadLenBytes = length(chunk) * 1;
        meta.global_indices = enh_hh_idx(start_idx:end_idx); % Shared parameters
        txInfo.enhPktMeta{end+1} = meta;
    end
    
    txInfo.subbands.HH_pool = struct('num_enh', length(q_enh_hh), 'origH', H, 'origW', W);
    
    % Packetization
    txwaves = struct('base', {{}}, 'enh', {{}});
    
    txInfo.numBasePkts = length(base_pkts_data);
    txInfo.basePktLens = zeros(1, txInfo.numBasePkts);
    txInfo.cfgBase = cell(1, txInfo.numBasePkts);
    for i = 1:txInfo.numBasePkts
        pktBytes = typecast(base_pkts_data{i}, 'uint8');
        txInfo.basePktLens(i) = length(pktBytes);
        [wave, cfg] = modem(pktBytes(:), double(policy.mcs_base));
        txwaves.base{i} = wave;
        txInfo.cfgBase{i} = cfg;
    end
    
    txInfo.numEnhPkts = length(enh_pkts_data);
    txInfo.enhPktLens = zeros(1, txInfo.numEnhPkts);
    txInfo.cfgEnh = cell(1, txInfo.numEnhPkts);
    for i = 1:txInfo.numEnhPkts
        pktBytes = typecast(enh_pkts_data{i}, 'uint8');
        txInfo.enhPktLens(i) = length(pktBytes);
        [wave, cfg] = modem(pktBytes(:), double(policy.mcs_enh));
        txwaves.enh{i} = wave;
        txInfo.cfgEnh{i} = cfg;
    end
    
    basePayloadBits = sum(txInfo.basePktLens) * 8;
    enhPayloadBits = sum(txInfo.enhPktLens) * 8;
    
    fprintf('  [Tx] Target Hybrid Base Payload: %d bits\n', hybrid_target_base_total * 16);
    fprintf('  [Tx] DWT-fine Base Payload: %d bits\n', basePayloadBits);
    fprintf('  [Tx] DWT-fine Enh Payload: %d bits\n', enhPayloadBits);
    fprintf('  [Tx] Note: Selected LH/HL indices are shared via metadata and excluded from payload bits.\n');
end

function [payloads, pktMeta, sb_info] = encode_subband_base_only(subband, N, Q, targetBlocksPerPkt, sb_name)
    [H, W] = size(subband);
    
    padH = ceil(H/N)*N;
    padW = ceil(W/N)*N;
    sb_padded = zeros(padH, padW);
    sb_padded(1:H, 1:W) = subband;
    
    nBlocksH = padH / N;
    nBlocksW = padW / N;
    numBlocks = nBlocksH * nBlocksW;
    
    num_pkts = ceil(numBlocks / targetBlocksPerPkt);
    payloads = cell(1, num_pkts);
    pktMeta = cell(1, num_pkts);
    
    for p = 1:num_pkts
        startBlock = (p-1) * targetBlocksPerPkt + 1;
        endBlock = min(p * targetBlocksPerPkt, numBlocks);
        numB = endBlock - startBlock + 1;
        
        pkt_data = zeros(numB * N * N, 1, 'int16');
        idx = 1;
        
        for b = startBlock:endBlock
            r = ceil(b / nBlocksW);
            c = b - (r-1)*nBlocksW;
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            block = sb_padded(rStart:rStart+N-1, cStart:cStart+N-1);
            
            % No DCT!
            q_block = round(block / Q);
            if any(q_block(:) > double(intmax('int16'))) || any(q_block(:) < double(intmin('int16')))
                q_block = max(min(q_block, double(intmax('int16'))), double(intmin('int16')));
            end
            pkt_data(idx:idx+N*N-1) = int16(q_block(:));
            idx = idx + N*N;
        end
        
        payloads{p} = pkt_data;
        
        meta = struct();
        meta.subband = sb_name;
        meta.layerType = 'base';
        meta.startBlock = startBlock;
        meta.numBlocks = numB;
        meta.bytesPerElem = 2;
        meta.payloadLenBytes = length(pkt_data) * meta.bytesPerElem;
        pktMeta{p} = meta;
    end
    
    sb_info = struct();
    sb_info.type = 'base_direct';
    sb_info.Q = Q;
    sb_info.N = N;
    sb_info.origH = H;
    sb_info.origW = W;
    sb_info.padH = padH;
    sb_info.padW = padW;
    sb_info.nBlocksH = nBlocksH;
    sb_info.nBlocksW = nBlocksW;
    sb_info.numBlocks = numBlocks;
    sb_info.payloadBaseLen = numBlocks * N * N;
    sb_info.payloadEnhLen = 0;
end
