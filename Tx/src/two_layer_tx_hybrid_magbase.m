function [txwaves, txInfo] = two_layer_tx_hybrid_magbase(img, policy, frameId)
    level = 1; % DWT level 
    N = 8; % DCT block size
    targetBlocksPerPkt = 16;
    % Derived Ks from policy master K
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
    txInfo.N = N;
    txInfo.K = Kb;
    txInfo.Kb = Kb;
    txInfo.Ke = Ke;
    txInfo.Khh = Khh;
    txInfo.dwtLevel = level;
    txInfo.dctBlockSize = N;
    txInfo.maskType = 'magbase'; % Changed to indicate magnitude-based selection
    txInfo.K_LH_base = Kb;
    txInfo.K_HL_base = Kb;
    txInfo.frameId = frameId;
    txInfo.policy = policy;
    txInfo.subband_order_base = {}; % Explicit order for RX parsing
    txInfo.subband_order_enh = {}; 
    txInfo.subbands = struct(); 
    txInfo.basePktMeta = {};
    txInfo.enhPktMeta = {};
    
    info = struct();
    [dwt_res, dwt_info] = DWT(img, info, level); %% 1-level DWT
    txInfo.dwt_info = dwt_info;
    
    base_pkts_data = {};
    enh_pkts_data = {};
    
    % 1. LL: Entirely in Base Layer (Direct, grouped by 8x8 blocks)
    LL = dwt_res.finalLL;
    sb_name = 'LL';
    [payloads, pktMeta, sb_info] = encode_subband_base_only(LL, N, Qb, targetBlocksPerPkt, sb_name);
    base_pkts_data = [base_pkts_data, payloads];
    txInfo.basePktMeta = [txInfo.basePktMeta, pktMeta];
    txInfo.subband_order_base{end+1} = sb_name;
    txInfo.subbands.(sb_name) = sb_info;
    
    % 2. LH: Split (Magnitude-based)
    LH = dwt_res.subbands(1).LH;
    sb_name = 'LH';
    [payloads_base, pktMeta_base, payloads_enh, pktMeta_enh, sb_info] = encode_subband_split_magbase(LH, N, Kb, Ke, Qb, Qe, targetBlocksPerPkt, sb_name);
    base_pkts_data = [base_pkts_data, payloads_base];
    txInfo.basePktMeta = [txInfo.basePktMeta, pktMeta_base];
    enh_pkts_data = [enh_pkts_data, payloads_enh];
    txInfo.enhPktMeta = [txInfo.enhPktMeta, pktMeta_enh];
    txInfo.subband_order_base{end+1} = sb_name;
    txInfo.subband_order_enh{end+1} = sb_name;
    txInfo.subbands.(sb_name) = sb_info;
    
    % 3. HL: Split (Magnitude-based)
    HL = dwt_res.subbands(1).HL;
    sb_name = 'HL';
    [payloads_base, pktMeta_base, payloads_enh, pktMeta_enh, sb_info] = encode_subband_split_magbase(HL, N, Kb, Ke, Qb, Qe, targetBlocksPerPkt, sb_name);
    base_pkts_data = [base_pkts_data, payloads_base];
    txInfo.basePktMeta = [txInfo.basePktMeta, pktMeta_base];
    enh_pkts_data = [enh_pkts_data, payloads_enh];
    txInfo.enhPktMeta = [txInfo.enhPktMeta, pktMeta_enh];
    txInfo.subband_order_base{end+1} = sb_name;
    txInfo.subband_order_enh{end+1} = sb_name;
    txInfo.subbands.(sb_name) = sb_info;
    
    % 4. HH: Entirely in Enhancement Layer (DCT, Zigzag-based)
    HH = dwt_res.subbands(1).HH;
    sb_name = 'HH';
    [payloads, pktMeta, sb_info] = encode_subband_enh_only(HH, N, Khh, Qe, targetBlocksPerPkt, sb_name);
    enh_pkts_data = [enh_pkts_data, payloads];
    txInfo.enhPktMeta = [txInfo.enhPktMeta, pktMeta];
    txInfo.subband_order_enh{end+1} = sb_name;
    txInfo.subbands.(sb_name) = sb_info;
    
    txwaves = struct();
    txwaves.base = {};
    txwaves.enh = {};
    
    % Base Packetization
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
    
    % Enhancement Packetization
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
    
    fprintf('  [Tx] Base packets: %d, Enh packets: %d\n', txInfo.numBasePkts, txInfo.numEnhPkts);
    fprintf('  [Tx] Note: HL/LH selection indices are assumed shared between Tx/Rx and excluded from payload.\n');
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
            dct_blk = dct2(block);
            q_block = round(dct_blk / Q);
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

function [payloads_base, pktMeta_base, payloads_enh, pktMeta_enh, sb_info] = encode_subband_split_magbase(subband, N, Kb, Ke, Qb, Qe, targetBlocksPerPkt, sb_name)
    [H, W] = size(subband);
    
    padH = ceil(H/N)*N;
    padW = ceil(W/N)*N;
    sb_padded = zeros(padH, padW);
    sb_padded(1:H, 1:W) = subband;
    
    nBlocksH = padH / N;
    nBlocksW = padW / N;
    numBlocks = nBlocksH * nBlocksW;
    
    num_pkts = ceil(numBlocks / targetBlocksPerPkt);
    payloads_base = cell(1, num_pkts);
    pktMeta_base = cell(1, num_pkts);
    payloads_enh = cell(1, num_pkts);
    pktMeta_enh = cell(1, num_pkts);
    
    for p = 1:num_pkts
        startBlock = (p-1) * targetBlocksPerPkt + 1;
        endBlock = min(p * targetBlocksPerPkt, numBlocks);
        numB = endBlock - startBlock + 1;
        
        pkt_base_data = zeros(numB * Kb, 1, 'int16');
        pkt_enh_data = zeros(numB * Ke, 1, 'int8');
        
        base_indices = zeros(Kb, numB);
        enh_indices = zeros(Ke, numB);
        
        pb_idx = 1;
        pe_idx = 1;
        block_idx_in_pkt = 1;
        
        for b = startBlock:endBlock
            r = ceil(b / nBlocksW);
            c = b - (r-1)*nBlocksW;
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            block = sb_padded(rStart:rStart+N-1, cStart:cStart+N-1);
            
            dct_blk = dct2(block);
            
            % Magnitude-based selection
            [~, sort_idx] = sort(abs(dct_blk(:)), 'descend');
            b_idx = sort_idx(1:Kb);
            e_idx = sort_idx(Kb+1 : Kb+Ke);
            
            base_indices(:, block_idx_in_pkt) = b_idx;
            enh_indices(:, block_idx_in_pkt) = e_idx;
            
            base_coeffs = dct_blk(b_idx);
            enh_coeffs = dct_blk(e_idx);
            
            q_base = round(base_coeffs / Qb);
            q_enh = round(enh_coeffs / Qe);
            
            q_base = max(min(q_base, double(intmax('int16'))), double(intmin('int16')));
            q_enh = max(min(q_enh, double(intmax('int8'))), double(intmin('int8')));
            
            pkt_base_data(pb_idx : pb_idx+Kb-1) = int16(q_base);
            pkt_enh_data(pe_idx : pe_idx+Ke-1) = int8(q_enh);
            
            pb_idx = pb_idx + Kb;
            pe_idx = pe_idx + Ke;
            block_idx_in_pkt = block_idx_in_pkt + 1;
        end
        
        payloads_base{p} = pkt_base_data;
        meta = struct();
        meta.subband = sb_name;
        meta.layerType = 'base';
        meta.startBlock = startBlock;
        meta.numBlocks = numB;
        meta.bytesPerElem = 2;
        meta.payloadLenBytes = length(pkt_base_data) * meta.bytesPerElem;
        meta.base_indices = base_indices; % Shared parameters
        pktMeta_base{p} = meta;
        
        payloads_enh{p} = pkt_enh_data;
        meta = struct();
        meta.subband = sb_name;
        meta.layerType = 'enh';
        meta.startBlock = startBlock;
        meta.numBlocks = numB;
        meta.bytesPerElem = 1;
        meta.payloadLenBytes = length(pkt_enh_data) * meta.bytesPerElem;
        meta.enh_indices = enh_indices; % Shared parameters
        pktMeta_enh{p} = meta;
    end
    
    sb_info = struct();
    sb_info.type = 'split_magbase';
    sb_info.N = N;
    sb_info.Kb = Kb;
    sb_info.Ke = Ke;
    sb_info.Qb = Qb;
    sb_info.Qe = Qe;
    sb_info.origH = H;
    sb_info.origW = W;
    sb_info.padH = padH;
    sb_info.padW = padW;
    sb_info.nBlocksH = nBlocksH;
    sb_info.nBlocksW = nBlocksW;
    sb_info.numBlocks = numBlocks;
    sb_info.payloadBaseLen = numBlocks * Kb;
    sb_info.payloadEnhLen = numBlocks * Ke;
end

function [payloads_enh, pktMeta_enh, sb_info] = encode_subband_enh_only(subband, N, Ke, Qe, targetBlocksPerPkt, sb_name)
    [H, W] = size(subband);
    
    padH = ceil(H/N)*N;
    padW = ceil(W/N)*N;
    sb_padded = zeros(padH, padW);
    sb_padded(1:H, 1:W) = subband;
    
    nBlocksH = padH / N;
    nBlocksW = padW / N;
    numBlocks = nBlocksH * nBlocksW;
    
    zz_idx = Zigzag_index(N);
    enh_idx = zz_idx(1:Ke);
    
    num_pkts = ceil(numBlocks / targetBlocksPerPkt);
    payloads_enh = cell(1, num_pkts);
    pktMeta_enh = cell(1, num_pkts);
    
    for p = 1:num_pkts
        startBlock = (p-1) * targetBlocksPerPkt + 1;
        endBlock = min(p * targetBlocksPerPkt, numBlocks);
        numB = endBlock - startBlock + 1;
        
        pkt_enh_data = zeros(numB * Ke, 1, 'int8');
        pe_idx = 1;
        
        for b = startBlock:endBlock
            r = ceil(b / nBlocksW);
            c = b - (r-1)*nBlocksW;
            rStart = (r-1)*N + 1;
            cStart = (c-1)*N + 1;
            block = sb_padded(rStart:rStart+N-1, cStart:cStart+N-1);
            
            dct_blk = dct2(block);
            
            enh_coeffs = dct_blk(enh_idx);
            q_enh = round(enh_coeffs / Qe);
            
            q_enh = max(min(q_enh, double(intmax('int8'))), double(intmin('int8')));
            
            pkt_enh_data(pe_idx : pe_idx+Ke-1) = int8(q_enh);
            pe_idx = pe_idx + Ke;
        end
        
        payloads_enh{p} = pkt_enh_data;
        meta = struct();
        meta.subband = sb_name;
        meta.layerType = 'enh';
        meta.startBlock = startBlock;
        meta.numBlocks = numB;
        meta.bytesPerElem = 1;
        meta.payloadLenBytes = length(pkt_enh_data) * meta.bytesPerElem;
        pktMeta_enh{p} = meta;
    end
    
    sb_info = struct();
    sb_info.type = 'enh_only';
    sb_info.N = N;
    sb_info.Ke = Ke;
    sb_info.Qe = Qe;
    sb_info.origH = H;
    sb_info.origW = W;
    sb_info.padH = padH;
    sb_info.padW = padW;
    sb_info.nBlocksH = nBlocksH;
    sb_info.nBlocksW = nBlocksW;
    sb_info.numBlocks = numBlocks;
    sb_info.payloadBaseLen = 0;
    sb_info.payloadEnhLen = numBlocks * Ke;
end
