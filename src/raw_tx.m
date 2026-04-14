function [txWave, txInfo] = raw_tx(block, policy, frameId, blockId)
    txInfo = struct();
    % Ensure block is 0-255 uint8 format for raw byte transmission
    if isfloat(block) && max(block(:)) <= 1.0
        block_uint8 = uint8(block * 255.0);
    else
        block_uint8 = uint8(block);
    end
    
    % Flatten block to 1D array 
    blkBytes = block_uint8(:);
    
    % Modulate using existing OFDM/WLAN chain
    [txWave, cfg] = modem(blkBytes, double(policy.mcs_base));
    
    % Populate txInfo
    txInfo.cfg = cfg;
    txInfo.fs = cfg.params.fs;
    txInfo.frameId = frameId;
    txInfo.blockId = blockId;
    txInfo.policy = policy;
    txInfo.blk_size = size(block, 1); % Assuming square block
end
