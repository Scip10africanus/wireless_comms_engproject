function [reconBlock, rxInfo] = raw_rx(rxWave, txInfo)
    rxInfo = struct();
    rxInfo.blockOK = false;
    rxInfo.failReason = '';
    
    BLK_SIZE = txInfo.blk_size;
    
    try
        % Demodulate waveform to get bytes
        [rxBytes, ok, reason] = demodem(rxWave, txInfo.cfg);
        
        if ok
            % Reshape to 2D block
            blk = reshape(rxBytes, BLK_SIZE, BLK_SIZE);
            reconBlock = double(blk);
            rxInfo.blockOK = true;
        else
            % If demodem fails internally but doesn't throw, fill block with zeros
            reconBlock = zeros(BLK_SIZE, BLK_SIZE);
            rxInfo.failReason = reason;
        end
    catch ME
        % If an actual error is thrown, catch it and fill with zeros
        reconBlock = zeros(BLK_SIZE, BLK_SIZE);
        rxInfo.failReason = ME.message;
    end
end
