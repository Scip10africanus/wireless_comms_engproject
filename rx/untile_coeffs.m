function mat = untile_coeffs(blocks, padInfo, blkSize) % untile blocks
    H = padInfo(1);
    W = padInfo(2);
    padH = ceil(H/blkSize)*blkSize;
    padW = ceil(W/blkSize)*blkSize;
    
    % Reconstruct padded matrix
    rows = padH / blkSize;
    cols = padW / blkSize;
    
    % Reshape blocks into grid
    blockGrid = reshape(blocks, rows, cols);
    mat = cell2mat(blockGrid);
    
    % Crop padding
    mat = mat(1:H, 1:W);
end