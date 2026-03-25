function [blocks, padInfo] = tile_coeffs(mat, blkSize)
    [H, W] = size(mat);
    padH = ceil(H/blkSize)*blkSize - H;
    padW = ceil(W/blkSize)*blkSize - W;
    
    if padH > 0 || padW > 0
        mat = padarray(mat, [padH padW], 'symmetric', 'post');
    end

    % Break into cell array of blocks
    r = repmat(blkSize, 1, size(mat,1)/blkSize);
    c = repmat(blkSize, 1, size(mat,2)/blkSize);
    blocks = mat2cell(mat, r, c);
    blocks = blocks(:); % Flatten to column vector
    
    padInfo = [H, W]; % Store original size for reconstruction
end