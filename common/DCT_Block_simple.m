function [res, info] = DCT_Block_simple(img, block_length)
% DCT_BLOCK_SIMPLE  Octave-compatible DCT block processing
% Same as DCT_Block but without padarray/blockproc
    [H, W] = size(img);
    Hp = ceil(H/block_length)*block_length;
    Wp = ceil(W/block_length)*block_length;

    % symmetric padding (manual, no padarray)
    padded = zeros(Hp, Wp);
    padded(1:H, 1:W) = img;
    if Hp > H
        for r = H+1:Hp
            mr = max(1, 2*H - r);
            padded(r, 1:W) = padded(mr, 1:W);
        end
    end
    if Wp > W
        for c = W+1:Wp
            mc = max(1, 2*W - c);
            padded(:,c) = padded(:,mc);
        end
    end

    % block DCT
    res = zeros(Hp, Wp);
    for br = 1:block_length:Hp
        for bc = 1:block_length:Wp
            block = padded(br:br+block_length-1, bc:bc+block_length-1);
            res(br:br+block_length-1, bc:bc+block_length-1) = dct2(block);
        end
    end

    info.H = H;  info.W = W;
    info.Hp = Hp; info.Wp = Wp;
    info.block_length = block_length;
end
