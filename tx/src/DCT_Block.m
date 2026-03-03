function [res,info] = DCT_Block(img,block_length)
    [H,W] = size(img);
    Hp = ceil(H/block_length)*block_length;
    Wp = ceil(W/block_length)*block_length;
    padded = padarray(img, [Hp-H, Wp-W], 'symmetric', 'post');% using symmetric padding for the edge
    res = blockproc(padded, [block_length block_length], @(bs) dct2(bs.data));
    info.H = H; info.W = W;
    info.Hp = Hp; info.Wp = Wp;
    info.block_length = block_length;
end