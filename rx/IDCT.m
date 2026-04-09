function recon = IDCT(res, info)
    block_length = info.block_length;
    recon_p = blockproc(res, [block_length block_length], @(bs) idct2(bs.data));
    recon = recon_p(1:info.H, 1:info.W);
end