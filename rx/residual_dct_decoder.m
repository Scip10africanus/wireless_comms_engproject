function residual = residual_dct_decoder(enhBits, txInfo)
% residual_dct_decoder - decode enhancement layer bits back to a residual image
% bits -> dequantize DCT coeffs -> un-zigzag -> IDCT each block -> residual

    N = txInfo.dctInfo.block_length;
    Hp = txInfo.dctInfo.Hp;
    Wp = txInfo.dctInfo.Wp;
    Qe = txInfo.Qe;
    enhBitsPerCoeff = txInfo.enhBitsPerCoeff;
    maxDctCoeffs = txInfo.maxDctCoeffs;
    imgH = txInfo.imgHeight;
    imgW = txInfo.imgWidth;

    nBlocksH = Hp / N;
    nBlocksW = Wp / N;
    zigzagIdx = Zigzag_index(N);

    % bits -> quantized coefficients
    nEnhCoeffs = txInfo.nEnhCoeffs;
    enhCoeffs_q = bits_to_coeffs(enhBits, enhBitsPerCoeff, nEnhCoeffs);
    enhCoeffs = enhCoeffs_q * Qe;

    % reconstruct each 8x8 block
    reconDCT = zeros(Hp, Wp);
    ptr = 1;

    for bRow = 1:nBlocksH
        for bCol = 1:nBlocksW
            block = zeros(N * N, 1);

            nKeep = min(maxDctCoeffs, length(enhCoeffs) - ptr + 1);
            if nKeep > 0 && ptr <= length(enhCoeffs)
                block(1:nKeep) = enhCoeffs(ptr:ptr+nKeep-1);
            end
            ptr = ptr + maxDctCoeffs;

            % un-zigzag then IDCT
            block2d = zeros(N, N);
            block2d(zigzagIdx) = block;

            rStart = (bRow-1)*N + 1;
            cStart = (bCol-1)*N + 1;
            reconDCT(rStart:rStart+N-1, cStart:cStart+N-1) = idct2(block2d);
        end
    end

    residual = reconDCT(1:imgH, 1:imgW);
end


function coeffs = bits_to_coeffs(bits, bitsPerCoeff, nCoeffs)
% convert offset-binary bit vector back to signed integers
    bits = bits(:);
    maxVal = 2^(bitsPerCoeff - 1) - 1;
    coeffs = zeros(nCoeffs, 1);

    for k = 1:nCoeffs
        uval = 0;
        for b = 1:bitsPerCoeff
            idx = (k-1)*bitsPerCoeff + b;
            if idx <= length(bits)
                uval = uval * 2 + bits(idx);
            else
                uval = uval * 2;
            end
        end
        coeffs(k) = uval - maxVal;
    end
end
