function [baseImage, baseCoeffs] = wavelet_base_decoder(baseBits, txInfo)
% wavelet_base_decoder - reconstruct base layer image from received bits
% bits -> dequantize LL coeffs -> multi-level IDWT -> image

    waveletLevels = txInfo.waveletLevels;
    llSize = txInfo.llSize;
    Qb = txInfo.Qb;
    baseBitsPerCoeff = txInfo.baseBitsPerCoeff;
    imgH = txInfo.imgHeight;
    imgW = txInfo.imgWidth;

    % convert bits back to quantized LL coefficients
    nBaseCoeffs = txInfo.nBaseCoeffs;
    baseCoeffs_q = bits_to_coeffs(baseBits, baseBitsPerCoeff, nBaseCoeffs);

    % dequantize
    baseCoeffs = baseCoeffs_q * Qb;

    % reshape vector back into LL subband matrix
    LL = reshape(baseCoeffs, llSize);

    % inverse DWT level by level (innermost first)
    baseRecon = LL;
    for level = waveletLevels:-1:1
        sz = txInfo.subbandSizes(level).inputSize;
        zeroSub = zeros(txInfo.subbandSizes(level).llSize);
        baseRecon = idwt2_simple(baseRecon, zeroSub, zeroSub, zeroSub);
        baseRecon = baseRecon(1:sz(1), 1:sz(2));
    end

    baseRecon = baseRecon(1:imgH, 1:imgW);
    baseImage = uint8(max(0, min(255, round(baseRecon))));
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
