function [LL, LH, HL, HH] = dwt2_simple(img)
% dwt2_simple - one-level 2D Haar wavelet decomposition (no toolbox needed)
% returns LL (approx), LH, HL, HH (detail subbands)

    img = double(img);
    [H, W] = size(img);

    % Pad to even dimensions if needed
    if mod(H, 2) ~= 0
        img(H+1, :) = img(H, :);
        H = H + 1;
    end
    if mod(W, 2) ~= 0
        img(:, W+1) = img(:, W);
        W = W + 1;
    end

    % Haar wavelet: split into 2x2 blocks
    % a = img(2i-1, 2j-1), b = img(2i-1, 2j), c = img(2i, 2j-1), d = img(2i, 2j)
    a = img(1:2:H, 1:2:W);
    b = img(1:2:H, 2:2:W);
    c = img(2:2:H, 1:2:W);
    d = img(2:2:H, 2:2:W);

    % Haar transform
    LL = (a + b + c + d) / 2;
    LH = (a - b + c - d) / 2;
    HL = (a + b - c - d) / 2;
    HH = (a - b - c + d) / 2;
end
