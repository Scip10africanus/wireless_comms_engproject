function img = idwt2_simple(LL, LH, HL, HH)
% idwt2_simple - inverse of dwt2_simple, reconstructs image from subbands

    [h, w] = size(LL);
    img = zeros(2*h, 2*w);

    % Inverse Haar: recover a, b, c, d from LL, LH, HL, HH
    a = (LL + LH + HL + HH) / 2;
    b = (LL - LH + HL - HH) / 2;
    c = (LL + LH - HL - HH) / 2;
    d = (LL - LH - HL + HH) / 2;

    img(1:2:2*h, 1:2:2*w) = a;
    img(1:2:2*h, 2:2:2*w) = b;
    img(2:2:2*h, 1:2:2*w) = c;
    img(2:2:2*h, 2:2:2*w) = d;
end
