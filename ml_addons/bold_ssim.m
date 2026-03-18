function val = bold_ssim(ref, test)
% BOLD_SSIM  Toolbox-free SSIM for double [0,1] or uint8 images
    ref  = double(ref);
    test = double(test);
    % Auto-detect range
    if max(ref(:)) > 1.5
        L = 255;
    else
        L = 1.0;
    end
    K1 = 0.01; K2 = 0.03;
    C1 = (K1*L)^2; C2 = (K2*L)^2;
    mu1 = mean(ref(:));   mu2 = mean(test(:));
    s1  = std(ref(:));    s2  = std(test(:));
    s12 = mean((ref(:)-mu1).*(test(:)-mu2));
    val = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1^2+s2^2+C2));
    val = max(0, min(1, val));
end
