function ssim_val = ml_calculate_ssim(ref, test)
% ML_CALCULATE_SSIM - Simple SSIM calculation for ML training data
    ref = double(ref);
    test = double(test);
    K1 = 0.01;
    K2 = 0.03;
    L = 255;
    C1 = (K1*L)^2;
    C2 = (K2*L)^2;
    mu1 = mean(ref(:));
    mu2 = mean(test(:));
    s1 = std(ref(:));
    s2 = std(test(:));
    s12 = mean((ref(:)-mu1).*(test(:)-mu2));
    ssim_val = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1^2+mu2^2+C1)*(s1^2+s2^2+C2));
end
