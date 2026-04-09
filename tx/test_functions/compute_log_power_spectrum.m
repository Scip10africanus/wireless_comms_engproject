function [LP, P] = compute_log_power_spectrum(I)
    % COMPUTE_LOG_POWER_SPECTRUM Computes the 2D log power spectrum of an image
    I0 = double(I);
    I0 = I0 - mean(I0(:));
    F = fftshift(fft2(I0));
    P = abs(F).^2;
    LP = log1p(P);
end
