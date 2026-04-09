function [r, raps] = radial_average_power_spectrum(P)
    % RADIAL_AVERAGE_POWER_SPECTRUM Computes the 1D radially averaged power spectrum
    [rows, cols] = size(P);
    [X, Y] = meshgrid(-floor(cols/2):ceil(cols/2)-1, -floor(rows/2):ceil(rows/2)-1);
    R = round(sqrt(X.^2 + Y.^2));
    maxR = max(R(:));
    
    raps = zeros(maxR+1, 1);
    counts = zeros(maxR+1, 1);
    
    for i = 1:numel(R)
        idx = R(i) + 1;
        raps(idx) = raps(idx) + P(i);
        counts(idx) = counts(idx) + 1;
    end
    
    valid = counts > 0;
    raps(valid) = raps(valid) ./ counts(valid);
    r = (0:maxR)';
end
