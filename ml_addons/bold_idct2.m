function X = bold_idct2(Y)
% BOLD_IDCT2  2D Inverse DCT (Type II) - NO TOOLBOX
% Pure MATLAB implementation: X = D' * Y * D
%
% Bold (Unubold) - EEME30003 Major Project
    [N, M] = size(Y);
    D_rows = dct_matrix(N);
    D_cols = dct_matrix(M);
    X = D_rows' * Y * D_cols;
end

function D = dct_matrix(N)
    D = zeros(N, N);
    for k = 1:N
        for n = 1:N
            D(k, n) = cos(pi * (2*(n-1)+1) * (k-1) / (2*N));
        end
    end
    D(1,:)   = D(1,:)   * sqrt(1/N);
    D(2:N,:) = D(2:N,:) * sqrt(2/N);
end
