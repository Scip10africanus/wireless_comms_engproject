function Y = bold_dct2(X)
% BOLD_DCT2  2D Discrete Cosine Transform (Type II) - NO TOOLBOX
% Pure MATLAB implementation using the DCT matrix: Y = D * X * D'
%
% Bold (Unubold) - EEME30003 Major Project
    [N, M] = size(X);
    D_rows = dct_matrix(N);
    D_cols = dct_matrix(M);
    Y = D_rows * X * D_cols';
end

function D = dct_matrix(N)
% Build the NxN DCT-II matrix
%   D(k,n) = alpha(k) * cos( pi*(2*n-1)*(k-1) / (2*N) )
%   alpha(1) = sqrt(1/N),  alpha(k>1) = sqrt(2/N)
    D = zeros(N, N);
    for k = 1:N
        for n = 1:N
            D(k, n) = cos(pi * (2*(n-1)+1) * (k-1) / (2*N));
        end
    end
    D(1,:)   = D(1,:)   * sqrt(1/N);
    D(2:N,:) = D(2:N,:) * sqrt(2/N);
end
