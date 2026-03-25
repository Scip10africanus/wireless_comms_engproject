function A = idct2(D)
% IDCT2 - 2-D inverse discrete cosine transform (Octave compatibility shim)
%
% A = idct2(D) returns the two-dimensional inverse DCT of matrix D.
% Computed as idct(idct(D).').', i.e. IDCT along columns then rows.

    [M, N] = size(D);
    D = double(D);

    % IDCT along columns
    A = idct_1d(D, M);

    % IDCT along rows
    A = idct_1d(A.', N).';
end

function X = idct_1d(Y, N)
% Apply 1-D inverse DCT to each column of Y
    X = zeros(size(Y));
    for n = 0:N-1
        for k = 0:N-1
            if k == 0
                alpha = sqrt(1 / N);
            else
                alpha = sqrt(2 / N);
            end
            X(n+1, :) = X(n+1, :) + alpha * Y(k+1, :) * cos(pi * (2*n + 1) * k / (2 * N));
        end
    end
end
