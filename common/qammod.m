function symbols = qammod(data, M, varargin)
% qammod - QAM modulation, written to match MATLAB's interface for Octave compat
% supports InputType='bit' and UnitAveragePower=true

    % Parse options
    inputType = 'integer';
    unitAvgPower = false;
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'inputtype'
                inputType = lower(varargin{i+1});
            case 'unitaveragepower'
                unitAvgPower = varargin{i+1};
        end
    end

    bitsPerSymbol = log2(M);
    sqrtM = sqrt(M);

    if strcmp(inputType, 'bit')
        % Convert bits to integer indices
        nSymbols = length(data) / bitsPerSymbol;
        data = reshape(data, bitsPerSymbol, nSymbols);
        indices = zeros(nSymbols, 1);
        for k = 1:nSymbols
            for b = 1:bitsPerSymbol
                indices(k) = indices(k) * 2 + data(b, k);
            end
        end
    else
        indices = data(:);
    end

    % Gray-coded QAM constellation
    I = mod(indices, sqrtM) - (sqrtM - 1) / 2;
    Q = floor(indices / sqrtM) - (sqrtM - 1) / 2;
    symbols = (2*I) + 1j*(2*Q);

    % Normalize to unit average power if requested
    if unitAvgPower
        % Use average power of full constellation (same scale for all symbols)
        allI = (0:sqrtM-1) - (sqrtM - 1) / 2;
        allQ = (0:sqrtM-1) - (sqrtM - 1) / 2;
        [AI, AQ] = meshgrid(2*allI, 2*allQ);
        allSyms = AI(:) + 1j*AQ(:);
        avgPower = mean(abs(allSyms).^2);
        symbols = symbols / sqrt(avgPower);
    end
end
