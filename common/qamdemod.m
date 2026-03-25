function output = qamdemod(symbols, M, varargin)
% qamdemod - QAM demodulation, matches MATLAB interface for Octave compat
% finds nearest constellation point and returns index or bits

    % Parse options
    outputType = 'integer';
    unitAvgPower = false;
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'outputtype'
                outputType = lower(varargin{i+1});
            case 'unitaveragepower'
                unitAvgPower = varargin{i+1};
        end
    end

    bitsPerSymbol = log2(M);
    sqrtM = sqrt(M);
    symbols = symbols(:);

    % If unit average power, scale back to standard constellation
    if unitAvgPower
        % Recreate reference constellation to find scaling factor
        refI = (0:sqrtM-1) - (sqrtM - 1) / 2;
        refQ = (0:sqrtM-1) - (sqrtM - 1) / 2;
        [RI, RQ] = meshgrid(2*refI, 2*refQ);
        refSyms = RI(:) + 1j*RQ(:);
        scale = sqrt(mean(abs(refSyms).^2));
        symbols = symbols * scale;
    end

    % Find nearest constellation point
    % Modulator maps: symbol = 2*index - (sqrtM-1)
    % Inverse:        index = (symbol + sqrtM - 1) / 2
    I_dec = round((real(symbols) + sqrtM - 1) / 2);
    Q_dec = round((imag(symbols) + sqrtM - 1) / 2);

    % Clip to valid range
    I_dec = max(0, min(sqrtM - 1, I_dec));
    Q_dec = max(0, min(sqrtM - 1, Q_dec));

    indices = I_dec + sqrtM * Q_dec;

    if strcmp(outputType, 'bit')
        % Convert indices to bits
        nSymbols = length(indices);
        output = zeros(nSymbols * bitsPerSymbol, 1);
        for k = 1:nSymbols
            val = indices(k);
            for b = bitsPerSymbol:-1:1
                output((k-1)*bitsPerSymbol + b) = mod(val, 2);
                val = floor(val / 2);
            end
        end
    else
        output = indices;
    end
end
