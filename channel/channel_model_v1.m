function [rxSignal, channelInfo] = channel_model_v1(txSignal, param)

    %% Default parameters

    if ~isfield(param, 'channelType')
        param.channelType = 'RICIAN';
    end
    if ~isfield(param, 'SNR_dB')
        param.SNR_dB = 20;
    end
    if ~isfield(param, 'ricianK')
        param.ricianK = 3; 
    end

    channelInfo = struct();
    channelInfo.type = param.channelType;

    %% Apply channel based on type
    switch upper(param.channelType)
        case 'RICIAN'
            [rxSignal, channelInfo] = apply_rician(txSignal, param, channelInfo);

        otherwise
            warning('Unknown channel type. Using RICIAN.');
            [rxSignal, channelInfo] = apply_rician(txSignal, param, channelInfo);
    end


    %% Calculate channel statistics
    channelInfo.txPower = mean(abs(txSignal).^2);
    channelInfo.rxPower = mean(abs(rxSignal).^2);
%    channelInfo.measuredSNR = 10*log10(channelInfo.rxPower / channelInfo.noisePower);
end

%% Channel Implementations

function [rxSignal, info] = apply_rician(txSignal, param, info)
% APPLY_RICIAN - Rician fading + AWGN
%
% Models a channel with line-of-sight (LOS) plus scattered components
    fs = 20e6;                      % Sample rate
    pathDelays = [0 1.2e-7 2.4e-7]; % 0ns, 120ns, 240ns
    pathGains  = [0 -3 -6];         % dB
    K_dB = param.ricianK;
    K = 10^(K_dB / 10);

    % fix the seed of random phase, the phase should be constance during
    % each frame
    if isfield(param, 'seed')
        currRngState = rng(); % Store current random phase
        rng(param.seed);      
    end

    pathGains_d = 10.^(pathGains / 20);
    pathGains_d = pathGains_d / sqrt(sum(pathGains_d.^2)); % Normalization
    
    nSamples = length(txSignal);
    scatteredWave = zeros(size(txSignal));
    
    for i = 1:length(pathGains_d)
        nDelay = round(pathDelays(i) * fs); 
        if nDelay < nSamples
            delayedSignal = [zeros(nDelay, 1); txSignal(1:end-nDelay)];
            % For same frame Bolock they have same phase 
            randomPhase = exp(1j * 2 * pi * rand());
            scatteredWave = scatteredWave + delayedSignal * pathGains_d(i) * randomPhase;
        end
    end

    Los_gain     = txSignal * sqrt(K/(K+1));
    scatter_gain = scatteredWave * sqrt(1/(K+1));
    
    fadedWave = Los_gain + scatter_gain;

    if isfield(param, 'seed')
        rng(currRngState); 
        % Back to random state to aviod infulence different block get same noise
    end
    
    % Add noise 
    rxSignal = awgn(fadedWave, param.SNR_dB, 'measured');
    
    info.K_dB = K_dB;
    info.faded_power = mean(abs(fadedWave).^2);
end






function params = channel_preset_highway()
% CHANNEL_PRESET_HIGHWAY - Highway/vehicular
    params.channelType = 'Rayleigh';
    params.dopplerHz = 200;  % High speed
end
