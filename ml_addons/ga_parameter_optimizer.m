function [optParams, gaMetrics] = ga_parameter_optimizer(chCond, curParams, gaSettings)
% GA_PARAMETER_OPTIMIZER  GA to find best [K, Qb_idx, Qe_idx]
% Evolves layer-split and quantization params for current channel conditions
% using tournament selection, crossover, and mutation.
% MCS is not in the search space; rules + LSTM ceiling handle MCS in
% rx_policy_update.m. The GA focuses on parameters where the analytic
% model is sufficient and rules are coarse.

    if nargin < 3, gaSettings = struct(); end
    if ~isfield(gaSettings, 'popSize'),      gaSettings.popSize = 40; end
    if ~isfield(gaSettings, 'generations'),  gaSettings.generations = 15; end
    if ~isfield(gaSettings, 'mutationRate'), gaSettings.mutationRate = 0.15; end
    if ~isfield(gaSettings, 'eliteCount'),   gaSettings.eliteCount = 4; end
    if ~isfield(gaSettings, 'verbose'),      gaSettings.verbose = false; end

    if ~isfield(chCond, 'snr_dB'), chCond.snr_dB = 20; end
    if ~isfield(curParams, 'K'),       curParams.K = 20; end
    if ~isfield(curParams, 'Qb_idx'),  curParams.Qb_idx = 0; end
    if ~isfield(curParams, 'Qe_idx'),  curParams.Qe_idx = 0; end
    if ~isfield(curParams, 'mcs_base'),curParams.mcs_base = 2; end
    if ~isfield(curParams, 'mcs_enh'), curParams.mcs_enh = 6; end

    %% chromosome: [K, Qb_idx, Qe_idx]   (MCS not in search space)
    nGenes = 3;
    gMin = [5  0 0];
    gMax = [60 3 3];   % DCT Qb/Qe lists have 4 entries (indices 0-3)

    popSize = gaSettings.popSize;
    nGen = gaSettings.generations;

    %% init population (seed with current params + random perturbations)
    pop = zeros(popSize, nGenes);
    curChr = clamp([double(curParams.K) double(curParams.Qb_idx) double(curParams.Qe_idx)], gMin, gMax);
    pop(1,:) = curChr; % elitism seed
    for i = 2:popSize
        perturb = round(randn(1,nGenes) .* [5 1 1]);
        pop(i,:) = clamp(curChr + perturb, gMin, gMax);
    end

    snr = chCond.snr_dB;

    prevQuality = struct('ssim', NaN, 'enhFailRatio', NaN);
    if isfield(chCond, 'prevSSIM') && ~isnan(chCond.prevSSIM)
        prevQuality.ssim = chCond.prevSSIM;
    end
    if isfield(chCond, 'prevEnhFailRatio') && ~isnan(chCond.prevEnhFailRatio)
        prevQuality.enhFailRatio = chCond.prevEnhFailRatio;
    end

    fitHist = zeros(nGen,1);

    %% evolve
    for gen = 1:nGen
        fit = zeros(popSize,1);
        for i = 1:popSize
            fit(i) = calc_fitness(pop(i,:), snr, prevQuality);
        end

        [bestFit, bestIdx] = max(fit);
        fitHist(gen) = bestFit;

        if gaSettings.verbose
            fprintf('  GA gen %d: fitness=%.3f K=%d Qb=%d Qe=%d\n', ...
                gen, bestFit, pop(bestIdx,1), pop(bestIdx,2), pop(bestIdx,3));
        end

        pop = evolve(pop, fit, gaSettings, gMin, gMax);
    end

    %% extract best
    fit = zeros(popSize,1);
    for i = 1:popSize
        fit(i) = calc_fitness(pop(i,:), snr, prevQuality);
    end
    [~, bestIdx] = max(fit);
    best = pop(bestIdx,:);

    optParams = struct();
    optParams.K        = uint8(best(1));
    optParams.Qb_idx   = uint8(best(2));
    optParams.Qe_idx   = uint8(best(3));
    optParams.mcs_base = uint8(curParams.mcs_base);   % pass-through, not evolved
    optParams.mcs_enh  = uint8(curParams.mcs_enh);    % pass-through, not evolved

    gaMetrics = struct();
    gaMetrics.fitnessHistory = fitHist;
    gaMetrics.finalFitness = fitHist(end);
    gaMetrics.generations = nGen;
    gaMetrics.bestChromosome = best;
end


%% ---- fitness ----

function f = calc_fitness(chr, snr_dB, prevQuality)
% Fitness for [K, Qb_idx, Qe_idx].
% Scores the K split ratio, quantization distortion vs compression
% tradeoff, and incorporates measured quality from the previous frame
% to close the feedback loop.
    if nargin < 3, prevQuality = struct('ssim', NaN, 'enhFailRatio', NaN); end
    K = chr(1); QbIdx = chr(2); QeIdx = chr(3);
    kRatio = K / 64;

    % Quantization distortion model (normalised 0-1, higher index = more noise)
    qDistBase = (QbIdx + 1) / 4;
    qDistEnh  = (QeIdx + 1) / 4;
    qScore = 1.0 - 0.5*(kRatio * qDistBase + (1 - kRatio) * qDistEnh);

    % Compression reward: coarser quant = fewer bits
    compScore = 0.3 * (QbIdx + QeIdx) / 6;

    % K penalties
    kPen = 0;
    if K < 10,  kPen = 0.1*(10 - K); end
    if K > 50,  kPen = 0.05*(K - 50); end

    % SNR-based base-layer reliability proxy
    RICIAN_PENALTY_dB = 4.0;
    effSnr = snr_dB - RICIAN_PENALTY_dB;
    if effSnr < 10
        baseReliability = 0.5;
    elseif effSnr < 20
        baseReliability = 0.5 + 0.5 * (effSnr - 10) / 10;
    else
        baseReliability = 1.0;
    end
    baseProt = 0.7 * kRatio * baseReliability + 0.3 * (1 - kRatio) * baseReliability;

    % SNR-aware weighting: at low SNR quality matters more, at high SNR compression OK
    snrWeight = min(1, max(0, (snr_dB - 10) / 30));
    f = (1 - snrWeight) * 2.0 * qScore ...
      +      snrWeight  * 1.0 * compScore ...
      + 0.5 * baseProt ...
      - kPen;

    % Measured quality correction (closes the loop)
    if isfield(prevQuality, 'ssim') && ~isnan(prevQuality.ssim)
        f = f + 0.5 * (prevQuality.ssim - 0.85);
        if prevQuality.ssim < 0.7
            f = f + 0.3 * (K - 20) / 40;
        end
    end

    if isfield(prevQuality, 'enhFailRatio') && ~isnan(prevQuality.enhFailRatio)
        if prevQuality.enhFailRatio > 0.5
            f = f - 0.3 * (1 - kRatio);
        end
    end
end


%% ---- GA operators ----

function newPop = evolve(pop, fit, settings, gMin, gMax)
    [popSize, nGenes] = size(pop);
    newPop = zeros(popSize, nGenes);

    % elitism
    [~, si] = sort(fit, 'descend');
    nElite = settings.eliteCount;
    newPop(1:nElite,:) = pop(si(1:nElite),:);

    % fill rest with crossover + mutation
    for i = (nElite+1):popSize
        c1 = randperm(popSize, min(3,popSize));
        [~,w1] = max(fit(c1));  p1 = pop(c1(w1),:);
        c2 = randperm(popSize, min(3,popSize));
        [~,w2] = max(fit(c2));  p2 = pop(c2(w2),:);

        % uniform crossover
        mask = rand(1,nGenes) > 0.5;
        child = p1;  child(mask) = p2(mask);

        % mutation
        for g = 1:nGenes
            if rand < settings.mutationRate
                child(g) = child(g) + round(randn * (gMax(g)-gMin(g)) * 0.2);
            end
        end
        newPop(i,:) = clamp(child, gMin, gMax);
    end
end

function c = clamp(c, gMin, gMax)
    c = round(max(gMin, min(gMax, c)));
end
