function [optParams, gaMetrics] = ga_parameter_optimizer(chCond, curParams, gaSettings)
% GA_PARAMETER_OPTIMIZER  GA to find best [K, mcs_base, mcs_enh]
% Qb_idx and Qe_idx are fixed at 1 (not optimised)
% Evolves TX params for current channel conditions using tournament
% selection, crossover, and mutation. Same pattern as ga_adaptive_demapper.

    if nargin < 3, gaSettings = struct(); end
    if ~isfield(gaSettings, 'popSize'),      gaSettings.popSize = 40; end
    if ~isfield(gaSettings, 'generations'),  gaSettings.generations = 15; end
    if ~isfield(gaSettings, 'mutationRate'), gaSettings.mutationRate = 0.15; end
    if ~isfield(gaSettings, 'eliteCount'),   gaSettings.eliteCount = 4; end
    if ~isfield(gaSettings, 'verbose'),      gaSettings.verbose = false; end

    if ~isfield(chCond, 'snr_dB'),     chCond.snr_dB = 20; end
    if ~isfield(curParams, 'K'),       curParams.K = 20; end
    if ~isfield(curParams, 'mcs_base'),curParams.mcs_base = 2; end
    if ~isfield(curParams, 'mcs_enh'), curParams.mcs_enh = 6; end

    % Fixed quantisation values — not optimised
    FIXED_QB = 1;
    FIXED_QE = 1;

    %% chromosome: [K, mcs_base, mcs_enh]  (Qb/Qe fixed — not in search space)
    nGenes = 3;
    gMin = [5  0 0];
    gMax = [60 7 7];

    popSize = gaSettings.popSize;
    nGen = gaSettings.generations;

    %% init population (seed with current params + random perturbations)
    pop = zeros(popSize, nGenes);
    curChr = [curParams.K curParams.mcs_base curParams.mcs_enh];
    pop(1,:) = curChr;  % elitism seed
    for i = 2:popSize
        perturb = round(randn(1,nGenes) .* [5 1 1]);
        pop(i,:) = clamp(curChr + perturb, gMin, gMax);
    end

    snr = chCond.snr_dB;
    fitHist = zeros(nGen,1);

    %% evolve
    for gen = 1:nGen
        fit = zeros(popSize,1);
        for i = 1:popSize
            fit(i) = calc_fitness(pop(i,:), snr);
        end

        [bestFit, bestIdx] = max(fit);
        fitHist(gen) = bestFit;

        if gaSettings.verbose
            fprintf('  GA gen %d: fitness=%.3f K=%d mcsB=%d mcsE=%d\n', ...
                gen, bestFit, pop(bestIdx,1), pop(bestIdx,4), pop(bestIdx,5));
        end

        % next generation
        pop = evolve(pop, fit, gaSettings, gMin, gMax);
    end

    %% extract best
    fit = zeros(popSize,1);
    for i = 1:popSize
        fit(i) = calc_fitness(pop(i,:), snr);
    end
    [~, bestIdx] = max(fit);
    best = pop(bestIdx,:);

    optParams = struct();
    optParams.K        = uint8(best(1));
    optParams.Qb_idx   = uint8(FIXED_QB);  % fixed — not optimised
    optParams.Qe_idx   = uint8(FIXED_QE);  % fixed — not optimised
    optParams.mcs_base = uint8(best(2));
    optParams.mcs_enh  = uint8(best(3));

    gaMetrics = struct();
    gaMetrics.fitnessHistory = fitHist;
    gaMetrics.finalFitness = fitHist(end);
    gaMetrics.generations = nGen;
    gaMetrics.bestChromosome = best;
end


%% ---- fitness ----

function f = calc_fitness(chr, snr_dB)
% analytical model: quality + throughput - penalties
% chr = [K, mcs_base, mcs_enh]  (Qb/Qe fixed at 1)
    K = chr(1); mcsB = chr(2); mcsE = chr(3);
    Qb = 1; Qe = 1;  % fixed — not optimised

    % MCS tables - must match ofdm_mod MCS_MOD: 0,1,2->QPSK | 3,4,5->16-QAM | 6,7->64-QAM
    modOrder = [4 4 4 16 16 16 64 64];
    codeRate = [0.5 0.75 1.0 0.5 0.75 1.0 0.67 0.75];

    bMod = modOrder(mcsB+1);  bRate = codeRate(mcsB+1);
    eMod = modOrder(mcsE+1);  eRate = codeRate(mcsE+1);

    % BER estimates
    bBER = approx_ber(snr_dB, bMod, bRate);
    eBER = approx_ber(snr_dB, eMod, eRate);

    % quality: base layer is critical — losing it ruins the whole image,
    % so weight it heavily regardless of K (previously kRatio≈0.3 was backwards)
    bQ = max(0, 1 - 10*bBER);
    eQ = max(0, 1 - 5*eBER);
    kRatio = K/64;  % still used for throughput score below
    qualScore = 0.7*bQ + 0.3*eQ;

    % throughput
    bTh = log2(bMod)*bRate;
    eTh = log2(eMod)*eRate;
    thScore = (kRatio*bTh + (1-kRatio)*eTh) / 6.0;

    % penalties
    qPen = 0.01*(Qb + Qe);
    kPen = 0;
    if K < 10,  kPen = 0.1*(10-K); end
    if K > 50,  kPen = 0.05*(K-50); end

    failPen = 0;
    if bBER > 0.1,     failPen = 2.0;
    elseif bBER > 0.05, failPen = 0.5;
    end

    f = 2.0*qualScore + 1.0*thScore - qPen - kPen - failPen;
end

function ber = approx_ber(snr_dB, M, cRate)
% rough M-QAM BER with coding gain
    snrLin = 10^(snr_dB/10);
    k = log2(M);
    if M == 2
        berU = 0.5 * erfc(sqrt(snrLin));
    else
        berU = (4/k)*(1-1/sqrt(M))*0.5*erfc(sqrt(3*k*snrLin/(2*(M-1))));
    end
    ber = max(0, min(0.5, berU / (1/(1-cRate+0.1))));
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
        % tournament selection (pick 3, best wins)
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
