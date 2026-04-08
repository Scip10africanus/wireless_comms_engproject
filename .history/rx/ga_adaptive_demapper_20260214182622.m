function [bits, optimizedBoundaries, gaMetrics] = ga_adaptive_demapper(rxSymbols, modOrder, pilotSymbols, pilotBits, params)
% GA_ADAPTIVE_DEMAPPER - Genetic Algorithm optimized symbol demapping
%
% Inputs:
%   rxSymbols    - Received symbols to demap (complex vector)
%   modOrder     - Modulation order (4, 16, 64)
%   pilotSymbols - Known pilot symbols for training (complex vector)
%   pilotBits    - Known pilot bits (for fitness evaluation)
%   params       - GA parameters (popSize, generations, etc.)
%
% Outputs:
%   bits                - Demapped bits
%   optimizedBoundaries - Best boundaries found by GA
%   gaMetrics          - GA performance (fitness history, etc.)

%% Default parameters
if ~isfield(params, 'popSize')
    params.popSize = 50;  % Population size
end
if ~isfield(params, 'generations')
    params.generations = 20;  % Number of GA iterations
end
if ~isfield(params, 'mutationRate')
    params.mutationRate = 0.1;  % Mutation probability
end
if ~isfield(params, 'eliteCount')
    params.eliteCount = 5;  % Keep top N solutions
end

%% Step 1: Initialize population of decision boundaries
nBoundaries = get_num_boundaries(modOrder);
population = initialize_population(params.popSize, nBoundaries, modOrder);

%% Step 2: Evolve population using GA
fitnessHistory = zeros(params.generations, 1);

for gen = 1:params.generations
    % Evaluate fitness of each chromosome
    fitness = zeros(params.popSize, 1);
    for i = 1:params.popSize
        fitness(i) = evaluate_fitness(population(i, :), pilotSymbols, pilotBits, modOrder);
    end
    
    % Track best fitness
    [bestFitness, bestIdx] = min(fitness);  % Lower BER is better
    fitnessHistory(gen) = bestFitness;
    
    if params.verbose
        fprintf('  Gen %d/%d: Best BER = %.4e\n', gen, params.generations, bestFitness);
    end
    
    % Create next generation
    population = evolve_population(population, fitness, params);
end

%% Step 3: Use best solution to demap all symbols
[~, bestIdx] = min(fitness);
optimizedBoundaries = population(bestIdx, :);
bits = demap_with_boundaries(rxSymbols, optimizedBoundaries, modOrder);

%% Step 4: Calculate metrics
gaMetrics = struct();
gaMetrics.fitnessHistory = fitnessHistory;
gaMetrics.finalBER = fitnessHistory(end);
gaMetrics.improvementPercent = (fitnessHistory(1) - fitnessHistory(end)) / fitnessHistory(1) * 100;
gaMetrics.optimizedBoundaries = optimizedBoundaries;
gaMetrics.generations = params.generations;

end

%% ==================== GA FUNCTIONS ====================

function nBounds = get_num_boundaries(modOrder)
    % Calculate number of decision boundaries needed
    switch modOrder
        case 4  % QPSK: 2 boundaries (I and Q)
            nBounds = 2;
        case 16  % 16-QAM: 6 boundaries (3 in I, 3 in Q)
            nBounds = 6;
        case 64  % 64-QAM: 14 boundaries (7 in I, 7 in Q)
            nBounds = 14;
        otherwise
            error('Unsupported modulation order');
    end
end

function pop = initialize_population(popSize, nBounds, modOrder)
    % Initialize population with random boundaries around ideal positions
    pop = zeros(popSize, nBounds);
    
    % Get ideal boundaries for this modulation
    idealBounds = get_ideal_boundaries(modOrder);
    
    % Add random variation (±20% of spacing)
    for i = 1:popSize
        pop(i, :) = idealBounds + 0.4 * randn(1, nBounds);  % Random offset
    end
end

function idealBounds = get_ideal_boundaries(modOrder)
    % Standard QAM decision boundaries
    switch modOrder
        case 4  % QPSK
            idealBounds = [0, 0];  % I=0, Q=0
            
        case 16  % 16-QAM (constellation: -3, -1, +1, +3)
            idealBounds = [-2, 0, 2, -2, 0, 2];  % 3 boundaries in I, 3 in Q
            
        case 64  % 64-QAM (constellation: -7, -5, -3, -1, +1, +3, +5, +7)
            idealBounds = [-6, -4, -2, 0, 2, 4, 6, -6, -4, -2, 0, 2, 4, 6];
            
        otherwise
            error('Unsupported modulation');
    end
end

function fitness = evaluate_fitness(boundaries, pilotSymbols, pilotBits, modOrder)
    % FITNESS FUNCTION: Lower is better
    % Demap pilot symbols using these boundaries
    demappedBits = demap_with_boundaries(pilotSymbols, boundaries, modOrder);
    
    % Calculate BER
    minLen = min(length(pilotBits), length(demappedBits));
    numErrors = sum(pilotBits(1:minLen) ~= demappedBits(1:minLen));
    fitness = numErrors / minLen;  % BER
    
    % Penalty for unrealistic boundaries (prevent divergence)
    maxOffset = 5;  % Maximum allowed offset from origin
    if any(abs(boundaries) > maxOffset)
        fitness = fitness + 1.0;  % Heavy penalty
    end
end

function bits = demap_with_boundaries(symbols, boundaries, modOrder)
    % Demap symbols using custom decision boundaries
    nSymbols = length(symbols);
    bitsPerSymbol = log2(modOrder);
    bits = zeros(nSymbols * bitsPerSymbol, 1);
    
    for i = 1:nSymbols
        % Extract I and Q components
        I = real(symbols(i));
        Q = imag(symbols(i));
        
        % Apply decision boundaries to get symbol index
        symIdx = apply_boundaries(I, Q, boundaries, modOrder);
        
        % Convert symbol index to bits (Gray coding)
        bitPattern = de2bi(symIdx, bitsPerSymbol, 'left-msb');
        bits((i-1)*bitsPerSymbol + 1 : i*bitsPerSymbol) = bitPattern;
    end
end

function symIdx = apply_boundaries(I, Q, boundaries, modOrder)
    % Apply custom boundaries to I/Q to determine symbol
    switch modOrder
        case 4  % QPSK
            I_thresh = boundaries(1);
            Q_thresh = boundaries(2);
            
            if I >= I_thresh && Q >= Q_thresh
                symIdx = 0;  % +1+j
            elseif I < I_thresh && Q >= Q_thresh
                symIdx = 1;  % -1+j
            elseif I >= I_thresh && Q < Q_thresh
                symIdx = 2;  % +1-j
            else
                symIdx = 3;  % -1-j
            end
            
        case 16  % 16-QAM
            I_bounds = boundaries(1:3);
            Q_bounds = boundaries(4:6);
            
            % Find which region I and Q fall into (4 regions each)
            I_region = sum(I > I_bounds);  % 0, 1, 2, or 3
            Q_region = sum(Q > Q_bounds);
            
            symIdx = I_region * 4 + Q_region;  % 0-15
            
        case 64  % 64-QAM
            I_bounds = boundaries(1:7);
            Q_bounds = boundaries(8:14);
            
            I_region = sum(I > I_bounds);  % 0-7
            Q_region = sum(Q > Q_bounds);
            
            symIdx = I_region * 8 + Q_region;  % 0-63
    end
end

function newPop = evolve_population(pop, fitness, params)
    % Evolve population through selection, crossover, mutation
    [popSize, nBounds] = size(pop);
    newPop = zeros(popSize, nBounds);
    
    % Step 1: Elitism - keep best solutions
    [~, sortedIdx] = sort(fitness);
    newPop(1:params.eliteCount, :) = pop(sortedIdx(1:params.eliteCount), :);
    
    % Step 2: Selection + Crossover + Mutation
    for i = (params.eliteCount + 1):popSize
        % Tournament selection (pick 3 random, choose best)
        candidates = randperm(popSize, 3);
        [~, winner] = min(fitness(candidates));
        parent1 = pop(candidates(winner), :);
        
        candidates = randperm(popSize, 3);
        [~, winner] = min(fitness(candidates));
        parent2 = pop(candidates(winner), :);
        
        % Crossover (single-point)
        crossPoint = randi([1, nBounds-1]);
        child = [parent1(1:crossPoint), parent2(crossPoint+1:end)];
        
        % Mutation
        if rand < params.mutationRate
            mutIdx = randi(nBounds);
            child(mutIdx) = child(mutIdx) + 0.2 * randn;  % Small random change
        end
        
        newPop(i, :) = child;
    end
end
