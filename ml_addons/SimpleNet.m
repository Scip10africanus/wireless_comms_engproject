classdef SimpleNet < handle
% SIMPLENET  Minimal feedforward neural network (MLP) from scratch
% No toolbox dependencies. Supports arbitrary depth, ReLU hidden layers,
% linear or sigmoid output, SGD with momentum, Xavier init.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   net = SimpleNet([10 64 32 1], 0.001);   % 10-in, 64, 32, 1-out
%   [y, cache] = net.forward(X);             % X: [inputDim x batch]
%   loss = net.train_step(X, Y);             % one gradient step
%   net.save_model('quality_net.mat');
%   net.load_model('quality_net.mat');

    properties
        W           % cell array of weight matrices
        b           % cell array of bias vectors
        nLayers     % number of weight layers
        layerSizes  % original architecture
        lr          % learning rate
        mom         % momentum coefficient
        vW          % momentum buffer for W
        vb          % momentum buffer for b
        outputAct   % 'linear' | 'sigmoid'
        trainLoss   % training loss history
    end

    methods
        function obj = SimpleNet(sizes, lr, momentum, outputAct)
        % SIMPLENET Constructor
        %   sizes      = [inputDim, hidden1, hidden2, ..., outputDim]
        %   lr         = learning rate (default 1e-3)
        %   momentum   = SGD momentum  (default 0.9)
        %   outputAct  = 'linear' | 'sigmoid' (default 'linear')
            if nargin < 2, lr = 1e-3; end
            if nargin < 3, momentum = 0.9; end
            if nargin < 4, outputAct = 'linear'; end

            obj.layerSizes = sizes;
            obj.nLayers = length(sizes) - 1;
            obj.lr  = lr;
            obj.mom = momentum;
            obj.outputAct = outputAct;
            obj.trainLoss = [];

            obj.W  = cell(1, obj.nLayers);
            obj.b  = cell(1, obj.nLayers);
            obj.vW = cell(1, obj.nLayers);
            obj.vb = cell(1, obj.nLayers);

            for i = 1:obj.nLayers
                fan_in  = sizes(i);
                fan_out = sizes(i+1);
                % He initialization (good for ReLU)
                scale = sqrt(2 / fan_in);
                obj.W{i}  = randn(fan_out, fan_in) * scale;
                obj.b{i}  = zeros(fan_out, 1);
                obj.vW{i} = zeros(fan_out, fan_in);
                obj.vb{i} = zeros(fan_out, 1);
            end
        end

        function [y, cache] = forward(obj, x)
        % FORWARD  Forward pass through all layers
        %   x: [inputDim x batchSize]
        %   y: [outputDim x batchSize]
            cache = struct('a', {}, 'z', {});
            a = x;
            for i = 1:obj.nLayers
                z = obj.W{i} * a + obj.b{i};
                cache(i).a = a;    % pre-activation input
                cache(i).z = z;    % pre-activation output

                if i < obj.nLayers
                    a = max(0, z);  % ReLU for hidden layers
                else
                    % Output activation
                    if strcmp(obj.outputAct, 'sigmoid')
                        a = 1 ./ (1 + exp(-z));
                    else
                        a = z;  % linear
                    end
                end
            end
            y = a;
        end

        function y = predict(obj, x)
        % PREDICT  Forward pass without cache (inference only)
            [y, ~] = obj.forward(x);
        end

        function loss = train_step(obj, x, y_true)
        % TRAIN_STEP  One forward+backward pass with SGD momentum
        %   x:      [inputDim x batchSize]
        %   y_true: [outputDim x batchSize]
            batchSize = size(x, 2);
            [y_pred, cache] = obj.forward(x);

            % MSE loss
            diff = y_pred - y_true;
            loss = 0.5 * mean(sum(diff.^2, 1));

            % Output gradient
            dA = diff / batchSize;
            if strcmp(obj.outputAct, 'sigmoid')
                sig = y_pred;
                dA = dA .* sig .* (1 - sig);
            end

            % Backprop through layers
            for i = obj.nLayers:-1:1
                if i < obj.nLayers
                    % ReLU derivative
                    dA = dA .* (cache(i).z > 0);
                end

                gW = dA * cache(i).a';
                gb = sum(dA, 2);

                if i > 1
                    dA = obj.W{i}' * dA;
                end

                % Gradient clipping
                gW = max(min(gW, 5), -5);
                gb = max(min(gb, 5), -5);

                % SGD with momentum
                obj.vW{i} = obj.mom * obj.vW{i} - obj.lr * gW;
                obj.vb{i} = obj.mom * obj.vb{i} - obj.lr * gb;
                obj.W{i}  = obj.W{i} + obj.vW{i};
                obj.b{i}  = obj.b{i} + obj.vb{i};
            end

            obj.trainLoss(end+1) = loss;
        end

        function [avgLoss] = train_epoch(obj, X, Y, batchSize)
        % TRAIN_EPOCH  Train one full pass over the dataset
        %   X: [inputDim x N], Y: [outputDim x N]
            if nargin < 4, batchSize = 64; end
            N = size(X, 2);
            perm = randperm(N);
            losses = [];
            for s = 1:batchSize:N
                idx = perm(s:min(s+batchSize-1, N));
                l = obj.train_step(X(:,idx), Y(:,idx));
                losses(end+1) = l;
            end
            avgLoss = mean(losses);
        end

        function save_model(obj, filepath)
        % SAVE_MODEL  Save weights to .mat file
            params = struct();
            params.W = obj.W;
            params.b = obj.b;
            params.layerSizes = obj.layerSizes;
            params.outputAct = obj.outputAct;
            params.lr = obj.lr;
            params.trainLoss = obj.trainLoss;
            save(filepath, '-struct', 'params');
            fprintf('[SimpleNet] Saved to %s\n', filepath);
        end

        function load_model(obj, filepath)
        % LOAD_MODEL  Load weights from .mat file
            params = load(filepath);
            obj.W = params.W;
            obj.b = params.b;
            if isfield(params, 'trainLoss')
                obj.trainLoss = params.trainLoss;
            end
            fprintf('[SimpleNet] Loaded from %s\n', filepath);
        end
    end
end
