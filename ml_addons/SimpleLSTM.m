classdef SimpleLSTM < handle
% SIMPLELSTM  LSTM network from scratch with full BPTT
% No toolbox dependencies. Implements forget/input/output gates,
% cell state, and backpropagation through time.
%
% Bold (Unubold) - EEME30003 Major Project
%
% Usage:
%   lstm = SimpleLSTM(1, 32, 1, 0.001);      % 1-in, 32 hidden, 1-out
%   [y, states] = lstm.forward_seq(X_seq);    % X: [inputDim x seqLen]
%   loss = lstm.train_step(X_seq, y_true);    % BPTT
%
% Architecture:
%   Input -> LSTM cell (recurrent) -> Dense (linear) -> Output

    properties
        inputSize
        hiddenSize
        outputSize
        lr
        mom

        % Gate weights: each [hiddenSize x (inputSize + hiddenSize)]
        Wf; Wi; Wc; Wo   % forget, input, candidate, output
        bf; bi; bc; bo    % biases [hiddenSize x 1]

        % Output layer
        Wy  % [outputSize x hiddenSize]
        by  % [outputSize x 1]

        % Momentum buffers
        vWf; vWi; vWc; vWo; vbf; vbi; vbc; vbo; vWy; vby

        trainLoss
    end

    methods
        function obj = SimpleLSTM(inputSize, hiddenSize, outputSize, lr, momentum)
            if nargin < 4, lr = 1e-3; end
            if nargin < 5, momentum = 0.9; end

            obj.inputSize  = inputSize;
            obj.hiddenSize = hiddenSize;
            obj.outputSize = outputSize;
            obj.lr  = lr;
            obj.mom = momentum;
            obj.trainLoss = [];

            d = inputSize + hiddenSize;
            s = hiddenSize;

            % Xavier init for gates
            sc = sqrt(2 / d);
            obj.Wf = randn(s, d) * sc;  obj.bf = zeros(s, 1);
            obj.Wi = randn(s, d) * sc;  obj.bi = zeros(s, 1);
            obj.Wc = randn(s, d) * sc;  obj.bc = zeros(s, 1);
            obj.Wo = randn(s, d) * sc;  obj.bo = zeros(s, 1);

            % Forget gate bias init to 1 (helps learning long dependencies)
            obj.bf = ones(s, 1);

            % Output layer
            sc2 = sqrt(2 / s);
            obj.Wy = randn(outputSize, s) * sc2;
            obj.by = zeros(outputSize, 1);

            % Init momentum buffers
            obj.vWf = zeros(s,d); obj.vWi = zeros(s,d);
            obj.vWc = zeros(s,d); obj.vWo = zeros(s,d);
            obj.vbf = zeros(s,1); obj.vbi = zeros(s,1);
            obj.vbc = zeros(s,1); obj.vbo = zeros(s,1);
            obj.vWy = zeros(outputSize, s);
            obj.vby = zeros(outputSize, 1);
        end

        function [y, cache] = forward_seq(obj, X_seq)
        % FORWARD_SEQ  Forward pass through full sequence
        %   X_seq: [inputSize x seqLen]
        %   y:     [outputSize x 1] (uses last hidden state)
        %   cache: struct with all intermediate values for BPTT

            T = size(X_seq, 2);  % sequence length
            s = obj.hiddenSize;

            % Allocate storage
            h_all = zeros(s, T+1);  % h_all(:,1) = h_0 = zeros
            c_all = zeros(s, T+1);  % c_all(:,1) = c_0 = zeros
            f_all = zeros(s, T);
            i_all = zeros(s, T);
            g_all = zeros(s, T);    % candidate (c_tilde)
            o_all = zeros(s, T);
            z_all = cell(1, T);     % concatenated [h; x]

            for t = 1:T
                h_prev = h_all(:, t);
                c_prev = c_all(:, t);
                x_t = X_seq(:, t);

                z = [h_prev; x_t];  % [d x 1]
                z_all{t} = z;

                f_t = sigmoid_(obj.Wf * z + obj.bf);
                i_t = sigmoid_(obj.Wi * z + obj.bi);
                g_t = tanh(obj.Wc * z + obj.bc);
                o_t = sigmoid_(obj.Wo * z + obj.bo);

                c_t = f_t .* c_prev + i_t .* g_t;
                h_t = o_t .* tanh(c_t);

                h_all(:, t+1) = h_t;
                c_all(:, t+1) = c_t;
                f_all(:, t) = f_t;
                i_all(:, t) = i_t;
                g_all(:, t) = g_t;
                o_all(:, t) = o_t;
            end

            % Output from last hidden state
            y = obj.Wy * h_all(:, T+1) + obj.by;

            cache.h_all = h_all;
            cache.c_all = c_all;
            cache.f_all = f_all;
            cache.i_all = i_all;
            cache.g_all = g_all;
            cache.o_all = o_all;
            cache.z_all = z_all;
            cache.X_seq = X_seq;
            cache.T = T;
        end

        function y = predict_seq(obj, X_seq)
        % PREDICT_SEQ  Inference only (no cache)
            [y, ~] = obj.forward_seq(X_seq);
        end

        function loss = train_step(obj, X_seq, y_true)
        % TRAIN_STEP  Full BPTT gradient computation + SGD update
        %   X_seq:  [inputSize x seqLen]
        %   y_true: [outputSize x 1]

            [y_pred, cache] = obj.forward_seq(X_seq);
            T = cache.T;
            s = obj.hiddenSize;

            % MSE loss
            diff = y_pred - y_true;
            loss = 0.5 * sum(diff.^2);

            % --- Backprop through output layer ---
            dWy = diff * cache.h_all(:, T+1)';
            dby = diff;
            dh_next = obj.Wy' * diff;   % gradient flowing into h_T

            % --- BPTT through time ---
            dWf = zeros(size(obj.Wf)); dbf = zeros(size(obj.bf));
            dWi = zeros(size(obj.Wi)); dbi = zeros(size(obj.bi));
            dWc = zeros(size(obj.Wc)); dbc = zeros(size(obj.bc));
            dWo = zeros(size(obj.Wo)); dbo = zeros(size(obj.bo));

            dc_next = zeros(s, 1);

            for t = T:-1:1
                h_t = cache.h_all(:, t+1);
                c_t = cache.c_all(:, t+1);
                c_prev = cache.c_all(:, t);
                f_t = cache.f_all(:, t);
                i_t = cache.i_all(:, t);
                g_t = cache.g_all(:, t);
                o_t = cache.o_all(:, t);
                z_t = cache.z_all{t};

                dh = dh_next;

                % Through o_t * tanh(c_t)
                tanh_c = tanh(c_t);
                do = dh .* tanh_c .* o_t .* (1 - o_t);
                dc = dh .* o_t .* (1 - tanh_c.^2) + dc_next;

                % Through cell state
                df = dc .* c_prev .* f_t .* (1 - f_t);
                di = dc .* g_t .* i_t .* (1 - i_t);
                dg = dc .* i_t .* (1 - g_t.^2);

                % Accumulate gate gradients
                dWf = dWf + df * z_t';  dbf = dbf + df;
                dWi = dWi + di * z_t';  dbi = dbi + di;
                dWc = dWc + dg * z_t';  dbc = dbc + dg;
                dWo = dWo + do * z_t';  dbo = dbo + do;

                % Gradient to previous timestep
                dz = obj.Wf' * df + obj.Wi' * di + obj.Wc' * dg + obj.Wo' * do;
                dh_next = dz(1:s);       % first hiddenSize elements
                dc_next = dc .* f_t;     % cell state gradient
            end

            % Gradient clipping (prevent explosion)
            clipVal = 5.0;
            dWf = max(min(dWf, clipVal), -clipVal);
            dWi = max(min(dWi, clipVal), -clipVal);
            dWc = max(min(dWc, clipVal), -clipVal);
            dWo = max(min(dWo, clipVal), -clipVal);
            dWy = max(min(dWy, clipVal), -clipVal);

            % SGD with momentum updates
            obj.vWf = obj.mom*obj.vWf - obj.lr*dWf; obj.Wf = obj.Wf + obj.vWf;
            obj.vbf = obj.mom*obj.vbf - obj.lr*dbf; obj.bf = obj.bf + obj.vbf;
            obj.vWi = obj.mom*obj.vWi - obj.lr*dWi; obj.Wi = obj.Wi + obj.vWi;
            obj.vbi = obj.mom*obj.vbi - obj.lr*dbi; obj.bi = obj.bi + obj.vbi;
            obj.vWc = obj.mom*obj.vWc - obj.lr*dWc; obj.Wc = obj.Wc + obj.vWc;
            obj.vbc = obj.mom*obj.vbc - obj.lr*dbc; obj.bc = obj.bc + obj.vbc;
            obj.vWo = obj.mom*obj.vWo - obj.lr*dWo; obj.Wo = obj.Wo + obj.vWo;
            obj.vbo = obj.mom*obj.vbo - obj.lr*dbo; obj.bo = obj.bo + obj.vbo;
            obj.vWy = obj.mom*obj.vWy - obj.lr*dWy; obj.Wy = obj.Wy + obj.vWy;
            obj.vby = obj.mom*obj.vby - obj.lr*dby; obj.by = obj.by + obj.vby;

            obj.trainLoss(end+1) = loss;
        end

        function save_model(obj, filepath)
            params = struct();
            params.Wf = obj.Wf; params.Wi = obj.Wi;
            params.Wc = obj.Wc; params.Wo = obj.Wo;
            params.bf = obj.bf; params.bi = obj.bi;
            params.bc = obj.bc; params.bo = obj.bo;
            params.Wy = obj.Wy; params.by = obj.by;
            params.inputSize  = obj.inputSize;
            params.hiddenSize = obj.hiddenSize;
            params.outputSize = obj.outputSize;
            params.trainLoss  = obj.trainLoss;
            save(filepath, '-struct', 'params');
            fprintf('[SimpleLSTM] Saved to %s\n', filepath);
        end

        function load_model(obj, filepath)
            p = load(filepath);
            obj.Wf = p.Wf; obj.Wi = p.Wi; obj.Wc = p.Wc; obj.Wo = p.Wo;
            obj.bf = p.bf; obj.bi = p.bi; obj.bc = p.bc; obj.bo = p.bo;
            obj.Wy = p.Wy; obj.by = p.by;
            if isfield(p,'trainLoss'), obj.trainLoss = p.trainLoss; end
            fprintf('[SimpleLSTM] Loaded from %s\n', filepath);
        end
    end
end

%% Local helper
function y = sigmoid_(x)
    y = 1 ./ (1 + exp(-x));
end
