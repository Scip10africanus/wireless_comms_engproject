classdef PolicyController < handle
    % PolicyController
    % Frame-level policy snapshot manager 
    properties
        CsvPath (1,1) string = "policy.csv"
        DefaultPolicy struct
        Policy struct

        % tracking for change 
        LastMTime double = NaN     % datenum
        LastVersion uint32 = uint32(0)

        RequireMonotonicVersion (1,1) logical = false  % true: reject version rollback
        FallbackToDefaultOnError (1,1) logical = false  % false: keep last policy on CSV error
        Verbose (1,1) logical = true
    end

    methods
        function obj = PolicyController(csvPath, defaultPolicy)
            if nargin >= 1 && ~isempty(csvPath)
                obj.CsvPath = string(csvPath);
            end
            if nargin >= 2 && ~isempty(defaultPolicy)
                obj.DefaultPolicy = defaultPolicy;
            else
                obj.DefaultPolicy = PolicyController.defaultPolicy();
            end
            obj.Policy = obj.DefaultPolicy;
        end

        function policy = tick(obj, frame_id) % frame id for furtrue extension
            if ~isfile(obj.CsvPath)
                policy = obj.Policy;
                return;
            end

            info = dir(obj.CsvPath);
            mtime = info.datenum;

            if ~isnan(obj.LastMTime) && mtime == obj.LastMTime % unchanged, reuse
                policy = obj.Policy;
                return;
            end

            try
                T = readtable(obj.CsvPath, 'TextType', 'string');
                if height(T) < 1
                    error("Empty policy table.");
                end
                
                row = T(end, :); % using latest params
                newp = obj.parseRowToPolicy(row, obj.DefaultPolicy);
                newp = obj.sanitizePolicy(newp);

                
                if obj.RequireMonotonicVersion && newp.version < obj.LastVersion % rollback
                    if obj.Verbose
                        warning("PolicyController: version rollback detected (new %u < last %u). Keeping old policy.", ...
                            newp.version, obj.LastVersion);
                    end
                    policy = obj.Policy;
                    return;
                end

                % accept
                obj.Policy = newp;
                obj.LastMTime = mtime;
                obj.LastVersion = newp.version;

                if obj.Verbose
                    fprintf("[Policy] loaded v%u tx_mode=%s K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d gap=%gus\n", ...
                        obj.Policy.version, obj.Policy.tx_mode, obj.Policy.K, obj.Policy.Qb_idx, obj.Policy.Qe_idx, ...
                        obj.Policy.mcs_base, obj.Policy.mcs_enh, obj.Policy.mux_gap_us);
                end

            catch ME
                if obj.Verbose
                    warning("PolicyController: failed to load policy (%s).", ME.message);
                end
                if obj.FallbackToDefaultOnError
                    obj.Policy = obj.DefaultPolicy;
                end
                % keep LastMTime unchanged so we retry next frame
            end

            policy = obj.Policy;
        end

        function params = makeParams(obj, frame_id, block_id, layer_type, N)
            % Build params struct for bitStream_packet according to current policy snapshot.
            if nargin < 5 || isempty(N)
                N = uint8(8);
            end
            p = obj.Policy;

            params = struct();
            params.Qb_idx = uint8(p.Qb_idx);
            params.Qe_idx = uint8(p.Qe_idx);
            params.K      = uint8(p.K);
            params.N      = uint8(N);

            params.codec_id = uint8(p.codec_id);
            params.crc_flag = uint8(p.crc_flag);
            params.endian_flag = uint8(p.endian_flag);
            params.reserved_flag = uint8(p.reserved_flag);

            if string(p.tx_mode) == "split"
                params.packet_mode = uint8(1);
                params.layer_type  = uint8(layer_type); % 01 => base, 11 => enh
            else
                params.packet_mode = uint8(0);
                params.layer_type  = uint8(0); % ignored by Rx
            end

            params.frame_id = uint8(frame_id);
            params.block_id = uint16(block_id);
        end

        function phy = makePhy(obj) % for PHY layer 
            p = obj.Policy;
            phy = struct();
            phy.mcs_base   = uint8(p.mcs_base);
            phy.mcs_enh    = uint8(p.mcs_enh);
            phy.mux_gap_us = double(p.mux_gap_us);
            phy.tx_mode    = string(p.tx_mode);
        end
    end

    methods (Access=private)
        function p = sanitizePolicy(~, p)
            tm = string(p.tx_mode);
            if ~(tm=="combined" || tm=="split")
                tm = "split";
            end
            p.tx_mode = tm;

            p.K      = uint8(max(0, min(64, double(p.K))));
            p.Qb_idx = uint8(max(0, min(15, double(p.Qb_idx))));
            p.Qe_idx = uint8(max(0, min(15, double(p.Qe_idx))));

            p.crc_flag    = uint8(p.crc_flag ~= 0);
            p.endian_flag = uint8(p.endian_flag ~= 0);
            p.reserved_flag = uint8(p.reserved_flag ~= 0);

            p.codec_id = uint8(max(0, min(3, double(p.codec_id))));
            p.mcs_base = uint8(max(0, min(7, double(p.mcs_base))));
            p.mcs_enh  = uint8(max(0, min(7, double(p.mcs_enh))));

            if isnan(p.mux_gap_us) || p.mux_gap_us < 0
                p.mux_gap_us = 0;
            end
        end
    end

    methods (Static)
        function p = defaultPolicy()
            p = struct();
            p.version       = uint32(0);
            p.timestamp     = "";
            p.tx_mode       = "combined";

            p.K             = uint8(20);
            p.Qb_idx        = uint8(0);
            p.Qe_idx        = uint8(0);
            p.crc_flag      = uint8(1);
            p.endian_flag   = uint8(1);
            p.codec_id      = uint8(0);
            p.reserved_flag = uint8(0);

            p.mcs_base      = uint8(0);
            p.mcs_enh       = uint8(5);
            p.mux_gap_us    = double(0);
        end

        function p = parseRowToPolicy(row, base)

            vars = lower(string(row.Properties.VariableNames)); % change to lower case
            row.Properties.VariableNames = cellstr(vars);

            p = base;

            p = setField(p, row, "version", "version", @uint32);
            p = setField(p, row, "timestamp", "timestamp", @string);
            p = setField(p, row, "tx_mode", "tx_mode", @string);

            p = setField(p, row, "k", "K", @uint8);
            p = setField(p, row, "qb_idx", "Qb_idx", @uint8);
            p = setField(p, row, "qe_idx", "Qe_idx", @uint8);
            p = setField(p, row, "crc_flag", "crc_flag", @uint8);
            p = setField(p, row, "endian_flag", "endian_flag", @uint8);
            p = setField(p, row, "codec_id", "codec_id", @uint8);
            p = setField(p, row, "reserved_flag", "reserved_flag", @uint8);

            p = setField(p, row, "mcs_base", "mcs_base", @uint8);
            p = setField(p, row, "mcs_enh", "mcs_enh", @uint8);
            p = setField(p, row, "mux_gap_us", "mux_gap_us", @double);

            function p2 = setField(p2, r, col, field, cast)
                col = lower(string(col));
                if any(strcmpi(r.Properties.VariableNames, col))
                    v = r.(col);
                    if iscell(v), v = v{1}; end
                    if ismissing(v), return; end
                    try
                        p2.(field) = cast(v);
                    catch
                        % ignore 
                    end
                end
            end
        end
    end
end
