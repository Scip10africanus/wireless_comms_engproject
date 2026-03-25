function pc = PolicyControllerSimple(csvPath)
% PolicyControllerSimple - struct-based version of PolicyController (works in Octave)
% reads policy.csv each frame and returns updated params
% usage: pc = PolicyControllerSimple('policy.csv')
%        [policy, pc] = pc.tick(pc, frameId)   <-- must capture pc back (pass by value!)

    pc = struct();
    pc.CsvPath = csvPath;
    pc.Policy = default_policy();
    pc.LastMTime = NaN;
    pc.Verbose = true;

    pc.tick = @policy_tick;
    pc.makeParams = @policy_makeParams;
    pc.makePhy = @policy_makePhy;
end

function [policy, pc] = policy_tick(pc, frame_id)
    if ~exist(pc.CsvPath, 'file')
        policy = pc.Policy;
        return;
    end

    try
        fid = fopen(pc.CsvPath, 'r');
        fgetl(fid);  % skip header
        dataLine = fgetl(fid);
        fclose(fid);

        if ischar(dataLine) && ~isempty(dataLine)
            p = strsplit(dataLine, ',');
            if length(p) >= 13
                newVersion = uint32(str2double(p{1}));

                % Use version number for staleness check — mtime has only
                % 1-second resolution and causes false cache hits in sequential sims
                if newVersion == pc.Policy.version
                    policy = pc.Policy;
                    return;
                end

                newp = default_policy();
                newp.version    = newVersion;
                newp.timestamp  = p{2};
                newp.tx_mode    = strtrim(p{3});
                newp.K          = uint8(str2double(p{4}));
                newp.Qb_idx     = uint8(str2double(p{5}));
                newp.Qe_idx     = uint8(str2double(p{6}));
                newp.crc_flag   = uint8(str2double(p{7}));
                newp.endian_flag= uint8(str2double(p{8}));
                newp.codec_id   = uint8(str2double(p{9}));
                newp.reserved_flag = uint8(str2double(p{10}));
                newp.mcs_base   = uint8(str2double(p{11}));
                newp.mcs_enh    = uint8(str2double(p{12}));
                newp.mux_gap_us = str2double(p{13});

                pc.Policy = newp;
                pc.LastMTime = now;

                if pc.Verbose
                    fprintf('[Policy] loaded v%d tx_mode=%s K=%d Qb=%d Qe=%d mcsB=%d mcsE=%d gap=%gus\n', ...
                        newp.version, newp.tx_mode, newp.K, newp.Qb_idx, newp.Qe_idx, ...
                        newp.mcs_base, newp.mcs_enh, newp.mux_gap_us);
                end
            end
        end
    catch ME
        if pc.Verbose
            warning('PolicyControllerSimple: %s', ME.message);
        end
    end
    policy = pc.Policy;
end

function params = policy_makeParams(pc, frame_id, block_id, layer_type, N)
    if nargin < 5, N = uint8(8); end
    p = pc.Policy;

    params = struct();
    params.Qb_idx   = uint8(p.Qb_idx);
    params.Qe_idx   = uint8(p.Qe_idx);
    params.K        = uint8(p.K);
    params.N        = uint8(N);
    params.codec_id = uint8(p.codec_id);
    params.crc_flag = uint8(p.crc_flag);
    params.endian_flag = uint8(p.endian_flag);
    params.reserved_flag = uint8(p.reserved_flag);

    if strcmp(p.tx_mode, 'split')
        params.packet_mode = uint8(1);
        params.layer_type  = uint8(layer_type);
    else
        params.packet_mode = uint8(0);
        params.layer_type  = uint8(0);
    end
    params.frame_id = uint8(frame_id);
    params.block_id = uint16(block_id);
end

function phy = policy_makePhy(pc)
    p = pc.Policy;
    phy = struct('mcs_base', uint8(p.mcs_base), 'mcs_enh', uint8(p.mcs_enh), ...
                 'mux_gap_us', double(p.mux_gap_us), 'tx_mode', p.tx_mode);
end

function p = default_policy()
    p = struct();
    p.version       = uint32(0);
    p.timestamp     = '';
    p.tx_mode       = 'split';
    p.K             = uint8(20);
    p.Qb_idx        = uint8(2);   % mid-range start so adaptation can increase or decrease
    p.Qe_idx        = uint8(2);
    p.crc_flag      = uint8(1);
    p.endian_flag   = uint8(1);
    p.codec_id      = uint8(0);
    p.reserved_flag = uint8(0);
    p.mcs_base      = uint8(2);   % QPSK (matches original hardcoded base modulation)
    p.mcs_enh       = uint8(6);   % 64-QAM (matches original hardcoded enh modulation)
    p.mux_gap_us    = double(0);
end
