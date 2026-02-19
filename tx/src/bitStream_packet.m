function pkt = bitStream_packet(params, base_coeff_int16, enh_coeff_int8) % big-endian order 
    K = uint8(params.K);
    packet_mode = uint8(params.packet_mode);
    if packet_mode == 0
        assert(numel(base_coeff_int16) == double(K), "base length must be K");% combined both layers
        assert(numel(enh_coeff_int8) == double(64 - K), "enh length must be 64-K");
        payload_len_bytes = uint8(64 + K); % (16K+(64−K) * 8) /2 
        layer_type = uint8(0); % ignored by Rx
    else
        layer_type = uint8(params.layer_type);
        if layer_type == 1 % base layer 
            assert(numel(base_coeff_int16) == double(K), "base length must be K");
            payload_len_bytes = uint8(2*K);
        elseif layer_type == 3 % enhance layer 
            assert(numel(enh_coeff_int8) == double(64 - K), "enh length must be 64-K");
            payload_len_bytes = uint8(64 - K);
        else
            error("Unsupported layer_type for packet_mode=1. Use 01(base) or 11(enh).");
        end
    end
    header = zeros(1,8,'uint8');
    header(1) = bitor(bitshift(uint8(params.Qb_idx),4), uint8(params.Qe_idx));
    header(2) = uint8(params.K);
    header(3) = uint8(params.N);
    flag = pack_flags5(uint8(params.crc_flag), uint8(params.endian_flag), layer_type, uint8(params.reserved_flag));
    header(4) = bitor( bitshift(uint8(params.codec_id),6),bitor( bitshift(packet_mode,5), flag ) );
    header(5) = uint8(params.frame_id);
    header(6:7) = pack_u16_be(uint16(params.block_id));
    header(8) = payload_len_bytes;
    payload = zeros(1,double(payload_len_bytes),'uint8');
    idx = 1;
    if packet_mode == 0
        % base: K * int16
        base_bytes = pack_int16_vec(base_coeff_int16, uint8(params.endian_flag));
        payload(idx:idx+numel(base_bytes)-1) = base_bytes; 
        idx = idx + numel(base_bytes);

        % enh: (64-K) * int8
        enh_bytes = typecast(int8(enh_coeff_int8(:)).', 'uint8');
        payload(idx:idx+numel(enh_bytes)-1) = enh_bytes; 
        idx = idx + numel(enh_bytes);

    else
        if layer_type == 1
            base_bytes = pack_int16_vec(base_coeff_int16, uint8(params.endian_flag));
            payload(idx:idx+numel(base_bytes)-1) = base_bytes; 
            idx = idx + numel(base_bytes);
        elseif layer_type == 3
            enh_bytes = typecast(int8(enh_coeff_int8(:)).', 'uint8');
            payload(idx:idx+numel(enh_bytes)-1) = enh_bytes; 
            idx = idx + numel(enh_bytes);
        end
    end

    assert(idx-1 == numel(payload), "Payload length mismatch");

    
    sync = uint32(hex2dec('AAAAAAAA')); %Sync word
    sync_bytes = pack_u32_be(sync);

    
    if uint8(params.crc_flag) == 1 % CRC enable
        crc = crc16_ccitt_false([header payload]);  % x^16 + x^12 + x^5 +1
        crc_bytes = pack_u16_be(crc);
        pkt = [sync_bytes header payload crc_bytes];
    else
        pkt = [sync_bytes header payload];
    end
end

function crc = crc16_ccitt_false(data_bytes)
    data = uint8(data_bytes(:)).';
    crc = uint16(hex2dec('FFFF'));
    poly = uint16(hex2dec('1021'));

    for i = 1:numel(data)
        crc = bitxor(crc, bitshift(uint16(data(i)), 8));
        for k = 1:8
            if bitand(crc, uint16(32768)) ~= 0
                crc = bitxor(bitshift(crc,1), poly);
            else
                crc = bitshift(crc,1);
            end
        end
    end
end

function flags5 = pack_flags5(crc_flag, endian_flag, layer_type, reserved_flag)
    flags5 = uint8(0);
    flags5 = bitor(flags5, bitshift(uint8(crc_flag~=0),4));
    flags5 = bitor(flags5, bitshift(uint8(endian_flag~=0),3));
    flags5 = bitor(flags5, bitshift(bitand(uint8(layer_type),3),1));
    flags5 = bitor(flags5, bitand(uint8(reserved_flag~=0),1));
end

function b = pack_u16_be(x)
    x = uint16(x);
    b = uint8([bitshift(x,-8), bitand(x,255)]);
end

function b = pack_u32_be(x)
    x = uint32(x);
    b = uint8([ ...
        bitshift(x,-24), ...
        bitand(bitshift(x,-16),255), ...
        bitand(bitshift(x,-8),255), ...
        bitand(x,255) ...
    ]);
end

function bytes = pack_int16_vec(v_int16, endian_flag)
% endian_flag: 0 little-endian, 1 big-endian
    v = int16(v_int16(:)).';
    u = uint16(typecast(v, 'uint16'));
    hi = uint8(bitshift(u,-8));
    lo = uint8(bitand(u,255));
    if endian_flag == 1
        bytes = reshape([hi; lo], 1, []);
    else
        bytes = reshape([lo; hi], 1, []);
    end
end

