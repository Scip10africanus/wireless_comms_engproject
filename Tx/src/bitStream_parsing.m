function out = bitStream_parsing(pkt_bytes)
    b = uint8(pkt_bytes(:)).'; % row
    sync_rx = unpack_u32_be(b(1:4)); %Sync check
    assert(sync_rx == uint32(hex2dec('AAAAAAAA')), "Sync word mismatch");
    header = b(5:12);

    Qb_idx = bitshift(header(1), -4);
    Qe_idx = bitand(header(1), 15);

    K = header(2);
    N = header(3);

    codec_id = bitshift(header(4), -6);
    packet_mode = bitand(bitshift(header(4), -5), 1);
    flags5 = bitand(header(4), 31);

    [crc_flag, endian_flag, layer_type, reserved_flag] = unpack_flags5(flags5);

    frame_id = header(5);
    block_id = unpack_u16_be(header(6:7));
    payload_len_bytes = header(8);

    assert(reserved_flag == 0, "Unsupported/reserved_flag set");


    base_total = 4 + 8 + double(payload_len_bytes);
    if crc_flag == 1
        expected_len = base_total + 2;
    else
        expected_len = base_total;
    end
    assert(numel(b) == expected_len, "Packet length mismatch (expected %d, got %d)", expected_len, numel(b));

    payload = b(13 : 12 + double(payload_len_bytes));

    
    if crc_flag == 1 %CRC verify
        crc_rx = unpack_u16_be(b(end-1:end));
        crc_calc = crc16_ccitt_false([header payload]);
        assert(crc_rx == crc_calc, "CRC mismatch");
    end

    base_coeff = [];
    enh_coeff = [];

    if packet_mode == 0
        assert(payload_len_bytes == uint8(64 + K), "payload_len_bytes inconsistent for combined mode");
        base_bytes_len = 2*double(K);
        base_bytes = payload(1:base_bytes_len);
        enh_bytes  = payload(base_bytes_len+1:end);

        base_coeff = unpack_int16_vec(base_bytes, endian_flag);
        enh_coeff = typecast(uint8(enh_bytes), 'int8');

    else
        if layer_type == 1
            assert(payload_len_bytes == uint8(2*K), "payload_len_bytes inconsistent for base-only");
            base_coeff = unpack_int16_vec(payload, endian_flag);
        elseif layer_type == 3
            assert(payload_len_bytes == uint8(64 - K), "payload_len_bytes inconsistent for enh-only");
            enh_coeff = typecast(uint8(payload), 'int8');
        else
            % should not happen if Tx follows spec, but keep safe:
            error("Unsupported layer_type in packet_mode=1");
        end
    end
    out = struct();
    out.params = struct('Qb_idx', Qb_idx, 'Qe_idx', Qe_idx, 'K', K, 'N', N, ...
        'codec_id', codec_id, 'packet_mode', packet_mode, ...
        'crc_flag', crc_flag, 'endian_flag', endian_flag, ...
        'layer_type', layer_type, 'reserved_flag', reserved_flag, ...
        'frame_id', frame_id, 'block_id', block_id, ...
        'payload_len_bytes', payload_len_bytes ...
    );
    out.payload_bytes = payload;
    out.base_coeff_int16 = base_coeff;
    out.enh_coeff_int8 = enh_coeff;
end

function [crc_flag, endian_flag, layer_type, reserved_flag] = unpack_flags5(flags5)
    crc_flag = bitand(bitshift(flags5,-4),1);
    endian_flag = bitand(bitshift(flags5,-3),1);
    layer_type = bitand(bitshift(flags5,-1),3);
    reserved_flag = bitand(flags5,1);
end

function x = unpack_u16_be(b2)
    b2 = uint8(b2);
    x = bitor(bitshift(uint16(b2(1)),8), uint16(b2(2)));
end

function x = unpack_u32_be(b4)
    b4 = uint8(b4);
    x = bitor( bitshift(uint32(b4(1)),24), ...
        bitor( bitshift(uint32(b4(2)),16), ...
        bitor( bitshift(uint32(b4(3)),8), uint32(b4(4)) ) ) );
end

function v_int16 = unpack_int16_vec(bytes, endian_flag)
    bytes = uint8(bytes(:)).';
    assert(mod(numel(bytes),2)==0, "int16 payload must be even bytes");

    if endian_flag == 1
        hi = bytes(1:2:end);
        lo = bytes(2:2:end);
    else
        lo = bytes(1:2:end);
        hi = bytes(2:2:end);
    end
    u = bitor(bitshift(uint16(hi),8), uint16(lo));
    v_int16 = typecast(u, 'int16');
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