function varargout = header_codec(mode, varargin)
% HEADER_CODEC - Encode/decode transmission header
%
% Header contains all metadata needed for RX to decode without prior knowledge.
%
% Usage:
%   headerBits = header_codec('encode', params)
%   params = header_codec('decode', headerBits)
%
% Header Format (128 bits = 16 bytes):
% ┌──────────────────────────────────────────────────────────────┐
% │ Field              │ Bits  │ Range        │ Description      │
% ├──────────────────────────────────────────────────────────────┤
% │ Sync Word          │ 16    │ 0xA5A5       │ Frame detection  │
% │ Image Height       │ 12    │ 0-4095       │ Pixels           │
% │ Image Width        │ 12    │ 0-4095       │ Pixels           │
% │ Channels           │ 2     │ 1-4          │ Color channels   │
% │ Wavelet Type       │ 3     │ 0-7          │ See table below  │
% │ Wavelet Levels     │ 3     │ 1-8          │ DWT levels       │
% │ Base Bits/Coeff    │ 4     │ 1-16         │ Bit depth        │
% │ DCT Block Size     │ 2     │ 4,8,16,32    │ Block dimension  │
% │ DCT Coeffs/Block   │ 7     │ 0-127        │ Coeffs kept      │
% │ Enh Bits/Coeff     │ 4     │ 1-16         │ Bit depth        │
% │ Base Mod Order     │ 3     │ 2,4,8,16,... │ Modulation       │
% │ Enh Mod Order      │ 3     │ 2,4,8,16,... │ Modulation       │
% │ Residual Scale     │ 8     │ 0.1-25.5     │ ×0.1 scaling     │
% │ Base Data Length   │ 20    │ 0-1M bits    │ Payload size     │
% │ Enh Data Length    │ 20    │ 0-1M bits    │ Payload size     │
% │ Reserved           │ 3     │ -            │ Future use       │
% │ CRC-8              │ 8     │ -            │ Header checksum  │
% └──────────────────────────────────────────────────────────────┘
% Total: 128 bits
%
% Wavelet Type Codes: 0=haar, 1=db2, 2=db4, 3=db6, 4=bior1.3, 5-7=reserved
% Mod Order Codes: 0=BPSK(2), 1=QPSK(4), 2=8PSK(8), 3=16QAM, 4=32QAM, 5=64QAM, 6=128QAM, 7=256QAM

    switch lower(mode)
        case 'encode'
            varargout{1} = encode_header(varargin{1});
        case 'decode'
            [varargout{1}, varargout{2}] = decode_header(varargin{1});
        case 'size'
            varargout{1} = 128;  % Header size in bits
        otherwise
            error('Unknown mode: %s. Use ''encode'' or ''decode''', mode);
    end
end

%% Encoder

function headerBits = encode_header(params)
% ENCODE_HEADER - Pack parameters into header bitstream

    bits = [];

    % Sync word (16 bits) - 0xA5A5
    bits = [bits, de2bi_msb(hex2dec('A5A5'), 16)];

    % Image height (12 bits)
    bits = [bits, de2bi_msb(min(4095, params.imageHeight), 12)];

    % Image width (12 bits)
    bits = [bits, de2bi_msb(min(4095, params.imageWidth), 12)];

    % Channels (2 bits) - encode as 0-3 for 1-4 channels
    bits = [bits, de2bi_msb(min(3, params.nChannels - 1), 2)];

    % Wavelet type (3 bits)
    waveletCode = wavelet_to_code(params.waveletType);
    bits = [bits, de2bi_msb(waveletCode, 3)];

    % Wavelet levels (3 bits) - encode as 0-7 for 1-8 levels
    bits = [bits, de2bi_msb(min(7, params.waveletLevels - 1), 3)];

    % Base bits per coeff (4 bits) - encode as 0-15 for 1-16 bits
    bits = [bits, de2bi_msb(min(15, params.baseBitsPerCoeff - 1), 4)];

    % DCT block size (2 bits) - 0=4, 1=8, 2=16, 3=32
    dctCode = log2(params.dctBlockSize) - 2;
    bits = [bits, de2bi_msb(min(3, max(0, dctCode)), 2)];

    % DCT coeffs per block (7 bits)
    bits = [bits, de2bi_msb(min(127, params.maxDctCoeffs), 7)];

    % Enhancement bits per coeff (4 bits)
    bits = [bits, de2bi_msb(min(15, params.enhancementBitsPerCoeff - 1), 4)];

    % Base modulation order (3 bits)
    baseModCode = mod_to_code(params.baseMod);
    bits = [bits, de2bi_msb(baseModCode, 3)];

    % Enhancement modulation order (3 bits)
    enhModCode = mod_to_code(params.enhancementMod);
    bits = [bits, de2bi_msb(enhModCode, 3)];

    % Residual scale (8 bits) - stored as value × 10
    scaleVal = round(params.residualScale * 10);
    bits = [bits, de2bi_msb(min(255, max(1, scaleVal)), 8)];

    % Base data length (20 bits)
    baseLen = 0;
    if isfield(params, 'baseDataLength')
        baseLen = params.baseDataLength;
    end
    bits = [bits, de2bi_msb(min(2^20-1, baseLen), 20)];

    % Enhancement data length (20 bits)
    enhLen = 0;
    if isfield(params, 'enhDataLength')
        enhLen = params.enhDataLength;
    end
    bits = [bits, de2bi_msb(min(2^20-1, enhLen), 20)];

    % Reserved (3 bits)
    bits = [bits, 0, 0, 0];

    % CRC-8 (8 bits) - calculated over first 120 bits
    crc = calculate_crc8(bits(1:120));
    bits = [bits, de2bi_msb(crc, 8)];

    headerBits = bits(:);
end

%% Decoder

function [params, valid] = decode_header(headerBits)
% DECODE_HEADER - Unpack header bitstream into parameters

    headerBits = headerBits(:)';
    params = struct();
    valid = true;

    if length(headerBits) < 128
        valid = false;
        params.error = 'Header too short';
        return;
    end

    idx = 1;

    % Sync word (16 bits)
    syncWord = bi2de_msb(headerBits(idx:idx+15));
    idx = idx + 16;
    if syncWord ~= hex2dec('A5A5')
        valid = false;
        params.error = 'Invalid sync word';
        params.syncWord = syncWord;
        return;
    end

    % Image height (12 bits)
    params.imageHeight = bi2de_msb(headerBits(idx:idx+11));
    idx = idx + 12;

    % Image width (12 bits)
    params.imageWidth = bi2de_msb(headerBits(idx:idx+11));
    idx = idx + 12;

    % Channels (2 bits)
    params.nChannels = bi2de_msb(headerBits(idx:idx+1)) + 1;
    idx = idx + 2;

    % Wavelet type (3 bits)
    waveletCode = bi2de_msb(headerBits(idx:idx+2));
    params.waveletType = code_to_wavelet(waveletCode);
    idx = idx + 3;

    % Wavelet levels (3 bits)
    params.waveletLevels = bi2de_msb(headerBits(idx:idx+2)) + 1;
    idx = idx + 3;

    % Base bits per coeff (4 bits)
    params.baseBitsPerCoeff = bi2de_msb(headerBits(idx:idx+3)) + 1;
    idx = idx + 4;

    % DCT block size (2 bits)
    dctCode = bi2de_msb(headerBits(idx:idx+1));
    params.dctBlockSize = 2^(dctCode + 2);
    idx = idx + 2;

    % DCT coeffs per block (7 bits)
    params.maxDctCoeffs = bi2de_msb(headerBits(idx:idx+6));
    idx = idx + 7;

    % Enhancement bits per coeff (4 bits)
    params.enhancementBitsPerCoeff = bi2de_msb(headerBits(idx:idx+3)) + 1;
    idx = idx + 4;

    % Base modulation order (3 bits)
    baseModCode = bi2de_msb(headerBits(idx:idx+2));
    params.baseMod = code_to_mod(baseModCode);
    idx = idx + 3;

    % Enhancement modulation order (3 bits)
    enhModCode = bi2de_msb(headerBits(idx:idx+2));
    params.enhancementMod = code_to_mod(enhModCode);
    idx = idx + 3;

    % Residual scale (8 bits)
    scaleVal = bi2de_msb(headerBits(idx:idx+7));
    params.residualScale = scaleVal / 10;
    idx = idx + 8;

    % Base data length (20 bits)
    params.baseDataLength = bi2de_msb(headerBits(idx:idx+19));
    idx = idx + 20;

    % Enhancement data length (20 bits)
    params.enhDataLength = bi2de_msb(headerBits(idx:idx+19));
    idx = idx + 20;

    % Reserved (3 bits) - skip
    idx = idx + 3;

    % CRC-8 (8 bits)
    receivedCRC = bi2de_msb(headerBits(idx:idx+7));
    calculatedCRC = calculate_crc8(headerBits(1:120));

    if receivedCRC ~= calculatedCRC
        valid = false;
        params.warning = 'CRC mismatch - header may be corrupted';
        params.receivedCRC = receivedCRC;
        params.calculatedCRC = calculatedCRC;
    end

    params.headerValid = valid;
end

%% Lookup Tables

function code = wavelet_to_code(waveletType)
    switch lower(waveletType)
        case 'haar',    code = 0;
        case 'db2',     code = 1;
        case 'db4',     code = 2;
        case 'db6',     code = 3;
        case 'bior1.3', code = 4;
        otherwise,      code = 0;
    end
end

function waveletType = code_to_wavelet(code)
    types = {'haar', 'db2', 'db4', 'db6', 'bior1.3', 'haar', 'haar', 'haar'};
    waveletType = types{code + 1};
end

function code = mod_to_code(modOrder)
    switch modOrder
        case 2,   code = 0;  % BPSK
        case 4,   code = 1;  % QPSK
        case 8,   code = 2;  % 8-PSK
        case 16,  code = 3;  % 16-QAM
        case 32,  code = 4;  % 32-QAM
        case 64,  code = 5;  % 64-QAM
        case 128, code = 6;  % 128-QAM
        case 256, code = 7;  % 256-QAM
        otherwise, code = 1; % Default QPSK
    end
end

function modOrder = code_to_mod(code)
    orders = [2, 4, 8, 16, 32, 64, 128, 256];
    modOrder = orders(code + 1);
end

%% Bit Conversion Utilities

function bits = de2bi_msb(val, nBits)
% DE2BI_MSB - Decimal to binary, MSB first
    bits = zeros(1, nBits);
    for i = nBits:-1:1
        bits(nBits - i + 1) = mod(val, 2);
        val = floor(val / 2);
    end
end

function val = bi2de_msb(bits)
% BI2DE_MSB - Binary to decimal, MSB first
    val = 0;
    for i = 1:length(bits)
        val = val * 2 + bits(i);
    end
end

%% CRC-8

function crc = calculate_crc8(bits)
% CALCULATE_CRC8 - CRC-8-CCITT
    poly = uint8(0x07);
    crc = uint8(0);

    % Process byte by byte
    nBytes = floor(length(bits) / 8);
    for b = 1:nBytes
        byteVal = uint8(bi2de_msb(bits((b-1)*8+1 : b*8)));
        crc = bitxor(crc, byteVal);
        for i = 1:8
            if bitand(crc, uint8(128)) ~= 0
                crc = bitxor(bitshift(crc, 1), poly);
            else
                crc = bitshift(crc, 1);
            end
        end
    end
    crc = double(crc);
end

%% Test Function

function test_header_codec()
% TEST_HEADER_CODEC - Verify encode/decode roundtrip

    params = struct();
    params.imageHeight = 256;
    params.imageWidth = 256;
    params.nChannels = 3;
    params.waveletType = 'db2';
    params.waveletLevels = 3;
    params.baseBitsPerCoeff = 10;
    params.dctBlockSize = 8;
    params.maxDctCoeffs = 32;
    params.enhancementBitsPerCoeff = 8;
    params.baseMod = 4;
    params.enhancementMod = 64;
    params.residualScale = 1.5;
    params.baseDataLength = 50000;
    params.enhDataLength = 200000;

    % Encode
    headerBits = header_codec('encode', params);
    fprintf('Header size: %d bits\n', length(headerBits));

    % Decode
    [decoded, valid] = header_codec('decode', headerBits);

    % Verify
    fprintf('Valid: %d\n', valid);
    fprintf('Height: %d -> %d\n', params.imageHeight, decoded.imageHeight);
    fprintf('Width: %d -> %d\n', params.imageWidth, decoded.imageWidth);
    fprintf('Wavelet: %s -> %s\n', params.waveletType, decoded.waveletType);
    fprintf('Base mod: %d -> %d\n', params.baseMod, decoded.baseMod);
    fprintf('Enh mod: %d -> %d\n', params.enhancementMod, decoded.enhancementMod);
end
