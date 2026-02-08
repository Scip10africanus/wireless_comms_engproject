function [image, metadata, status] = bitstream_unpacker(bitstream, params)
% BITSTREAM_UNPACKER - Unpack bitstream and reconstruct image
%
% Handles:
%   - Header parsing (image dimensions, bit depth, format info)
%   - CRC verification (optional)
%   - Bit-to-byte conversion
%   - Image reconstruction from byte stream
%   - Support for grayscale and RGB images
%
% Inputs:
%   bitstream - Received bit vector (column)
%   params    - Structure with fields:
%       .hasHeader   - true if bitstream includes metadata header (default: true)
%       .imageHeight - Image height (required if no header)
%       .imageWidth  - Image width (required if no header)
%       .nChannels   - Number of color channels (required if no header)
%       .bitDepth    - Bits per pixel per channel (default: 8)
%       .hasCRC      - true if CRC is included (default: false)
%
% Outputs:
%   image    - Reconstructed image (uint8)
%   metadata - Structure with parsed header info
%   status   - Structure with unpacking status and error info

    %% Initialize outputs
    status.success = true;
    status.errors = {};
    status.warnings = {};
    metadata = struct();

    %% Default parameters
    if ~isfield(params, 'hasHeader')
        params.hasHeader = true;
    end
    if ~isfield(params, 'bitDepth')
        params.bitDepth = 8;
    end
    if ~isfield(params, 'hasCRC')
        params.hasCRC = false;
    end

    %% Ensure column vector
    bitstream = bitstream(:);
    totalBits = length(bitstream);

    %% 1. Parse header if present
    if params.hasHeader
        [metadata, headerBits, status] = parse_header(bitstream, status);
        if ~status.success
            image = [];
            return;
        end
        % Remove header from bitstream
        bitstream = bitstream(headerBits + 1:end);
    else
        % Use provided parameters
        if ~isfield(params, 'imageHeight') || ~isfield(params, 'imageWidth')
            status.success = false;
            status.errors{end+1} = 'Image dimensions required when no header present';
            image = [];
            return;
        end
        metadata.height = params.imageHeight;
        metadata.width = params.imageWidth;
        metadata.nChannels = getfield_default(params, 'nChannels', 1);
        metadata.bitDepth = params.bitDepth;
    end

    %% 2. CRC verification (if present)
    if params.hasCRC
        [bitstream, crcValid, status] = verify_crc(bitstream, status);
        metadata.crcValid = crcValid;
        if ~crcValid
            status.warnings{end+1} = 'CRC verification failed - image may be corrupted';
        end
    end

    %% 3. Calculate expected data size
    nPixels = metadata.height * metadata.width * metadata.nChannels;
    expectedBits = nPixels * metadata.bitDepth;

    % Check bitstream length
    if length(bitstream) < expectedBits
        status.warnings{end+1} = sprintf('Bitstream too short: expected %d, got %d bits', ...
            expectedBits, length(bitstream));
        % Pad with zeros
        bitstream = [bitstream; zeros(expectedBits - length(bitstream), 1)];
    elseif length(bitstream) > expectedBits
        status.warnings{end+1} = sprintf('Bitstream too long: expected %d, got %d bits (truncating)', ...
            expectedBits, length(bitstream));
        bitstream = bitstream(1:expectedBits);
    end

    %% 4. Convert bits to bytes
    bytes = bits_to_bytes(bitstream, metadata.bitDepth);

    %% 5. Reshape to image
    image = reconstruct_image(bytes, metadata);

    %% 6. Store additional metadata
    metadata.totalBits = totalBits;
    metadata.dataBits = expectedBits;
    metadata.bitErrorEstimate = estimate_bit_errors(image, metadata);
end

%% Header Parsing Functions

function [metadata, headerBits, status] = parse_header(bitstream, status)
% PARSE_HEADER - Extract image metadata from header
%
% Header format (64 bits total):
%   Bits 1-16:  Image height (0-65535)
%   Bits 17-32: Image width (0-65535)
%   Bits 33-36: Number of channels (1-16)
%   Bits 37-40: Bit depth (1-16)
%   Bits 41-48: Compression type (0=none, 1=RLE, etc.)
%   Bits 49-64: Reserved / checksum

    headerBits = 64;
    metadata = struct();

    if length(bitstream) < headerBits
        status.success = false;
        status.errors{end+1} = 'Bitstream too short to contain header';
        return;
    end

    header = bitstream(1:headerBits);

    % Parse fields
    metadata.height = bits_to_uint(header(1:16));
    metadata.width = bits_to_uint(header(17:32));
    metadata.nChannels = bits_to_uint(header(33:36));
    metadata.bitDepth = bits_to_uint(header(37:40));
    metadata.compressionType = bits_to_uint(header(41:48));
    metadata.headerChecksum = bits_to_uint(header(49:64));

    % Validate
    if metadata.height == 0 || metadata.width == 0
        status.success = false;
        status.errors{end+1} = 'Invalid image dimensions in header';
    end
    if metadata.nChannels == 0
        metadata.nChannels = 1;  % Default to grayscale
    end
    if metadata.bitDepth == 0
        metadata.bitDepth = 8;  % Default to 8-bit
    end

    % Verify header checksum (simple XOR of all fields)
    expectedChecksum = bitxor(metadata.height, metadata.width);
    expectedChecksum = bitxor(expectedChecksum, metadata.nChannels);
    expectedChecksum = bitxor(expectedChecksum, metadata.bitDepth);
    if metadata.headerChecksum ~= mod(expectedChecksum, 65536)
        status.warnings{end+1} = 'Header checksum mismatch - header may be corrupted';
    end
end

function val = bits_to_uint(bits)
% BITS_TO_UINT - Convert bit vector to unsigned integer (MSB first)
    bits = bits(:)';
    val = 0;
    for i = 1:length(bits)
        val = val * 2 + bits(i);
    end
end

%% CRC Functions

function [dataStream, valid, status] = verify_crc(bitstream, status)
% VERIFY_CRC - Verify CRC-16 at end of bitstream

    crcBits = 16;

    if length(bitstream) <= crcBits
        valid = false;
        status.warnings{end+1} = 'Bitstream too short for CRC verification';
        dataStream = bitstream;
        return;
    end

    dataStream = bitstream(1:end-crcBits);
    receivedCRC = bits_to_uint(bitstream(end-crcBits+1:end));

    % Calculate CRC-16-CCITT
    calculatedCRC = calculate_crc16(dataStream);

    valid = (receivedCRC == calculatedCRC);
end

function crc = calculate_crc16(data)
% CALCULATE_CRC16 - CRC-16-CCITT implementation

    polynomial = uint16(hex2dec('1021'));
    crc = uint16(hex2dec('FFFF'));

    data = data(:)';

    for i = 1:8:length(data)
        if i + 7 <= length(data)
            byte = bits_to_uint(data(i:i+7));
        else
            byte = bits_to_uint([data(i:end), zeros(1, 8-(length(data)-i+1))]);
        end

        crc = bitxor(crc, bitshift(uint16(byte), 8));
        for j = 1:8
            if bitand(crc, uint16(hex2dec('8000'))) ~= 0
                crc = bitxor(bitshift(crc, 1), polynomial);
            else
                crc = bitshift(crc, 1);
            end
        end
    end

    crc = double(crc);
end

%% Bit-to-Byte Conversion

function bytes = bits_to_bytes(bitstream, bitDepth)
% BITS_TO_BYTES - Convert bit vector to byte vector

    nBytes = floor(length(bitstream) / bitDepth);
    bytes = zeros(nBytes, 1);

    for i = 1:nBytes
        startIdx = (i-1) * bitDepth + 1;
        endIdx = i * bitDepth;
        bytes(i) = bits_to_uint(bitstream(startIdx:endIdx));
    end
end

%% Image Reconstruction

function image = reconstruct_image(bytes, metadata)
% RECONSTRUCT_IMAGE - Reshape byte vector to image matrix

    height = metadata.height;
    width = metadata.width;
    nChannels = metadata.nChannels;

    % Ensure we have enough bytes
    expectedBytes = height * width * nChannels;
    if length(bytes) < expectedBytes
        bytes = [bytes; zeros(expectedBytes - length(bytes), 1)];
    end
    bytes = bytes(1:expectedBytes);

    % Reshape based on channel count
    if nChannels == 1
        % Grayscale
        image = uint8(reshape(bytes, [width, height])');
    else
        % Color (RGB or multi-channel)
        image = zeros(height, width, nChannels, 'uint8');
        for c = 1:nChannels
            channelStart = (c-1) * height * width + 1;
            channelEnd = c * height * width;
            channelData = bytes(channelStart:channelEnd);
            image(:,:,c) = uint8(reshape(channelData, [width, height])');
        end
    end
end

%% Utility Functions

function val = getfield_default(s, field, default)
% GETFIELD_DEFAULT - Get structure field with default value
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

function errorEst = estimate_bit_errors(image, metadata)
% ESTIMATE_BIT_ERRORS - Rough estimate of bit errors based on image statistics

    % Look for obvious corruption patterns
    imgDouble = double(image);

    % Check for abnormal pixel distributions
    errorEst.meanIntensity = mean(imgDouble(:));
    errorEst.stdIntensity = std(imgDouble(:));

    % Count potentially corrupted pixels (extreme values)
    errorEst.minPixels = sum(image(:) == 0) / numel(image);
    errorEst.maxPixels = sum(image(:) == 255) / numel(image);

    % Estimate based on edge detection (corrupted images often have unusual edges)
    if metadata.nChannels == 1
        edges = abs(diff(imgDouble, 1, 1)) + abs(diff(imgDouble, 1, 2));
        errorEst.edgeDensity = mean(edges(:)) / 255;
    else
        grayImg = 0.299*imgDouble(:,:,1) + 0.587*imgDouble(:,:,2) + 0.114*imgDouble(:,:,3);
        edges = abs(diff(grayImg, 1, 1)) + abs(diff(grayImg, 1, 2));
        errorEst.edgeDensity = mean(edges(:)) / 255;
    end
end
