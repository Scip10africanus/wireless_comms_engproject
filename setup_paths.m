%% SETUP_PATHS - Add all project directories to MATLAB path
%
% Run this script before using any project functions:
%   >> setup_paths
%
% Project Structure:
%   rx/       - Receiver modules (demodulation, decoding, quality)
%   tx/       - Transmitter modules (encoding, modulation)
%   channel/  - Channel models (AWGN, Rayleigh, etc.)
%   common/   - Shared utilities (header codec, subcarrier mapping)
%   config/   - Configuration files
%   tests/    - Test scripts

projectRoot = fileparts(mfilename('fullpath'));

addpath(fullfile(projectRoot, 'rx'));
addpath(fullfile(projectRoot, 'tx'));
addpath(fullfile(projectRoot, 'tx', 'src'));
if isfolder(fullfile(projectRoot, 'tx_old'))
    addpath(fullfile(projectRoot, 'tx_old'));
end
addpath(fullfile(projectRoot, 'channel'));
addpath(fullfile(projectRoot, 'common'));
addpath(fullfile(projectRoot, 'config'));
addpath(fullfile(projectRoot, 'tests'));
addpath(fullfile(projectRoot, 'examples'));
addpath(fullfile(projectRoot, 'ml_addons'));

fprintf('Project paths added:\n');
fprintf('  ✓ rx/        - Receiver modules\n');
fprintf('  ✓ tx/        - Transmitter modules\n');
fprintf('  ✓ tx/src/    - Transmitter source functions\n');
fprintf('  ✓ channel/   - Channel models\n');
fprintf('  ✓ common/    - Shared utilities\n');
fprintf('  ✓ config/    - Configuration files\n');
fprintf('  ✓ tests/     - Test scripts\n');
fprintf('  ✓ examples/  - Example scripts\n');
fprintf('  ✓ ml_addons/ - ML add-on modules\n');
fprintf('\nReady to use. Try: test_closed_loop\n');
