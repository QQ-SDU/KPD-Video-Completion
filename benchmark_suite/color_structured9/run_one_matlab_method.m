function run_one_matlab_method(method_name, case_mat, out_mat, seed)
% Run one MATLAB baseline on one channel case.
% Methods supported here: TNN, KBR, SALTS.

if nargin < 4
    seed = 1;
end

here = fileparts(mfilename('fullpath'));
root = fileparts(here);

addpath(fullfile(root,'Benchmark','adapters','tnn_matlab'));
addpath(fullfile(root,'Benchmark','adapters','kbr_matlab'));
addpath(fullfile(root,'Benchmark','adapters','salts'));

switch upper(strtrim(method_name))
    case 'TNN'
        run_tnn_same_mask(case_mat, fullfile(root,'Methods','TNN'), out_mat);
    case 'KBR'
        run_kbr_same_mask(case_mat, fullfile(root,'Methods','KBR'), out_mat);
    case 'SALTS'
        run_salts_same_mask(case_mat, fullfile(root,'Methods','SALTS'), out_mat, seed);
    otherwise
        error('Unsupported MATLAB method: %s', method_name);
end
end
