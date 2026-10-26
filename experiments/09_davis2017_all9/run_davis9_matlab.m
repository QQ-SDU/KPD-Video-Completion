function run_davis9_matlab(method, case_mat, out_mat, seed, benchroot)
% Run official-method same-mask MATLAB wrappers. No test GT is accessed.
addpath(genpath(fullfile(benchroot,'Benchmark','adapters','tnn_matlab')));
addpath(genpath(fullfile(benchroot,'Benchmark','adapters','kbr_matlab')));
addpath(genpath(fullfile(benchroot,'Benchmark','adapters','salts')));
if ~isfile(case_mat), error('Missing case %s',case_mat); end
if ~isfolder(fileparts(out_mat)), mkdir(fileparts(out_mat)); end
switch method
    case 'TNN'
        run_tnn_same_mask(case_mat,fullfile(benchroot,'Methods','TNN'),out_mat);
    case 'KBR'
        run_kbr_same_mask(case_mat,fullfile(benchroot,'Methods','KBR'),out_mat);
    case 'SALTS'
        run_salts_same_mask(case_mat,fullfile(benchroot,'Methods','SALTS'),out_mat,seed);
    otherwise
        error('Unsupported MATLAB method %s',method);
end
end
