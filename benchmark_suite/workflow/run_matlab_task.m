function run_matlab_task(dataset_name, method, case_name, overwrite)
% Run exactly ONE MATLAB reconstruction in a fresh MATLAB process.
% This is the robust mode used by V2.0 to avoid long-session heap accumulation.
if nargin < 4; overwrite=false; end
here=fileparts(mfilename('fullpath')); root=fileparts(here);
profile=strtrim(fileread(fullfile(root,'artifacts','CURRENT_PROFILE.txt')));
case_mat=fullfile(root,'artifacts','cases',dataset_name,[case_name '.mat']);
if ~isfile(case_mat); error('Case not found: %s',case_mat); end
method_dir=fullfile(root,'artifacts','results',dataset_name,profile,'reconstructions',method);
if ~exist(method_dir,'dir'); mkdir(method_dir); end
out_mat=fullfile(method_dir,[method '__' case_name '.mat']);
partial_mat=fullfile(method_dir,[method '__' case_name '.partial.mat']);
if isfile(out_mat) && ~overwrite
    fprintf('SKIP existing: %s\n',out_mat); return;
end
if isfile(partial_mat); delete(partial_mat); end
addpath(fullfile(root,'Benchmark','adapters','tnn_matlab'));
addpath(fullfile(root,'Benchmark','adapters','kbr_matlab'));
addpath(fullfile(root,'Benchmark','adapters','tctv_matlab'));
addpath(fullfile(root,'Benchmark','adapters','salts'));
addpath(fullfile(root,'Benchmark','adapters','twlrs_matlab'));
tok=regexp(case_name,'__seed(\d+)$','tokens','once'); if isempty(tok); seed=1; else; seed=str2double(tok{1}); end
fprintf('START %s :: %s\n',method,case_name);
switch method
    case 'TNN'
        run_tnn_same_mask(case_mat,fullfile(root,'Methods','TNN'),partial_mat);
    case 'KBR'
        run_kbr_same_mask(case_mat,fullfile(root,'Methods','KBR'),partial_mat);
    case 'TCTV'
        run_tctv_same_mask(case_mat,fullfile(root,'external','Guaranteed-Tensor-Recovery-Fused-Low-rankness-and-Smoothness','TCTV'),partial_mat);
    case 'SALTS'
        run_salts_same_mask(case_mat,fullfile(root,'Methods','SALTS'),partial_mat,seed);
    case 'TWLRS'
        run_twlrs_same_mask(case_mat,fullfile(root,'external','TWLRS'),partial_mat);
    otherwise
        error('Unknown MATLAB method: %s',method);
end
if ~isfile(partial_mat); error('Method returned without output: %s',partial_mat); end
[ok,msg]=movefile(partial_mat,out_mat,'f'); if ~ok; error('Cannot finalize output: %s',msg); end
fprintf('DONE %s :: %s\n',method,case_name);
end
