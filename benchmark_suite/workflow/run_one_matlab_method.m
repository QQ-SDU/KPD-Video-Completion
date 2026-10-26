function run_one_matlab_method(dataset_name,method)
here=fileparts(mfilename('fullpath')); root=fileparts(here);
profile=strtrim(fileread(fullfile(root,'artifacts','CURRENT_PROFILE.txt')));
case_dir=fullfile(root,'artifacts','cases',dataset_name);
result_root=fullfile(root,'artifacts','results',dataset_name,profile,'reconstructions',method);
if ~exist(result_root,'dir'), mkdir(result_root); end
addpath(fullfile(root,'Benchmark','adapters','tnn_matlab'));
addpath(fullfile(root,'Benchmark','adapters','kbr_matlab'));
addpath(fullfile(root,'Benchmark','adapters','tctv_matlab'));
addpath(fullfile(root,'Benchmark','adapters','salts'));
addpath(fullfile(root,'Benchmark','adapters','twlrs_matlab'));
cases=dir(fullfile(case_dir,'*.mat'));
for ci=1:numel(cases)
    case_mat=fullfile(cases(ci).folder,cases(ci).name); [~,case_name,~]=fileparts(case_mat);
    out_mat=fullfile(result_root,[method '__' case_name '.mat']); if isfile(out_mat), fprintf('SKIP %s\n',out_mat); continue; end
    tok=regexp(case_name,'__seed(\d+)$','tokens','once'); if isempty(tok), seed=1; else, seed=str2double(tok{1}); end
    switch method
        case 'TNN', run_tnn_same_mask(case_mat,fullfile(root,'Methods','TNN'),out_mat);
        case 'KBR', run_kbr_same_mask(case_mat,fullfile(root,'Methods','KBR'),out_mat);
        case 'SALTS', run_salts_same_mask(case_mat,fullfile(root,'Methods','SALTS'),out_mat,seed);
        otherwise, error('Unknown method %s',method);
    end
end
end
