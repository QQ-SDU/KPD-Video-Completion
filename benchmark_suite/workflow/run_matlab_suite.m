function run_matlab_suite(dataset_name, overwrite)
% Robust MATLAB benchmark runner for:
% Final paper MATLAB methods: TNN, KBR, SALTS.
%
% R2022a compatibility fix:
% The previous runner used a non-MATLAB file-flush call that could terminate
% MATLAB after the first completed method.  This version
% never keeps the status file open across a method.  Each status/event row
% is written with fopen(...,'a') -> fprintf -> fclose, so it is flushed
% immediately and is compatible with MATLAB R2022a.
%
% Existing final reconstruction files are skipped automatically.

if nargin < 2
    overwrite = false;
end

here = fileparts(mfilename('fullpath'));
root = fileparts(here);

profile_file = fullfile(root,'artifacts','CURRENT_PROFILE.txt');
if ~isfile(profile_file)
    error('CURRENT_PROFILE.txt missing. Run workflow/sync_profile.py first.');
end
profile = strtrim(fileread(profile_file));

case_dir = fullfile(root,'artifacts','cases',dataset_name);
result_root = fullfile(root,'artifacts','results',dataset_name,profile,'reconstructions');
log_dir = fullfile(root,'artifacts','logs',dataset_name,profile);

if ~exist(result_root,'dir'), mkdir(result_root); end
if ~exist(log_dir,'dir'), mkdir(log_dir); end

addpath(fullfile(root,'Benchmark','adapters','tnn_matlab'));
addpath(fullfile(root,'Benchmark','adapters','kbr_matlab'));
addpath(fullfile(root,'Benchmark','adapters','tctv_matlab'));
addpath(fullfile(root,'Benchmark','adapters','salts'));
addpath(fullfile(root,'Benchmark','adapters','twlrs_matlab'));

methods = {'TNN','KBR','SALTS'};
cases = dir(fullfile(case_dir,'*.mat'));
if isempty(cases)
    error('No MAT cases in %s. Run prepare_cases.py first.', case_dir);
end

status_path = fullfile(log_dir,'matlab_status.csv');
event_path  = fullfile(log_dir,'matlab_events.csv');

if ~isfile(status_path)
    local_append_line(status_path, ...
        'timestamp,method,case,status,returncode,wall_seconds,output');
end
if ~isfile(event_path)
    local_append_line(event_path, ...
        'timestamp,method,case,event,output');
end

total = numel(methods) * numel(cases);
job = 0;

for mi = 1:numel(methods)
    method = methods{mi};
    method_dir = fullfile(result_root,method);
    if ~exist(method_dir,'dir'), mkdir(method_dir); end

    for ci = 1:numel(cases)
        job = job + 1;

        case_mat = fullfile(cases(ci).folder,cases(ci).name);
        [~,case_name,~] = fileparts(case_mat);
        out_mat = fullfile(method_dir,[method '__' case_name '.mat']);

        if isfile(out_mat) && ~overwrite
            fprintf('[%d/%d] SKIP %s :: %s\n',job,total,method,case_name);
            local_append_csv(status_path, datestr(now,31), method, case_name, ...
                'SKIP', 0, 0, out_mat);
            local_append_event(event_path, datestr(now,31), method, case_name, ...
                'SKIP_EXISTING', out_mat);
            continue;
        end

        tok = regexp(case_name,'__seed(\d+)$','tokens','once');
        if isempty(tok)
            seed = 1;
        else
            seed = str2double(tok{1});
        end

        fprintf('\n============================================================\n');
        fprintf('[%d/%d] RUN %s :: %s\n',job,total,method,case_name);
        fprintf('============================================================\n');

        local_append_event(event_path, datestr(now,31), method, case_name, ...
            'START', out_mat);

        wall = tic;
        rc = 0;
        st = 'DONE';

        try
            switch method
                case 'TNN'
                    run_tnn_same_mask(case_mat,fullfile(root,'Methods','TNN'),out_mat);

                case 'KBR'
                    run_kbr_same_mask(case_mat,fullfile(root,'Methods','KBR'),out_mat);

                case 'SALTS'
                    run_salts_same_mask(case_mat,fullfile(root,'Methods','SALTS'),out_mat,seed);

                otherwise
                    error('Unknown MATLAB method: %s',method);
            end

        catch ME
            rc = 1;
            st = 'FAILED';
            fprintf(2,'[FAILED] %s :: %s\n%s\n', ...
                method,case_name,getReport(ME,'extended','hyperlinks','off'));
        end

        elapsed = toc(wall);

        local_append_csv(status_path, datestr(now,31), method, case_name, ...
            st, rc, elapsed, out_mat);
        local_append_event(event_path, datestr(now,31), method, case_name, ...
            st, out_mat);

        fprintf('[%s] %s :: %s | wall=%.3f s\n',st,method,case_name,elapsed);
    end
end

fprintf('\nMATLAB suite finished for dataset=%s\n',dataset_name);
end

function local_append_line(pathname, line)
fid = fopen(pathname,'a');
if fid < 0
    error('Cannot open log file: %s', pathname);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',line);
end

function local_append_csv(pathname, timestamp, method, case_name, status, rc, wall_seconds, output)
fid = fopen(pathname,'a');
if fid < 0
    error('Cannot open status file: %s', pathname);
end
cleanupObj = onCleanup(@() fclose(fid));
output = strrep(output,'"','""');
fprintf(fid,'%s,%s,%s,%s,%d,%.6f,"%s"\n', ...
    timestamp,method,case_name,status,rc,wall_seconds,output);
end

function local_append_event(pathname, timestamp, method, case_name, event_name, output)
fid = fopen(pathname,'a');
if fid < 0
    error('Cannot open event file: %s', pathname);
end
cleanupObj = onCleanup(@() fclose(fid));
output = strrep(output,'"','""');
fprintf(fid,'%s,%s,%s,%s,"%s"\n', ...
    timestamp,method,case_name,event_name,output);
end
