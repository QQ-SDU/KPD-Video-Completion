function run_salts_same_mask(case_mat, repo_dir, out_mat, seed)
% Same-mask bridge for the official SALTS demo supplied by the user.
% Ground truth in case_mat is NOT used by SALTS.
% The official real-data demo evaluates abs(YY), so the saved reconstruction
% is converted with abs() before evaluation.

if nargin < 4
    seed = 1;
end
addpath(genpath(repo_dir));
S = load(case_mat, 'observed', 'mask');
data = double(S.observed);
Omega = logical(S.mask);

% Same convention as the authors' real-data demo.
R = size(data);
lambda = [1 1 1];

rng(seed,'twister');
tic;
[YY, ~, ~, ~, errlist] = TC_LTNND33(data, R, lambda, Omega);
runtime_seconds = toc;

recon = min(max(abs(YY),0),1);
recon(Omega)=data(Omega);
save(out_mat, 'recon', 'runtime_seconds', 'R', 'lambda', 'seed', 'errlist', '-v7');
fprintf('SALTS done: %.3f s\n', runtime_seconds);
end
