function run_tnn_same_mask(case_mat, repo_dir, out_mat)
% Same-mask bridge for Canyi Lu's official TNN tensor-completion code.
% File I/O is excluded from timing.

addpath(genpath(repo_dir));
S = load(case_mat, 'observed', 'mask');
M = double(S.observed);
mask = logical(S.mask);
omega = find(mask);

opts = struct();
opts.tol = 1e-8;
opts.max_iter = 500;
opts.rho = 1.1;
opts.mu = 1e-4;
opts.max_mu = 1e10;
opts.DEBUG = 0;

tic;
[Xhat, obj, err, iter] = lrtc_tnn(M, omega, opts);
runtime_seconds = toc;

recon = min(max(double(Xhat),0),1);
recon(mask)=M(mask);
save(out_mat, 'recon', 'runtime_seconds', 'iter', 'obj', 'err', '-v7');
fprintf('TNN done: %.3f s, iter=%d\n', runtime_seconds, iter);
end
