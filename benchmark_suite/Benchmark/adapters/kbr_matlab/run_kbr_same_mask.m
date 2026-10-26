function run_kbr_same_mask(case_mat, repo_dir, out_mat)
% Same-mask bridge for official KBR-TC code.
% Uses the parameter defaults from the authors' Demo_TC_MSI.m.
% File I/O is excluded from timing.

addpath(genpath(repo_dir));
S = load(case_mat, 'observed', 'mask');
observed = double(S.observed);
Omega = logical(S.mask);

Par2.tol = 1e-5;
Par2.maxIter = 1000;
Par2.maxSubiter = 1;
Par2.rho = 1.05;
Par2.mu = 1e-5;      % authors' beta(1e-2) * 1e-3
Par2.lambda = 0.1;

tic;
recon = KBR_TC(observed, Omega, Par2);
runtime_seconds = toc;
recon=min(max(double(recon),0),1);
recon(Omega)=observed(Omega);

save(out_mat, 'recon', 'runtime_seconds', 'Par2', '-v7');
fprintf('KBR done: %.3f s\n', runtime_seconds);
end
