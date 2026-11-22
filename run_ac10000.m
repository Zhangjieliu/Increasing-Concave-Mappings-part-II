function results = run_ac10000()
% IEEE123-derived 10,000-bus AC benchmark.
% Input: ieee10000.mat, in the same folder as this function.
% Compare CPF, ICM bisection, and eight-worker ICM nine-section search.

folder = fileparts(mfilename('fullpath'));
S = load(fullfile(folder, 'ieee10000.mat'), 'mpc');
mpc = S.mpc;

assert(size(mpc.bus,1) == 10000, ...
    'This benchmark requires exactly 10,000 buses.');

M = icm_radial_model(mpc);

opt = struct( ...
    'alpha_tol', 1e-4, ...
    'pf_tol',    1e-7, ...
    'map_tol',   1e-9, ...
    'sign_tol',  1e-9, ...
    'maxit',     100);

fprintf('%s | AC | 10000 buses | log-current ICM\n', M.name);

% Start an eight-worker thread pool.
tick = tic;
pool = gcp('nocreate');

if isempty(pool)
    pool = parpool('Threads', 8);
else
    assert(pool.NumWorkers == 8 && isa(pool,'parallel.ThreadPool'), ...
        'Use an eight-worker thread pool for this benchmark.');
end

pool_seconds = toc(tick);

% Warm up each method before timing.
run = @(k) run_method(k, M, opt);

for k = 1:3
    run(k);
end

% Repeat three times and rotate the execution order.
seconds = zeros(3,3);
answers = cell(3,3);

for rep = 1:3
    for k = mod((0:2) + rep - 1, 3) + 1
        tick = tic;
        answers{k,rep} = run(k);
        seconds(k,rep) = toc(tick);
    end

    fprintf(['Repeat %d: CPF %.6f s | ICM bisection %.6f s | ' ...
        'ICM nine-section %.6f s\n'], ...
        rep, seconds(1,rep), seconds(2,rep), seconds(3,rep));
end

% Independently refine the physical fold outside the timed runs.
audit_opt = opt;
audit_opt.audit = true;

reference = ac_cpf(M, audit_opt);
alpha_ref = reference.reference_alpha;

for k = 1:numel(answers)
    a = answers{k};

    assert(abs(a.alpha - alpha_ref) <= opt.alpha_tol, ...
        'Loading-factor mismatch.');

    assert(a.residual <= opt.pf_tol, ...
        'Original AC power mismatch is too large.');

    if isfield(a, 'upper')
        assert(a.alpha <= alpha_ref + 1e-8 && ...
            alpha_ref <= a.upper + 1e-8, ...
            'The ICM interval does not contain the fold reference.');
    end
end

results = struct( ...
    'case_name',    M.name, ...
    'options',      opt, ...
    'answers',      {answers}, ...
    'raw_seconds',  seconds, ...
    'seconds',      median(seconds,2), ...
    'pool_seconds', pool_seconds, ...
    'reference',    alpha_ref, ...
    'bus_ids',      M.bus_ids);

results.matlab_version = version;
results.pool_type = class(pool);
results.workers = pool.NumWorkers;

fprintf(['Median: CPF %.6f s | ICM bisection %.6f s | ' ...
    'ICM nine-section %.6f s\n'], results.seconds);

fprintf('Maximum loading factor: %.10f | tolerance %.1e\n', ...
    alpha_ref, opt.alpha_tol);
end


function out = run_method(k, M, opt)
if k == 1
    out = ac_cpf(M, opt);
else
    out = icm_search(M, opt, k == 3);
end
end


function out = icm_search(M, opt, use_parallel)
% Every trial uses an independently constructed initial point.

a = 0;
b = 1;
best = icm_point(M, 0, opt);
updates = 0;
calls = 0;

% Find a feasible-infeasible bracket.
for k = 1:40
    upper = icm_point(M, b, opt);
    require_decided(upper);

    updates = updates + upper.updates;
    calls = calls + 1;

    if upper.status == -1
        break
    end

    a = b;
    best = upper;
    b = 2*b;
end

assert(upper.status == -1, 'ICM:Bracket', ...
    'No excluded upper endpoint was found.');

history = zeros(0,4);

while b - a > opt.alpha_tol
    old = [a,b];

    if use_parallel
        alpha = a + (1:8)*(b-a)/9;
        trial = cell(1,8);

        parfor j = 1:8
            trial{j} = icm_point(M, alpha(j), opt);
        end
    else
        alpha = (a+b)/2;
        trial = {icm_point(M, alpha, opt)};
    end

    for j = 1:numel(trial)
        require_decided(trial{j});
    end

    solved = cellfun(@(v) v.status == 1, trial);

    assert(~any(diff(double(solved)) > 0), 'ICM:Order', ...
        'Feasibility classifications are inconsistent along this path.');

    j = sum(solved);

    if j > 0
        a = alpha(j);
        best = trial{j};
    end

    if j < numel(alpha)
        b = alpha(j+1);
        upper = trial{j+1};
    end

    updates = updates + sum(cellfun(@(v) v.updates, trial));
    calls = calls + numel(trial);
    history(end+1,:) = [old,a,b]; %#ok<AGROW>
end

out = struct( ...
    'alpha',        a, ...
    'upper',        b, ...
    'V',            best.V, ...
    'residual',     best.residual, ...
    'updates',      updates, ...
    'calls',        calls, ...
    'rounds',       size(history,1), ...
    'history',      history, ...
    'upper_reason', upper.reason);
end


function require_decided(v)
% An iteration limit is not an infeasibility certificate.
if v.status == 0
    error('ICM:Unresolved', ...
        ['alpha=%.12g remains unresolved: %s. ' ...
         'The bracket was not updated.'], ...
        v.alpha, v.reason);
end
end


function out = icm_point(M, alpha, opt)
% Exact Newton iteration for the class-3 log-current ICM.
% Variable: z = -log(ell).
% Status: 1 solved; -1 excluded; 0 unresolved.

assert(isscalar(alpha) && isfinite(alpha) && alpha >= 0, ...
    'Invalid loading factor.');

out = struct( ...
    'status',   0, ...
    'V',        [], ...
    'residual', Inf, ...
    'updates',  0, ...
    'reason',   'iteration limit', ...
    'history',  zeros(0,3), ...
    'alpha',    alpha);

n = M.n;
A = M.A;
AT = M.AT;
r = M.r;
x = M.x;

if alpha == 0
    out.status = 1;
    out.V = M.V0*ones(n,1);
    out.residual = 0;
    out.reason = 'solved';
    return
end

d = abs(M.V0)^2 + alpha*M.dlam;

if any(d <= 0)
    out.status = -1;
    out.reason = 'nonpositive analytic voltage bound';
    return
end

c = alpha^2*(M.pbar.^2 + M.qbar.^2);

if any(~isfinite(c) | c <= 0) || any(~isfinite(d))
    out.reason = 'invalid logarithmic data';
    return
end

% Construct an analytic upper initial point independently at each alpha.
z = (1 + norm(log(d)-log(c),inf))*ones(n,1);

D = @(v) spdiags(v,0,n,n);
O = sparse(n,n);
z2 = r.^2 + x.^2;
history = zeros(opt.maxit+1,3);

for it = 0:opt.maxit
    ell = exp(-z);

    ur = A\(r.*ell);
    ux = A\(x.*ell);

    P = alpha*M.pbar + ur - r.*ell;
    Q = alpha*M.qbar + ux - x.*ell;

    s = AT\(2*r.*ur + 2*x.*ux - z2.*ell);
    h = P.^2 + Q.^2 + ell.*s;

    if any(~isfinite([z;ell;s;h])) || any(ell <= 0 | h <= 0)
        out.reason = 'nonfinite or underflowed iterate';
        return
    end

    R = z - log(d) + log(h);

    history(it+1,:) = [norm(R,inf), min(R), NaN];
    out.history = history(1:it+1,:);

    if min(R) < -opt.sign_tol
        out.reason = 'residual sign lost numerically';
        return
    end

    if any(d-s <= -opt.sign_tol)
        out.status = -1;
        out.reason = 'nonpositive voltage upper bound';
        return
    end

    if norm(R,inf) <= opt.map_tol && all(d-s > 0)
        [V,residual] = recover_voltage(M,alpha,ell,P,Q,d-s);
        out.residual = residual;

        if isfinite(residual) && residual <= opt.pf_tol
            out.status = 1;
            out.V = V;
            out.reason = 'solved';
            return
        end
    end

    if it == opt.maxit
        return
    end

    % Sparse augmented system for the exact ICM Newton correction.
    B = [ ...
        D(1 + ell.*(2*P.*r + 2*Q.*x - s)./h), ...
        D(2*P./h), D(2*Q./h), D(ell./h); ...
        D(r.*ell), A, O, O; ...
        D(x.*ell), O, A, O; ...
        D(-z2.*ell), D(-2*r), D(-2*x), AT];

    step = B\[R;zeros(3*n,1)];
    delta = step(1:n);
    out.updates = out.updates + 1;

    if any(~isfinite(step))
        out.reason = 'nonfinite Newton correction';
        return
    end

    % Check the reduced Newton equation without forming a dense Jacobian.
    e = -ell.*delta;
    dr = A\(r.*e);
    dx = A\(x.*e);
    ds = AT\(2*r.*dr + 2*x.*dx - z2.*e);

    Jdelta = delta + ...
        (2*P.*(dr-r.*e) + 2*Q.*(dx-x.*e) + ...
         s.*e + ell.*ds)./h;

    linerr = norm(Jdelta-R,inf);

    if ~isfinite(linerr) || linerr > 1e-8*max(1,norm(R,inf))
        out.reason = 'inaccurate Newton linear solve';
        return
    end

    history(it+1,3) = min(delta);
    out.history = history(1:it+1,:);

    if min(delta) < -opt.sign_tol
        margin = 128*eps*max([1,norm(z,inf),norm(R,inf)]);

        if min(R) > margin && linerr <= 0.25*min(R)
            out.status = -1;
            out.reason = 'ICM monotonicity violation';
        else
            out.reason = 'monotonicity test is numerically inconclusive';
        end

        return
    end

    z = z - delta;
end
end


function [V,residual] = recover_voltage(M,alpha,ell,P,Q,v2)
% Recover complex voltages and check the original AC equations.

allv2 = [abs(M.V0)^2;v2];

Ps = P + M.r.*ell;
Qs = Q + M.x.*ell;

drop = angle(allv2(M.parent+1) - ...
    (M.r+1i*M.x).*(Ps-1i*Qs));

theta = angle(M.V0) + M.AT\drop;
allV = [M.V0;sqrt(v2).*exp(1i*theta)];

edge = (allV(M.parent+1)-allV(2:end))./(M.r+1i*M.x);

I = accumarray(M.parent+1,real(edge),[M.n+1,1],@sum,0) ...
    + 1i*accumarray(M.parent+1,imag(edge),[M.n+1,1],@sum,0);

I(2:end) = I(2:end)-edge;
V = allV(2:end);

residual = norm( ...
    V.*conj(I(2:end)) + alpha*(M.p+1i*M.q), inf);
end


function out = ac_cpf(M,opt)
% Sparse pseudo-arclength continuation for the original AC equations.

n = M.n;

if isfield(M,'Y') && isfield(M,'b')
    W.Y = M.Y;
    W.b = M.b;
else
    y = 1./(M.r+1i*M.x);
    W.Y = M.A*spdiags(y,0,n,n)*M.A.';
    W.b = -M.V0*(M.A*(y.*(M.parent==0)));
end

W.p = [M.p;M.q];
W.x0 = [real(M.V0)*ones(n,1);imag(M.V0)*ones(n,1)];

[alpha,x,updates,q] = continuation(W,opt.pf_tol);

V = x(1:n)+1i*x(n+1:end);
residual = norm( ...
    V.*conj(W.Y*V+W.b) + alpha*(M.p+1i*M.q), inf);

assert(residual <= opt.pf_tol, ...
    'CPF power mismatch exceeds pf_tol.');

out = struct( ...
    'alpha',      alpha, ...
    'V',          V, ...
    'residual',   residual, ...
    'updates',    updates, ...
    'state',      x, ...
    'nullvector', q);

if isfield(opt,'audit') && opt.audit
    [reference,fold_residual,audit_updates] = ...
        fold_reference(x,alpha,q,W);

    out.reference_alpha = reference;
    out.alpha_error = abs(alpha-reference);
    out.fold_residual = fold_residual;
    out.audit_updates = audit_updates;

    assert(out.alpha_error <= opt.alpha_tol, ...
        'CPF load-limit error exceeds alpha_tol.');
end
end


function [g,J] = power_flow(x,alpha,W)
% AC power mismatch and sparse rectangular-coordinate Jacobian.

n = size(W.Y,1);
V = x(1:n)+1i*x(n+1:end);

current = W.Y*V+W.b;
power = V.*conj(current);
g = [real(power);imag(power)] + alpha*W.p;

if nargout < 2
    return
end

D = spdiags(conj(current),0,n,n);
K = spdiags(V,0,n,n)*conj(W.Y);

Je = D+K;
Jf = 1i*(D-K);

J = [real(Je),real(Jf);imag(Je),imag(Jf)];
end


function [alpha,x,updates,q] = continuation(W,tol)
% Follow the solution curve and locate a zero loading-factor tangent.

x = W.x0;
m = numel(x);
z = [x;0];

updates = 0;
p = sparse(W.p);
rhs = [zeros(m,1);1];

[f,J] = power_flow(x,0,W);

assert(norm(f(1:m/2)+1i*f(m/2+1:end),inf) <= tol, ...
    'The zero-load initial power flow did not converge.');

t = [J,p;sparse(1,m),1]\rhs;
t = t/norm(t);

hmax = 2*max(1,sqrt(m/1000));
h = min(hmax,1/t(end));

nose = false;
midpoint = false;
found = false;

for j = 1:3000
    if ~nose
        anchor = z;
        normal = t;
        s = h;
        predictor = z+h*t;
        z1 = predictor;
    else
        s = (left*wb-right*wa)/(wb-wa);

        if midpoint || ~(s > left && s < right)
            s = (left+right)/2;
        end

        fraction = (s-left)/(right-left);
        z1 = (1-fraction)*za + fraction*zb;
    end

    % Newton corrector.
    ok = false;

    for k = 0:10
        if any(~isfinite(z1)) || z1(end) < 0
            break
        end

        [f,J] = power_flow(z1(1:m),z1(end),W);
        g = [f;normal.'*(z1-anchor)-s];

        mismatch = norm(f(1:m/2)+1i*f(m/2+1:end),inf);

        if max(mismatch,abs(g(end))) <= tol
            ok = true;
            break
        end

        if k == 10
            break
        end

        z1 = z1-[J,p;sparse(normal.')]\g;
        updates = updates+1;
    end

    if ~ok
        if nose
            if midpoint
                error('CPF fold correction did not converge.');
            end
            midpoint = true;
        else
            h = h/2;

            if h < 1e-7
                error('CPF step is too small.');
            end
        end

        continue
    end

    % Update the normalized tangent.
    tn = [J,p;sparse(normal.')]\rhs;

    if any(~isfinite(tn)) || norm(tn) == 0
        error('Invalid CPF tangent.');
    end

    tn = tn/norm(tn);

    if tn.'*normal < 0
        tn = -tn;
    end

    if nose
        f = tn(end);
        event_steps = event_steps+1;

        if abs(f) <= 1e-6
            z = z1;
            found = true;
            break
        end

        if event_steps >= 80
            error('CPF fold refinement exceeded its iteration limit.');
        end

        if f > 0
            za = z1;
            left = s;
            fa = f;
            wa = f;

            if side == 1
                wb = wb/2;
            else
                wb = fb;
            end

            side = 1;
        else
            zb = z1;
            right = s;
            fb = f;
            wb = f;

            if side == -1
                wa = wa/2;
            else
                wa = fa;
            end

            side = -1;
        end

        midpoint = false;

    elseif tn(end) <= 0
        if tn(end) == 0
            z = z1;
            found = true;
            break
        end

        za = z;
        zb = z1;
        anchor = za;

        normal = zb-za;
        normal = normal/norm(normal);

        if min(t.'*normal,tn.'*normal) <= 0
            h = h/2;

            if h < 1e-7
                error('CPF step is too small.');
            end

            continue
        end

        left = 0;
        right = normal.'*(zb-anchor);

        fa = t(end);
        fb = tn(end);
        wa = fa;
        wb = fb;

        side = 0;
        nose = true;
        event_steps = 0;

    else
        error_predictor = norm(z1-predictor,inf);

        h = min(hmax,h*min(2,max(0.5, ...
            sqrt(1e-3/max(error_predictor,1e-15)))));

        z = z1;
        t = tn;
    end
end

if ~found
    error('CPF exceeded its iteration limit.');
end

alpha = z(end);
x = z(1:m);
q = tn(1:m);
end


function [alpha,residual,updates] = fold_reference(x,alpha,q,W)
% Refine the physical fold equations independently of continuation.

m = numel(x);
q = q/norm(q);
c = q;
p = sparse(W.p);
updates = 0;

[~,J0] = power_flow(zeros(m,1),0,W);

for k = 0:10
    [g,J] = power_flow(x,alpha,W);

    H = [g;J*q;c.'*q-1];
    residual = norm(H,inf);

    if residual <= 1e-9
        return
    end

    if k == 10
        break
    end

    [~,Jq] = power_flow(q,0,W);

    K = [ ...
        J,       p,           sparse(m,m); ...
        Jq-J0,   sparse(m,1),  J; ...
        sparse(1,m), 0,       sparse(c.')];

    delta = K\H;

    x = x-delta(1:m);
    alpha = alpha-delta(m+1);
    q = q-delta(m+2:end);

    updates = updates+1;
end

error('The independent fold reference did not converge.');
end


function M = icm_radial_model(mpc)
% Validate and construct a series-only radial PQ network.
% MATPOWER-format data are read directly; MATPOWER is not required.

b = mpc.bus;
br = mpc.branch(mpc.branch(:,11)>0,:);
N = size(b,1);

assert(N >= 2, 'ICM:Size', ...
    'At least two buses are required.');

assert(numel(unique(b(:,1))) == N, 'ICM:BusIDs', ...
    'Bus IDs must be unique.');

[ok1,f] = ismember(br(:,1),b(:,1));
[ok2,t] = ismember(br(:,2),b(:,1));

assert(all(ok1 & ok2) && all(f ~= t), 'ICM:Branches', ...
    'Invalid branch endpoints.');

ref = find(b(:,2)==3);
npv = nnz(b(:,2)==2);
ns = nnz(any(b(:,5:6)~=0,2));
nc = nnz(br(:,5)~=0);
nt = nnz(br(:,9)~=0 & br(:,9)~=1);
nshift = nnz(br(:,10)~=0);

if numel(ref) ~= 1 || size(br,1) ~= N-1 || ...
        npv > 0 || ns+nc+nt+nshift > 0

    error('ICM:Inapplicable', ...
        ['Class-3 ICM requires a radial PQ series network with one slack.\n' ...
         'Input: %d buses, %d active branches, %d PV, %d slack,\n' ...
         '%d bus shunts, %d charging terms, %d nonunit taps, %d phase shifts.\n' ...
         'The input has not been modified; no ICM result was computed.'], ...
        N,size(br,1),npv,numel(ref),ns,nc,nt,nshift);
end

assert(all(b(setdiff(1:N,ref),2)==1), 'ICM:BusTypes', ...
    'All other buses must be PQ.');

assert(all(br(:,3)>0 & br(:,4)>0), 'ICM:Impedance', ...
    'Positive R and X are required.');

assert(isfinite(mpc.baseMVA) && mpc.baseMVA>0, 'ICM:Base', ...
    'baseMVA must be positive and finite.');

assert(all(isfinite(b(:))) && all(isfinite(br(:))), 'ICM:Data', ...
    'Nonfinite network data.');

g = zeros(0,10);

if isfield(mpc,'gen') && ~isempty(mpc.gen)
    g = mpc.gen(mpc.gen(:,8)>0,:);

    assert(all(isfinite(g(:))), 'ICM:Data', ...
        'Nonfinite generator data.');

    assert(all(g(:,1)==b(ref,1)), 'ICM:Generation', ...
        'This implementation supports loads and a slack generator only.');
end

% Root the supplied tree without changing its branches.
m = N-1;
H = sparse([f;t],[t;f],[(1:m)';(1:m)'],N,N);

order = zeros(N,1);
order(1) = ref;

position = zeros(N,1);
position(ref) = 1;

parent = zeros(m,1);
edge = zeros(m,1);
tail = 1;

for head = 1:N
    assert(head <= tail, 'ICM:Disconnected', ...
        'The supplied network is disconnected.');

    u = order(head);
    [neighbors,~,branch_id] = find(H(:,u));

    for j = 1:numel(neighbors)
        v = neighbors(j);

        if position(v) > 0
            continue
        end

        tail = tail+1;
        order(tail) = v;
        position(v) = tail;
        parent(tail-1) = head-1;
        edge(tail-1) = branch_id(j);
    end
end

assert(nnz(parent==0)==1, 'ICM:PQConnected', ...
    'The PQ subgraph must be connected.');

M.n = m;
M.parent = parent;
M.r = br(edge,3);
M.x = br(edge,4);
M.p = b(order(2:end),3)/mpc.baseMVA;
M.q = b(order(2:end),4)/mpc.baseMVA;

assert(all(M.p>=0 & M.q>=0), 'ICM:LoadSigns', ...
    'This loading path requires nonnegative P and Q demands.');

Vset = b(ref,8);

if ~isempty(g)
    assert(max(g(:,6))-min(g(:,6)) < 1e-12, 'ICM:SlackVoltage', ...
        'Slack generators have inconsistent voltage setpoints.');

    Vset = g(1,6);
end

assert(Vset > 0, 'ICM:SlackVoltage', ...
    'The slack voltage must be positive.');

M.V0 = Vset*exp(1i*b(ref,9)*pi/180);

assert(abs(M.V0) > 0, 'ICM:SlackVoltage', ...
    'The slack voltage must be nonzero.');

% Build sparse reduced incidence and admittance matrices.
k = find(parent>0);
M.A = speye(m)-sparse(parent(k),k,1,m,m);
M.AT = M.A.';

M.pbar = M.A\M.p;
M.qbar = M.A\M.q;

assert(all(M.pbar.^2+M.qbar.^2 > 0), 'ICM:ZeroFlow', ...
    'Every branch subtree must contain a nonzero load.');

M.dlam = -2*(M.AT\(M.r.*M.pbar + M.x.*M.qbar));

y = 1./(M.r+1i*M.x);
M.Y = M.A*spdiags(y,0,m,m)*M.AT;
M.b = -M.V0*(M.A*(y.*(parent==0)));

M.bus_ids = b(order(2:end),1);
M.ref_bus_id = b(ref,1);
M.name = 'supplied radial AC case';

if isfield(mpc,'name')
    M.name = mpc.name;
end

% Nonnegative demands and positive impedances imply K >= 0.
% A connected PQ tree with one slack edge implies W > 0.
end

