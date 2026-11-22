function results = run_parallel8_cpf(case_id)
% run_parallel8_cpf(1)：IEEE300
% run_parallel8_cpf(2)：PGLib10000

if nargin<1, case_id=1; end

nw=8;
alpha_tol=1e-4;
max_nr=10;
repeats=3;
sample_seconds=0.5;

file=fullfile(fileparts(mfilename('fullpath')),'IEEE_cases.mat');
S=load(file);
M=S.cases{case_id};

models={M.dc,M.ac};
names={'DC';'AC'};

fprintf('%s：%d个节点，%d个线程\n',M.name,size(M.bus,1),nw);

%% 创建线程池
t0=tic;
pool=gcp('nocreate');

if ~isempty(pool) && ...
        (pool.NumWorkers~=nw || ~isa(pool,'parallel.ThreadPool'))
    delete(pool);
    pool=[];
end

if isempty(pool)
    pool=parpool('Threads',nw);
end

pool_seconds=toc(t0);

%% CPF、串行二分、8点并行
T=zeros(2,3,repeats);
alpha=zeros(2,3);
err=zeros(2,3);
rounds=zeros(2,3);
updates=zeros(2,3);
reference=zeros(2,1);

upper=nan(2,3);
traces=cell(2,3);
counts=zeros(2,3,repeats);
residual=zeros(2,3);
warmup_seconds=zeros(2,1);

for model=1:2
    W=models{model};

    assert(norm(pf(W.x0,W.a0,W),inf)<=1e-7, ...
        '起始潮流未收敛');

    data=parallel.pool.Constant(W);

    % 预热
    warm=tic;
    for mode=1:3
        run_method(mode,W,data,alpha_tol,max_nr);
    end
    warmup_seconds(model)=toc(warm);

    lo=inf(1,3);
    hi=-inf(1,3);
    states=zeros(numel(W.x0),3);

    for rep=1:repeats
        order=mod((0:2)+rep-1,3)+1;

        for mode=order
            elapsed=0;
            count=0;

            while elapsed<sample_seconds || count==0
                tick=tic;

                [a,x,its,nround,ub,history,q]=run_method( ...
                    mode,W,data,alpha_tol,max_nr);

                elapsed=elapsed+toc(tick);
                count=count+1;

                lo(mode)=min(lo(mode),a);
                hi(mode)=max(hi(mode),a);

                residual(model,mode)=max(residual(model,mode), ...
                    norm(pf(x,a,W),inf));
            end

            T(model,mode,rep)=elapsed/count;
            counts(model,mode,rep)=count;

            alpha(model,mode)=a;
            states(:,mode)=x;
            updates(model,mode)=its;
            rounds(model,mode)=nround;
            upper(model,mode)=ub;
            traces{model,mode}=history;

            if mode==1
                q_cpf=q;
            end
        end

        fprintf(['%s，第%d轮：CPF %.6f s，' ...
            '串行二分 %.6f s，8点并行 %.6f s\n'], ...
            names{model},rep, ...
            T(model,1,rep),T(model,2,rep),T(model,3,rep));
    end

    % 独立核对最大负载精度，不计入算法时间
    reference(model)=fold_reference( ...
        states(:,1),alpha(model,1),q_cpf,W);

    err(model,:)=max( ...
        abs(lo-reference(model)),abs(hi-reference(model)));

    assert(all(err(model,:)<=alpha_tol), ...
        '最大负载误差超过alpha_tol');

    assert(all(residual(model,:)<=1e-7), ...
        '最终潮流残差未达标');

    delete(data);
end

%% 结果
seconds=median(T,3);

results.summary=table( ...
    names,alpha(:,1),alpha(:,2),alpha(:,3), ...
    seconds(:,1),seconds(:,2),seconds(:,3), ...
    seconds(:,1)./seconds(:,3), ...
    seconds(:,2)./seconds(:,3), ...
    'VariableNames',{ ...
    'model','alpha_CPF','alpha_binary','alpha_parallel8', ...
    'CPF_s','binary_s','parallel8_s', ...
    'speedup_vs_CPF','speedup_vs_binary'});

results.accuracy=table( ...
    names,reference,err(:,1),err(:,2),err(:,3), ...
    rounds(:,2),rounds(:,3),upper(:,3)-alpha(:,3), ...
    'VariableNames',{ ...
    'model','alpha_reference','error_CPF','error_binary', ...
    'error_parallel8','binary_rounds','parallel8_rounds', ...
    'final_width'});

results.raw_seconds=T;
results.timed_runs=counts;
results.newton_updates=updates;
results.traces=traces;
results.pool_seconds=pool_seconds;
results.warmup_seconds=warmup_seconds;
results.alpha_tol=alpha_tol;
results.max_nr=max_nr;
results.max_residual=residual;
results.case_name=M.name;

disp(results.summary);
disp(results.accuracy);

fprintf('加速比大于1，表示8点并行更快。\n');

end % 主函数结束


%% 方法选择
function [a,x,its,nround,ub,history,q]= ...
    run_method(mode,W,data,tol,max_nr)

q=[];
history=[];
ub=NaN;
nround=NaN;

if mode==1
    [a,x,its,q]=cpf(W);
elseif mode==2
    [a,x,its,nround,ub,history]=binary_limit(W,tol,max_nr);
else
    [a,x,its,nround,ub,history]= ...
        parallel8_limit(W,data,tol,max_nr);
end

end % run_method结束


%% 8点并行搜索
function [a,x,steps,rounds,right,history]= ...
    parallel8_limit(W,data,tol,max_nr)

[left,right,x,steps]=bracket(W,max_nr);

rounds=0;
history=zeros(ceil(log((right-left)/tol)/log(9)),5);

while right-left>tol
    % 8个内点，将区间分成9段
    aa=left+(1:8)*(right-left)/9;
    xleft=x;

    trial=zeros(numel(x),8);
    ok=false(1,8);
    its=zeros(1,8);

    % 8个负载点同时计算，任务之间没有依赖
    parfor j=1:8
        [u,good,it]=nr(xleft,aa(j),data.Value,max_nr);
        trial(:,j)=u;
        ok(j)=good;
        its(j)=it;
    end

    steps=steps+sum(its);
    rounds=rounds+1;

    assert(~any(diff(double(ok))>0), ...
        '点的判定结果不连续');

    j=sum(ok);
    oldleft=left;
    oldright=right;

    if j<8
        right=aa(j+1);
    end

    if j>0
        left=aa(j);
        x=trial(:,j);
    end

    history(rounds,:)=[oldleft oldright left right j];
end

history=history(1:rounds,:);
a=left;

end % parallel8_limit结束


%% 串行二分
function [a,x,steps,rounds,right,history]= ...
    binary_limit(W,tol,max_nr)

[left,right,x,steps]=bracket(W,max_nr);

rounds=0;
history=zeros(ceil(log((right-left)/tol)/log(2)),5);

while right-left>tol
    a=(left+right)/2;

    oldleft=left;
    oldright=right;

    [trial,ok,it]=nr(x,a,W,max_nr);
    steps=steps+it;

    if ok
        left=a;
        x=trial;
    else
        right=a;
    end

    rounds=rounds+1;
    history(rounds,:)=[oldleft oldright left right ok];
end

history=history(1:rounds,:);
a=left;

end % binary_limit结束


%% 建立初始区间
function [left,right,x,steps]=bracket(W,max_nr)

left=W.a0;
right=W.a0+1;
x=W.x0;
steps=0;

for j=1:30
    [trial,ok,it]=nr(x,right,W,max_nr);
    steps=steps+it;

    if ~ok
        return
    end

    left=right;
    x=trial;
    right=W.a0+2*(right-W.a0);
end

error('未找到搜索上端');

end % bracket结束


%% 单个点的牛顿迭代
function [x,ok,steps]=nr(x,a,W,max_nr)

ok=false;
steps=0;

for k=0:max_nr
    if any(~isfinite(x)) || (W.kind==1 && any(x<=0))
        return
    end

    [g,J]=pf(x,a,W);

    if norm(g,inf)<=1e-7
        ok=true;
        return
    end

    if k==max_nr
        return
    end

    x=x-J\g;
    steps=steps+1;
end

end % nr结束


%% 潮流方程及雅可比矩阵
function [g,J]=pf(x,a,W)

n=size(W.Y,1);

if W.kind==1
    v=x;
else
    v=x(1:n)+1i*x(n+1:end);
end

w=W.Y*v+W.b;
s=v.*conj(w);

if W.kind==1
    g=s+W.c+a*W.p;

    if nargout>1
        J=spdiags(v,0,n,n)*W.Y+spdiags(w,0,n,n);
    end
else
    g=[real(s);imag(s)];

    pv=W.pv;
    rows=n+pv;

    g(rows)=abs(v(pv)).^2;
    g=g+W.c+a*W.p;

    if nargout<2
        return
    end

    D=spdiags(conj(w),0,n,n);
    K=spdiags(v,0,n,n)*conj(W.Y);

    Je=D+K;
    Jf=1i*(D-K);

    J=[real(Je),real(Jf);imag(Je),imag(Jf)];
    J(rows,:)=0;

    J=J ...
        +sparse(rows,pv,2*real(v(pv)),2*n,2*n) ...
        +sparse(rows,n+pv,2*imag(v(pv)),2*n,2*n);
end

end % pf结束


%% CPF
function [alpha,x,updates,q]=cpf(W)

x=W.x0;
m=numel(x);
z=[x;W.a0];
updates=0;

p=sparse(W.p);
rhs=[zeros(m,1);1];

[~,gx]=pf(x,W.a0,W);

t=[gx,p;sparse(1,m),1]\rhs;
t=t/norm(t);

hmax=2*max(1,sqrt(m/1000));
h=min(hmax,1/t(end));

nose=false;
midpoint=false;
found=false;

for j=1:3000
    if ~nose
        anchor=z;
        normal=t;
        s=h;

        zp=z+h*t;
        z1=zp;
    else
        s=(left*wb-right*wa)/(wb-wa);

        if midpoint || ~(s>left && s<right)
            s=(left+right)/2;
        end

        q=(s-left)/(right-left);
        z1=(1-q)*za+q*zb;
    end

    ok=false;

    for k=0:10
        x=z1(1:m);

        if any(~isfinite(z1)) || z1(end)<0 || ...
                (W.kind==1 && any(x<=0))
            break
        end

        [f,gx]=pf(x,z1(end),W);
        g=[f;normal.'*(z1-anchor)-s];

        if norm(g,inf)<=1e-7
            ok=true;
            break
        end

        if k==10
            break
        end

        z1=z1-[gx,p;sparse(normal.')]\g;
        updates=updates+1;
    end

    if ~ok
        if nose
            if midpoint
                error('鼻点校正未收敛');
            end

            midpoint=true;
        else
            h=h/2;

            if h<1e-7
                error('CPF步长过小');
            end
        end

        continue
    end

    tn=[gx,p;sparse(normal.')]\rhs;

    if any(~isfinite(tn)) || norm(tn)==0
        error('CPF切线无效');
    end

    tn=tn/norm(tn);

    if tn.'*normal<0
        tn=-tn;
    end

    if nose
        f=tn(end);
        event_steps=event_steps+1;

        if abs(f)<=1e-6
            z=z1;
            found=true;
            break
        end

        if event_steps>=80
            error('鼻点定位次数超限');
        end

        if f>0
            za=z1;
            left=s;
            fa=f;
            wa=f;

            if side==1
                wb=wb/2;
            else
                wb=fb;
            end

            side=1;
        else
            zb=z1;
            right=s;
            fb=f;
            wb=f;

            if side==-1
                wa=wa/2;
            else
                wa=fa;
            end

            side=-1;
        end

        midpoint=false;

    elseif tn(end)<=0
        if tn(end)==0
            z=z1;
            found=true;
            break
        end

        za=z;
        zb=z1;
        anchor=za;

        normal=zb-za;
        normal=normal/norm(normal);

        if min(t.'*normal,tn.'*normal)<=0
            h=h/2;

            if h<1e-7
                error('CPF步长过小');
            end

            continue
        end

        left=0;
        right=normal.'*(zb-anchor);

        fa=t(end);
        fb=tn(end);
        wa=fa;
        wb=fb;
        side=0;

        nose=true;
        event_steps=0;

    else
        e=norm(z1-zp,inf);

        h=min(hmax, ...
            h*min(2,max(0.5,sqrt(1e-3/max(e,1e-15)))));

        z=z1;
        t=tn;
    end
end

if ~found
    error('CPF迭代次数超限');
end

alpha=z(end);
x=z(1:m);
q=tn(1:m);

end % cpf结束


%% 独立求解折点方程
function a=fold_reference(x,a,q,W)

m=numel(x);
q=q/norm(q);
c=q;
p=sparse(W.p);

[~,J0]=pf(zeros(m,1),0,W);

for k=0:10
    [g,J]=pf(x,a,W);
    H=[g;J*q;c.'*q-1];

    if norm(H,inf)<=1e-9
        return
    end

    if k==10
        break
    end

    [~,Jq]=pf(q,0,W);

    A=[ ...
        J, p, sparse(m,m); ...
        Jq-J0, sparse(m,1), J; ...
        sparse(1,m), 0, sparse(c.')];

    d=A\H;

    x=x-d(1:m);
    a=a-d(m+1);
    q=q-d(m+2:end);
end

error('参考折点未通过核验');

end % fold_reference结束