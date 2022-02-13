function [x_1, P_1, iter, L]=denoiseTV_3D(y, lambda, varargin)
%This method uses the TV regularizer and can be applied only to 3D data.
%x: denoised image
%P: dual variables
%L: Lipschitz constant
%iter: number of iterations for getting to the solution.
%bc: Boundary conditions for the differential operators
%    ('reflexive'|'circular'|'zero')

gpuDevice(1);

[maxiter,L,tol,optim,verbose,img,bounds,P,bc]=process_options(varargin,...
    'maxiter',100,'L',12,'tol',1e-4,'optim','fgp','verbose',false,'img',[],...
    'bounds',[-inf +inf],'P',zeros([size(y) 3]),'bc','reflexive');

y = gpuArray(single(y));
P = gpuArray(single(P));

count = 0;
flag = false;

if verbose
    fprintf('******************************************\n');
    fprintf('**   3D Denoising with TV Regularizer   **\n');
    fprintf('******************************************\n');
    fprintf('#iter     relative-dif   \t fun_val         Duality Gap    \t   ISNR\n')
    fprintf('====================================================================\n');
end

switch optim
    case 'fgp'
        t=1;
        F=P;
        for i=1:maxiter
            K=y-lambda*AdjTVOp3D(F,bc);
            Pnew=F+(1/(L*lambda))*TVOp3D(project(K,bounds),bc);
            Pnew=projectL2(Pnew);
            
            %relative error
            re=norm(Pnew(:)-P(:))/norm(Pnew(:));
            if (re<tol)
                count=count+1;
            else
                count=0;
            end
            
            tnew=(1+sqrt(1+4*t^2))/2;
            F=Pnew+(t-1)/tnew*(Pnew-P);
            P=Pnew;
            t=tnew;
            
            if verbose
                if ~isempty(img)
                    k=y-lambda*AdjTVOp3D(P,bc);
                    x=project(k,bounds);
                    fun_val=cost(y,x,lambda,bc);
                    dual_fun_val=dualcost(y,k,bounds);
                    dual_gap=(fun_val-dual_fun_val);
                    ISNR=20*log10(norm(y(:)-img(:))/norm(x(:)-img(:)));
                    % printing the information of the current iteration
                    fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
                else
                    k=y-lambda*AdjTVOp3D(P,bc);
                    x=project(k,bounds);
                    fun_val=cost(y,x,lambda,bc);
                    dual_fun_val=dualcost(y,k,bounds);
                    dual_gap=(fun_val-dual_fun_val);
                    fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
                end
            end
            
            if count >=5
                flag=true;
                iter=i;
                break;
            end
        end
        
    case 'gp'        
        for i=1:maxiter            
            K=y-lambda*AdjTVOp3D(P,bc);            
            Pnew=P+(1/(L*lambda))*TVOp3D(project(K,bounds),bc);
            Pnew=projectL2(Pnew);
            
            %relative error
            re=norm(Pnew(:)-P(:))/norm(Pnew(:));
            if (re<tol)
                count=count+1;
            else
                count=0;
            end
            
            P=Pnew;
            
            if verbose
                if ~isempty(img)
                    k=y-lambda*AdjTVOp3D(P,bc);
                    x=project(k,bounds);
                    fun_val=cost(y,x,lambda,bc);
                    dual_fun_val=dualcost(y,k,bounds);
                    dual_gap=(fun_val-dual_fun_val);
                    ISNR=20*log10(norm(y(:)-img(:))/norm(x(:)-img(:)));
                    % printing the information of the current iteration
                    fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val,dual_gap,ISNR);
                else
                    k=y-lambda*AdjTVOp3D(P,bc);
                    x=project(k,bounds);
                    fun_val=cost(y,x,lambda,bc);
                    dual_fun_val=dualcost(y,k,bounds);
                    dual_gap=(fun_val-dual_fun_val);
                    fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val,dual_gap);
                end
            end
            
            if count >=5
                flag=true;
                iter=i;
                break;
            end
        end
end

if ~flag
    iter=maxiter;
end

x=project(y-lambda*AdjTVOp3D(P,bc),bounds);

x_1 = gather(x);
P_1 = gather(P);

clear x P
wait(gpuDevice(1));


function Df=TVOp3D(f,bc) %TV operator with reflexive boundary conditions

% [d1,d2,d3]=size(f);
% Df=zeros(d1,d2,d3,2);
% Df(:,:,:,1)=shift(f,[-1,0,1],bc)-f;
% Df(:,:,:,2)=shift(f,[0,-1,0],bc)-f;
% Df(:,:,:,3)=shift(f,[0,0,-1],bc)-f;

bc = 'circular';
Df = zeros([size(f) 3]);
Df = gpuArray(single(Df));

filter1 = cat(1,0,-1,1);
filter2 = cat(2,0,-1,1);
filter3 = cat(3,0,-1,1);

filter1 = gpuArray(filter1);
filter2 = gpuArray(filter2);
filter3 = gpuArray(filter3);

% f = gpuArray(single(f));

Df(:,:,:,1) = imfilter(f,filter1,bc);
Df(:,:,:,2) = imfilter(f,filter2,bc);
Df(:,:,:,3) = imfilter(f,filter3,bc);

% clear filter1 filter2 filter3
% wait(gpuDevice(2));


function g=AdjTVOp3D(P,bc) %Adjoint TV operator

% P1=P(:,:,:,1);
% P1=shiftAdj(P1,[-1,0,0],bc)-P1;
% P2=P(:,:,:,2);
% P2=shiftAdj(P2,[0,-1,0],bc)-P2;
% P3=P(:,:,:,3);
% P3=shiftAdj(P3,[0,0,-1],bc)-P3;
% g=P1+P2+P3;

bc = 'circular';

filter1 = cat(1,1,-1,0);
filter2 = cat(2,1,-1,0);
filter3 = cat(3,1,-1,0);

filter1 = gpuArray((filter1));
filter2 = gpuArray((filter2));
filter3 = gpuArray((filter3));

fx = P(:,:,:,1);
fy = P(:,:,:,2);
fz = P(:,:,:,3);

fx = imfilter(fx,filter1,bc);
fy = imfilter(fy,filter2,bc);
fz = imfilter(fz,filter3,bc);

g = fx+fy+fz;

% clear filter1 filter2 filter3
% wait(gpuDevice(2));

function PB=projectL2(B)
PB = B./repmat(max(1,sqrt(sum(B.^2, 4))),[1 1 1 3]);

function Px=project(x,bounds)
lb=bounds(1);%lower box bound
ub=bounds(2);%upper box bound

if isequal(lb,-Inf) && isequal(ub,Inf)
    Px=x;
elseif isequal(lb,-Inf) && isfinite(ub)
    x(x>ub)=ub;
    Px=x;
elseif isequal(ub,Inf) && isfinite(lb)
    x(x<lb)=lb;
    Px=x;
else
    x(x<lb)=lb;
    x(x>ub)=ub;
    Px=x;
end

function [Q,TVnorm]=cost(y,f,lambda,bc)

% fx=shift(f,[-1,0,0],bc)-f;
% fy=shift(f,[0,-1,0],bc)-f;
% fz=shift(f,[0,0,-1],bc)-f;

bc = 'circular';

filter1 = cat(1,0,-1,1);
filter2 = cat(2,0,-1,1);
filter3 = cat(3,0,-1,1);

filter1 = gpuArray((filter1));
filter2 = gpuArray((filter2));
filter3 = gpuArray((filter3));

% f = gpuArray(single(f));

fx = imfilter(f,filter1,bc);
fy = imfilter(f,filter2,bc);
fz = imfilter(f,filter3,bc);

TVf=sqrt(fx.^2+fy.^2+fz.^2);% Amplitude of the gradient vector

TVnorm=sum(TVf(:));
Q=0.5*norm(y(:)-f(:))^2+lambda*TVnorm;

% clear filter1 filter2 filter3
% wait(gpuDevice(2));

function Q=dualcost(y,f,bounds)
r=f-project(f,bounds);
Q=0.5*(sum(r(:).^2)+sum(y(:).^2)-sum(f(:).^2));