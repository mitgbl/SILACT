function fhat = fistaEst(gout, gin, forwardObj, lambda, ...
    fhat, numIter, stepSize, tv_maxiter, tv_bounds, f)
% Function to reconstruct

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Convert gem to gin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% aviobj1 = VideoWriter('vid1.avi');
% open(aviobj1)
% aviobj2 = VideoWriter('vid2.avi');
% open(aviobj2)
computeSnr = @(f, fhat) 20*log10(norm(f(:))/norm(f(:)-fhat(:)));

if(~exist('f','var'))
    f = zeros(size(fhat));
end

Ntheta = size(gout, 3);
[Ny, Nx, Nz] = size(fhat);

P = zeros([size(fhat) 3]);

s = fhat;
t = 1;

cost = zeros(numIter, 1);
snr = zeros(numIter, 1);
gouthat = forwardObj.computeFullForward(fhat, gin); % we can observe here to see the output

stepSize0 = stepSize;

for iIter = 1:numIter
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Compute gradient
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    stepSize = stepSize0/sqrt(iIter);
    
%     it = randperm(Ntheta);
%     it = it(1:7);
    LL = size(gin,3);
    ginit = gin(:,:,1:LL);
    goutit = gout(:,:,1:LL);
    
    [grad, gouthatit] = forwardObj.computeFullGrad(goutit, s, ginit);    
    
    gouthat(:,:,1:LL) = gouthatit;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Update solution
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fhatnext = s - stepSize*grad;    
    [fhatnext, P] = denoiseTV_3D(fhatnext, lambda*stepSize,...
        'maxiter', tv_maxiter,...
        'bounds', tv_bounds,...
        'P', P,...
        'verbose', true);    % proximal optimization 
    
    tnext = 0.5*(1+sqrt(1+4*t*t));
    s = fhatnext + ((t-1)/tnext)*(fhatnext-fhat);
    
    fhat = fhatnext;
    t = tnext;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Performance metrics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %gouthat = forwardObj.computeFullForward(fhat, gin);   
    cost(iIter) = ((0.5/Ntheta)*norm(gouthat(:)-gout(:))^2 + lambda*tv_cost(fhat))/256^3;
    snr(iIter) = computeSnr(f, fhat);
    
    fprintf('t = %3.d: cost = %12.8e, snr = %f\n', iIter, cost(iIter), snr(iIter));
    
    % Plot the projection of the phantom along each of the spatial
    figure(202);
    set(202, 'Name', sprintf('t = %d', iIter));
    
    subplot(2, 3, 1:2);
    semilogy(1:iIter, cost(1:iIter), 'b-',...
        iIter, cost(iIter), 'ro');
    xlim([1 numIter]);
    grid on;
    title('cost');
    
    subplot(2, 3, 3);
    plot(1:iIter, snr(1:iIter), 'b-',...
        iIter, snr(iIter), 'ro');
    xlim([1 numIter]);
    grid on;
    title('snr');
    
    mm = min(fhat(:));
    MM = max(fhat(:));
    
    subplot(2, 3, 4);
    imagesc(squeeze(mean(fhat, 3)));
    axis square off;
    title('XY');
    
    subplot(2, 3, 5);
    imagesc(squeeze(mean(fhat, 2)));
    axis square off;
    title('YZ');
    
    subplot(2, 3, 6);
    imagesc(squeeze(mean(fhat, 1)));
    axis square off;
    title('XZ');
    colormap gray;
    
    drawnow;
    
img1=cat(2,squeeze(mean(fhat, 3)),squeeze(mean(fhat, 2)));
img2=cat(2,squeeze(mean(fhat, 1))',zeros(Nz,Nz));
img3=cat(1,img1,img2);
for i=1:5
figure(104);imagesc(img3);colormap gray;
drawnow
% frame1 =getframe;
% writeVideo(aviobj1,frame1)
end


img4=cat(2,squeeze(fhat(:,:,Nz/2-10)),squeeze(fhat(:,Nx/2,:)));
img5=cat(2,squeeze(fhat(Nx/2,:,:))',zeros(Nz,Nz));
img6=cat(1,img4,img5);
for i=1:5
figure(105);imagesc(img6);colormap gray;
drawnow
% frame2 =getframe;
% writeVideo(aviobj2,frame2)
end

gout_pre = forwardObj.computeFullForward(fhat, gin);

% figure(600);imagesc(abs(squeeze(gout_pre(:,:,20))));colormap jet;colorbar
% drawnow
% figure(601);imagesc(abs(squeeze(gout(:,:,20))));colormap jet;colorbar
% drawnow

end
% close(aviobj1)
% close(aviobj2)
function TVnorm = tv_cost(f)

fx=shift(f,[-1,0,0],'reflexive')-f;
fy=shift(f,[0,-1,0],'reflexive')-f;
fz=shift(f,[0,0,-1],'reflexive')-f;

TVf=sqrt(fx.^2+fy.^2+fz.^2);% Amplitude of the gradient vector

TVnorm=sum(TVf(:));