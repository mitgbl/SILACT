%%
function Adj_1 = Rytov_adj(G,kx_l,ky_l,kz_l,padd1,k0,kres,xtick,NA,Nframe)

gpuDevice(1);
gpu_G = gpuArray(single(G));
gpu_kx_1 = gpuArray(single(kx_l));
gpu_ky_1 = gpuArray(single(ky_l));
gpu_kz_1 = gpuArray(single(kz_l));
gpu_xtick = gpuArray(single(xtick));
% gpu_k0 = gpuArray(k0);
% gpu_kres = gpuArray(kres);


[jj ii] = meshgrid(1:padd1,1:padd1);

U = gpuArray((ii-padd1/2-1)*kres);
V = gpuArray((jj-padd1/2-1)*kres);

Adj = gpuArray(zeros(padd1,padd1,padd1));

for num = 1:Nframe
    u0 = gpu_kx_1(num); v0 = gpu_ky_1(num); w0 = gpu_kz_1(num);
    G0 = gpu_G(:,:,num);

    ind = find((U+u0).^2+(V+v0).^2<(k0*sin(NA/180*pi))^2);
    w = sqrt(k0^2-(U(ind)+u0).^2-(V(ind)+v0).^2);
    W = w - w0;
    
    tmp1 = 1./(-i*4*pi*w).*G0(sub2ind([padd1 padd1],ii(ind),jj(ind)));
    
    for zz = 1:padd1
        tmp2 = gpuArray(zeros(padd1,padd1));
        tmp2(ind) = tmp1.*exp(i*2*pi*W*gpu_xtick(zz));
        Adj(:,:,zz) = Adj(:,:,zz) + tmp2;
        clear tmp2
    end
end

for zz = 1:padd1
    Adj_sec = Adj(:,:,zz);
    Adj(:,:,zz) = fftshift(ifftn(ifftshift(Adj_sec)))*(kres*padd1)^2;
end

Adj_1 = gather(Adj);

clear gpu_G gpu_kx_1 gpu_ky_1 gpu_kz_1 gpu_xtick U V Adj;
wait(gpuDevice(1));