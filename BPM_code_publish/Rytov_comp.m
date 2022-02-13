%%
function Comp_1 = Rytov_comp(F,kx_l,ky_l,kz_l,padd1,k0,kres,xtick,NA,Nframe)

gpuDevice(3);
gpu_F = gpuArray(single(F));
gpu_kx_1 = gpuArray(single(kx_l));
gpu_ky_1 = gpuArray(single(ky_l));
gpu_kz_1 = gpuArray(single(kz_l));
gpu_xtick = gpuArray(single(xtick));

[jj ii] = meshgrid(1:padd1,1:padd1);

U = gpuArray((ii-padd1/2-1)*kres);
V = gpuArray((jj-padd1/2-1)*kres);

Comp = gpuArray(zeros(padd1,padd1,padd1));

for num = 1:Nframe
    u0 = gpu_kx_1(num); v0 = gpu_ky_1(num); w0 = gpu_kz_1(num);

    ind = find((U+u0).^2+(V+v0).^2<(k0*sin(NA/180*pi))^2);
    w = sqrt(k0^2-(U(ind)+u0).^2-(V(ind)+v0).^2);
    W = w - w0;

    kk = floor(W/kres)+padd1/2+1;
    tmp1 = 1./((4*pi*w).^2).*gpu_F(sub2ind([padd1 padd1 padd1],ii(ind),jj(ind),kk));

    for zz = 1:padd1
        tmp2 = gpuArray(zeros(padd1,padd1));
        tmp2(ind) = tmp1.*exp(i*2*pi*W*gpu_xtick(zz));
        Comp(:,:,zz) = Comp(:,:,zz) + tmp2;
        clear tmp2
    end
end

for zz = 1:padd1
    Comp_sec = Comp(:,:,zz);
    Comp(:,:,zz) = fftshift(ifftn(ifftshift(Comp_sec)))*(kres*padd1)^2;
end

Comp_1 = gather(Comp);

clear gpu_F gpu_kx_1 gpu_ky_1 gpu_kz_1 gpu_xtick U V Comp;
wait(gpuDevice(3));