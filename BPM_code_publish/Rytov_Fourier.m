%%
function F_tomo3 = Rytov_Fourier(G,kx_l,ky_l,kz_l,padd1,k0,kres,res,ktick,NA,Nframe)

F_tomo3 = zeros(padd1,padd1,padd1);
N_rep3 = zeros(padd1,padd1,padd1);

for num = 1:Nframe
    F_Field0 = G(:,:,num);

%     Fourier mapping
    [col0,row0] = meshgrid(1:padd1,1:padd1);
    
    kx = ktick(row0) + kx_l(num);
    ky = ktick(col0) + ky_l(num);
    kz = sqrt(k0^2 - kx.^2 - ky.^2);

    W = kz - kz_l(num);
    depth0 = round(W/kres) + padd1/2 + 1;
%     disp(depth0);

    zp = 0;
   F_Field0 = i*4*pi.*kz.*exp(-i*2*pi*W*zp).*F_Field0;     %

    ind = find(sqrt(kx.^2+ky.^2)/k0<sin(NA/180*pi));
%    ind = find(sqrt(kx.^2+ky.^2)/k0<NA);

    row = row0(ind); col = col0(ind); depth = depth0(ind);
    F_Field = F_Field0(ind);
%     disp(depth);
%     depth = depth(depth>0);

%     figure(810), imagesc(log(abs(F_Field0))), axis image;

    for ppp = 1:length(ind)
        ii = row(ppp);
        jj = col(ppp);
        kk = depth(ppp);
        
%         disp(kk);

        if(N_rep3(ii,jj,kk)==0)
            F_tomo3(ii,jj,kk) = F_Field(ppp);
            N_rep3(ii,jj,kk) = 1;
        else
            F_tomo3(ii,jj,kk) = F_tomo3(ii,jj,kk).*N_rep3(ii,jj,kk) + F_Field(ppp);
            N_rep3(ii,jj,kk) = N_rep3(ii,jj,kk) + 1;
            F_tomo3(ii,jj,kk) = F_tomo3(ii,jj,kk)./N_rep3(ii,jj,kk);
        end
    end
%     figure(910), imagesc(log(abs(squeeze(F_tomo3(padd1/2+1,:,:))))), axis image;
end