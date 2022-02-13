%% This is the function to recover the complex field from interference intensity in the circle
function [image_recover_1,image_recover_2,image_recover_3,image_recover_4]=phase_recover_multiplex_2(mi,mj,mask,intensity,NA,k_0,delta_x)

F_intensity=fftshift(fft2(intensity)); % Calculate the Frequency domain of the image
F_intensity_1=F_intensity.*mask; % extract the useful frequencies

[M,N]=size(intensity); % the image size of the image

% mi = x_max-M/2-1;mj = y_max-N/2-1;
F_intensity_2 = circshift(F_intensity_1,[-mi -mj]);
mask = circshift(mask,[-mi -mj]);
% Shift the extracted frequencies to the center
% F_1 = F_intensity_2(1:round(M/2),1:round(N/2));
% F_2 = F_intensity_2(round(M/2):M,1:round(N/2));
% F_3 = F_intensity_2(1:round(M/2),round(N/2):N);
% F_4 = F_intensity_2(round(M/2):M,round(N/2):N);
% figure;imagesc(abs(F_intensity_2));colormap jet;colorbar

[x_max(1),y_max(1)]=find(F_intensity_2==max(max(F_intensity_2(1:round(M/2)-80,:))));
[x_max(2),y_max(2)]=find(F_intensity_2==max(max(F_intensity_2(round(M/2)+80:M,:))));
[x_max(3),y_max(3)]=find(F_intensity_2==max(max(F_intensity_2(:,round(N/2)+80:N))));
[x_max(4),y_max(4)]=find(F_intensity_2==max(max(F_intensity_2(:,1:round(N/2)-80))));
disp(x_max);
disp(y_max);

% [M,N]=size(intensity); % the image size of the image

k_max=NA*k_0;
cutoff=round(k_max/(2*pi/(delta_x*M)));

for m=1:M
    for n=1:N
      if sqrt((m-x_max(1)-1)^2+(n-y_max(1)-1)^2)<cutoff
         mask_1(m,n)=1;
      else
         mask_1(m,n)=0;
     end
    end
end

for m=1:M
    for n=1:N
      if sqrt((m-x_max(2)-1)^2+(n-y_max(2)-1)^2)<cutoff
         mask_2(m,n)=1;
      else
         mask_2(m,n)=0;
     end
    end
end

for m=1:M
    for n=1:N
      if sqrt((m-x_max(3)-1)^2+(n-y_max(3)-1)^2)<cutoff 
         mask_3(m,n)=1;
      else
         mask_3(m,n)=0;
     end
    end
end

for m=1:M
    for n=1:N
      if sqrt((m-x_max(4)-1)^2+(n-y_max(4)-1)^2)<cutoff 
         mask_4(m,n)=1;
      else
         mask_4(m,n)=0;
     end
    end
end

figure(111);subplot(221);imagesc(mask-~mask_1);colormap gray
subplot(222);imagesc(mask-~mask_2);colormap gray
subplot(223);imagesc(mask-~mask_3);colormap gray
subplot(224);imagesc(mask-~mask_4);colormap gray

F_intensity_sub_1 = F_intensity_2.*mask_1;
F_intensity_sub_2 = F_intensity_2.*mask_2;
F_intensity_sub_3 = F_intensity_2.*mask_3;
F_intensity_sub_4 = F_intensity_2.*mask_4;

image_recover_1=(ifft2(fftshift(F_intensity_sub_1)));
image_recover_2=(ifft2(fftshift(F_intensity_sub_2)));
image_recover_3=(ifft2(fftshift(F_intensity_sub_3)));
image_recover_4=(ifft2(fftshift(F_intensity_sub_4)));

        
% image_recover=(ifft2(fftshift(F_intensity_2))); % recover the complex field with ifft
% the fftshift is important, this function must be used even times