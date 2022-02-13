%% This is the function to recover the complex field from interference intensity in the circle
function [image_recover]=phase_recover_circle(mi,mj,mask_1,intensity)

F_intensity=fftshift(fft2(intensity)); % Calculate the Frequency domain of the image
F_intensity_1=F_intensity.*mask_1; % extract the useful frequencies
figure(111);imagesc(log(abs(F_intensity_1)));colormap jet

% mi = x_max-M/2-1;mj = y_max-N/2-1;
F_intensity_2 = circshift(F_intensity_1,[-mi -mj]);
% Shift the extracted frequencies to the center
        
image_recover=(ifft2(fftshift(F_intensity_2))); % recover the complex field with ifft
% the fftshift is important, this function must be used even times
