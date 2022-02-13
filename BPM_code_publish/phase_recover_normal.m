%% This is the function to recover the complex field from interference intensity in the center
function [mi,mj,mask_1]=phase_recover_normal(NA,k_0,intensity,delta_x,delta_y)

% NA is the numerical aperture of the objective
% k_0 is the wavenumber of the illumination laser
% intensity is the interference image
% delta_x, delta_y are the pixel sizes

F_intensity=fftshift(fft2(intensity)); % Calculate the Frequency domain of the image
[M,N]=size(intensity); % the image size of the image

xx = 1:M;
yy = 1:N;

[X,Y] = meshgrid(xx,yy);
X = X';
Y = Y';

% figure(222);imagesc(log(abs(F_intensity)));colormap jet;axis square;axis off;hold on


[x_max_0,y_max_0]=find(F_intensity==max(max(F_intensity)));

% mask_0=zeros(M,N);
k_max=NA*k_0;
cutoff=round(k_max/(2*pi/(delta_x*M)));

R0 = sqrt((X-x_max_0-1).^2+(Y-y_max_0-1).^2);
mask_0 = ones(M,N);
mask_0(R0 < 2*cutoff) = 0;
F_intensity_0 = F_intensity.*mask_0;
% figure(224);imagesc(log(abs(F_intensity_0)));colormap jet;axis square;axis off;hold on

F_intensity_left = F_intensity_0;
F_intensity_left(X<round(N/2)) = 0;
% figure(225);imagesc(log(abs(F_intensity_left)));colormap jet;axis square;axis off;hold on
[x_max,y_max]=find(F_intensity_left == max(F_intensity_left(:)));

% disp('x_max = ');disp(x_max);
% disp('y_max = ');disp(y_max);

mask_1=zeros(M,N);
R = sqrt((X-x_max-1).^2+(Y-y_max-1).^2);
mask_1(R<cutoff) = 1; 

F_intensity_1=F_intensity.*mask_1; % extract the useful frequencies

%  figure(223);imagesc(log(abs(F_intensity_1)));colormap jet;

mi = x_max-M/2-1;mj = y_max-N/2-1;
% mi = x_max;mj = y_max;


