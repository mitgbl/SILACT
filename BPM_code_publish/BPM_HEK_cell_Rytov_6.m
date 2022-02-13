clear;clc;close all
%% parameters
frames = 1:49;

lampta=0.532; % wavelength of laser/um
k_0=2*pi/lampta; % wave number
NA = 1.25;
delta_x = 20;
delta_y = 20;
NN = 256;
lambda0 = 532e-9; % free space wavelength (m)
magnification = 380; % Mnominal*Fused/Fnominal
% camera_pixel_size = (delta_x/magnification)*MM/NN*1e-6; % camera pixel size (m)

str0 = cd;
str1 = 'D:\20200702_both_rbcs';
% modify the path of the data over here
% 
paths = dir(str1);
% paths = dir(str1);
% mkdir([str1,'\GT_BPM']);
% mkdir([str1,'\GT_views']);


L = length(paths);        
   
 for m = 39:L

        str2 = [str1,'\',paths(m).name];
        str_2 = [str1,'\',paths(3).name];
        str3 = [str_2,'\bg_s'];
        str4 = [str2,'\sample_s'];
%         str5 = [str2,'\bg_m'];
%         str6 = [str2,'\sample_m'];
        
%         mkdir([str2,'\true_phases']);
        
        paths_bg = dir(str3);
        paths_sample = dir(str4);
        
        for mm = 1:49
            
            filename_bg = [str3,'\',paths_bg(mm+2).name];
            filename_s = [str4,'\',paths_sample(mm+2).name];
            I_bg = im2double(imread(filename_bg));
            I_bg = squeeze(I_bg(:,:,1));
            I_m = im2double(imread(filename_s));
            I_m = squeeze(I_m(:,:,1));
            
            [mx_b(mm),my_b(mm),~]=phase_recover_normal(NA,k_0,I_bg,delta_x/magnification,delta_y/magnification);
%             rho_bg(mm) = sqrt(mi^2+mj^2);
            [mx_s(mm),my_s(mm),~]=phase_recover_normal(NA,k_0,I_m,delta_x/magnification,delta_y/magnification);
%             rho_s(mm) = sqrt(mi^2+mj^2);
            
            sample_bg_i(:,:,mm) = I_bg;
            sample_i(:,:,mm) = I_m;
            
        end
        
        
        mx_max_b = max(mx_b);
        mx_min_b = min(mx_b);
        my_max_b = max(my_b);
        my_min_b = min(my_b);
        mx_ctr_b = (mx_max_b + mx_min_b)./2;
        my_ctr_b = (my_max_b + my_min_b)./2;
        dist_b = (mx_b - mx_ctr_b).^2 + (my_b - my_ctr_b).^2;
        [~,first_b] = (min(dist_b));
        sample_bg = sample_bg_i(:,:,[first_b:49,1:(first_b-1)]);
        
        mx_max_s = max(mx_s);
        mx_min_s = min(mx_s);
        my_max_s = max(my_s);
        my_min_s = min(my_s);
        mx_ctr_s = (mx_max_s + mx_min_s)./2;
        my_ctr_s = (my_max_s + my_min_s)./2;
        dist_s = (mx_s - mx_ctr_s).^2 + (my_s - my_ctr_s).^2;
        [~,first_s] = (min(dist_s));
        sample = sample_i(:,:,[first_s:49,1:(first_s-1)]);
%         sample_bg = sample_bg_i;
%         sample = sample_i;
        
%         clear paths_bg paths_sample
        
%         paths_bg = dir(str5);
%         paths_sample = dir(str6);
        
%         str3 = [str2,'\bg_s'];
%         paths_bg = dir(str3);      
%         filename_1 = [str3,'\',paths_bg(3).name];    
%         Im_0 = imread(filename_1);
%         Im_0 = Im_0(:,:,1);
%         intensity_0 = im2double(Im_0);
        intensity_0 = sample_bg(:,:,1);
        [MM,~] = size(intensity_0);
        
        camera_pixel_size = (delta_x/magnification)*MM/NN*1e-6; % camera pixel size (m)
        
%         filename_2 = [str4,'\',paths_sample(3).name];
        
        [mi,mj,mask_1]=phase_recover_normal(NA,k_0,intensity_0,delta_x/magnification,delta_y/magnification);
        
        for l = 1:length(frames)
%             
            I_sample = sample(:,:,l);
            I_bg = sample_bg(:,:,l);
            
            C_im=phase_recover_circle(mi,mj,mask_1,I_sample);
            E_1 = abs(C_im);
            Phi_1 = (angle(C_im));
%             Phi_1 = imresize(Phi_1,[512,512]);
           
            
            
            C_bg=phase_recover_circle(mi,mj,mask_1,I_bg);
            E_bg = abs(C_bg);
            Phi_bg = (angle(C_bg));

           
            Phi1 = (unwrap2(Phi_1-Phi_bg));
            [Phi1] = RemoveTilt_phi(100,100,(Phi1));
            
            Phi_stack(:,:,l) = Phi1;
          figure(668);imagesc(Phi1,[-2*pi,2*pi]);colormap jet;colorbar
          
%           print([str2,'\true_phases\phase_',num2str(l),'.jpg'],'-djpeg','-r600');
%              caxis([-2*pi,2*pi]);
             
              Phi_bg = unwrap2((Phi_bg));
             Phi_bg_stack(:,:,l) = Phi_bg;
            
        end
        
%         save([str2,'\Phase_true_stack.mat'],'Phi_stack');
%       
%         [MM,~] = size(intensity_0);
             
%         theta_i = Tilt_angles(str3,str0);
%% Generate input wave


        n0 = 1.337; % background refractive index (immersion oil)
%         n0 = 1.56;
        k0 = 2*pi/lambda0; % free space wavenumber (1/m)
        ks = n0*k0; % medium wavenumber (1/m)

%%% depends on cropping
        aa=1;
        Nx = aa*NN; % number of pixels along x
        Ny = aa*NN; % number of pixels along y
        Nz = aa*100; % number of pixels along z

        dx = 1*camera_pixel_size; % discretization step along x
        dy = 1*camera_pixel_size; % discretization step along y
        dz = 1*camera_pixel_size; % discretization step along z

        Lx = Nx*dx; % length of computational window along x
        Ly = Ny*dy; % length of computational window along y
        Lz = Nz*dz; % length of computational window along z

        x = dx*(-Nx/2+1:Nx/2)'; % computational grid along x (horiz)
        y = dy*(-Ny/2+1:Ny/2)'; % computational grid along y (vert)
        z = dz*(1:Nz)'; % computational grid along z

        [XX, YY] = meshgrid(x, y); %2D meshgrid

        dkx = 2*pi/Lx; % frequency discretization step along x
        dky = 2*pi/Ly; % frequency discretization step along y
        dkz = 2*pi/Lz; % frequency discretization step along z

        kx = dkx*[0:Nx/2-1, -Nx/2:-1]'; % frequency grid along x
        ky = dky*[0:Ny/2-1, -Ny/2:-1]'; % frequency grid along y
        kz = dkz*[0:Nz/2-1, -Nz/2:-1]'; % frequency grid along z

        [Kxx, Kyy] = meshgrid(kx, ky); % 2D frequency meshgrid

        K1 = Kxx+Kyy;
        K2 = Kxx.^2+Kyy.^2; % frequency norm for all points

        dphi = real(K2./(ks+sqrt(ks^2-K2))); % diffraction phase factor
%         dphi0 = real(K2./(2.*ks)); % diffraction phase factor

        [~, mid_index_y] = min(abs(y)); % midpoint index along x
        [~, mid_index_x] = min(abs(x)); % midpoint index along y
        [~, mid_index_z] = min(abs(z-Lz/2)); % midpoint index along z

        forwardObj = PlaneWave3D(Lx, Ly, Lz, Nx, Ny, Nz);
%% Phase retrieval        
        
        angx = 0;
        angy = 0;
        
        Bx = 0.3*Lx; % beam scale (m)
        Bx = Bx/cos(angx); % projection of the propagation plane to XY plane
    
        By = 0.3*Ly;
        By = By/cos(angy);
    
        ain = exp(-((XX./Bx).^2 + (YY./By).^2)); % illumination beam amplitude
%    pin = k*(sin(angx)*XX + sin(angy)*YY);
       
        theta = pi/3; % maximum scanning angle (radians)
        Ntheta = 48; % number of scanning angles
        %thetas = linspace(-Ltheta/2, Ltheta/2, Ntheta);
        psais = [0,linspace(0, 2*pi, Ntheta-1)];
        
        parfor ll = 1:49           

            Phi = squeeze(Phi_stack(:,:,ll));
            Phi = imresize(Phi,[NN,NN]);
            Phi_bg = squeeze(Phi_bg_stack(:,:,ll));
            Phi_bg = imresize(Phi_bg,[NN,NN]);
            x_c = 100;
            y_c = 100;
        
            [fx,fy,~] = RemoveTilt(x_c,y_c,Phi_bg);% remove the phase tilting
        
            [XX1,YY1] = meshgrid([1:NN],[1:NN]);
            Phi_art = fx(1).*XX1 + fy(1).*YY1 + fx(2) + fy(2);% artificial phase background
        
            g_in = ain.*exp(1i.*Phi_art);
            g_in = ifft2(fft2(g_in).*exp(-1i*(-Lz/2)*1.14*dphi));
            g_out = ain.*exp(1i.*(Phi+Phi_art));

        
            G_out(:,:,ll) = g_out; 
            G_in(:,:,ll) = g_in;

        end

        %% Rytov reconstruction

        pixelsize = (delta_x/magnification)*MM/NN;
        NA = 1.25;
        lambda = 0.532;
        index_medium = 1.337;
        k0 = 1/lambda*index_medium;
        
        phi = pi/3; % need to be calculted
        N = 48;
        theta = linspace(0,2*pi,N);

        k_x = [0,k0.*cos(phi).*cos(theta)];
        k_y = [0,k0.*cos(phi).*sin(theta)];
        
%         phase = Phi_stack;
        
        for mm = 1:49
           
            phase(:,:,mm) = imresize(squeeze(Phi_stack(:,:,mm)),[256,256]);
            
        end

        [xx, yy, frame]=size(phase);
        originalSize = xx;

        % cropsize=round(xx*pixelsize*NA/lambda)*2;
        cropsize = xx;
        if mod(cropsize,2) ~= 0
            cropsize = cropsize + 1;
        end
        crop_factor = cropsize/originalSize;
        xr=(cropsize*pixelsize*NA/lambda);
        yr =(cropsize*pixelsize*NA/lambda);
        k_z = zeros(1, frame);
        for ii = 1:frame
            k_z(ii) = sqrt(k0^2-k_x(ii)^2-k_y(ii)^2);
        end
        res = pixelsize/crop_factor;
        kres = 1/(res*cropsize);
        ktick = ((0:cropsize-1)-cropsize/2)*kres;
        xtick = ((1:cropsize)-cropsize/2)*res;
% 
        G = zeros(size(phase));
        for ii = 1:frame
            G(:,:,ii) = fftshift(fftn(fftshift(1i*phase(:,:,ii))))*res^2;
        end
% Fourier-based reconstruction
        max_ang = 60; % in degree
        F_tomo3 = Rytov_Fourier(G,k_x,k_y,k_z,cropsize,k0,kres,res,ktick,max_ang,frame);

        % save('Fourier_based_Reconst.mat','F_tomo3');

        F_tomo3 = HandleSingularity(F_tomo3);
        F_mask = zeros(size(F_tomo3));
        F_mask(F_tomo3~=0) = 1;

        f_map = fftshift(ifftn(ifftshift(F_tomo3)))*(kres*cropsize)^3;
        n = sqrt(1-real(f_map)/(2*pi*k0)^2)*index_medium;
        if ~isreal(n)
            n = real(n);
        else
        end

%
        [max_idx, max_idy, max_idz] = ind2sub(size(n),find(n==max(max(max(n)))));
        figure(1); subplot(221); imagesc(squeeze(n(:,:,cropsize/2+128-xx/2-00)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-y');
        subplot(222); imagesc(squeeze(n(:,cropsize/2+128-xx/2,:)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-z');
        subplot(223); imagesc(squeeze(n(cropsize/2+128-xx/2,:,:)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('y-z');

%% BPM reconstruction       
        fhat00 = n(:,:,79:178)-index_medium;
%         fhat00 = zeros(NN,NN,100);
        stepSize = 1e-3;
        bounds = [0,0.1];
        numIter = 50;
        lambda = 0.001;
        tv_maxiter = 10;

        fhat = fistaEst(-G_out, -G_in, forwardObj, lambda, fhat00, numIter, stepSize,...
            tv_maxiter, bounds);
%  end
% end
        index_medium = 1.337;
        n_pred = fhat + index_medium;

%     n_pred = n_pred(:,:,:);
    
    save([str2,'\RI_map_BPM.mat'],'n_pred');
    save([str1,'\GT_BPM\RI_BPM_',num2str(m-2),'.mat'],'n_pred');
    
dx = camera_pixel_size;
dz = camera_pixel_size;

xx = (1:256).*dx;
yy = (1:256).*dx;
zz = (1:100).*dz;


figure(555); subplot(221); imagesc(xx,yy,squeeze(n_pred(:,:,50)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n_pred(:,128,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n_pred(128,:,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('y-z');
        
        print([str2,'\RI_BPM.jpg'],'-djpeg','-r600');
        print([str1,'\GT_views\RI_BPM_',num2str(m-2),'.jpg'],'-djpeg','-r600');
 end

