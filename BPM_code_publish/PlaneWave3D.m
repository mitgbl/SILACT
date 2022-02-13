classdef PlaneWave3D < handle
    % Two-dimensional split-step Fourier beam propagation method for a
    % plane-wave beam.
    %
    % U. S. Kamilov, BIG, EPFL, 2014.
    
    properties(Constant)
        lambda0 = 532e-9; % Free-space wavelength [m]
        n0 = 1.337; % Background refractive index (immersion oil)
%         n0 = 1.56;
        k0 = 2*pi/PlaneWave3D.lambda0; % free space wavenumber [m-1]
        k = PlaneWave3D.n0*PlaneWave3D.k0; % medium wavenumber [m-1]
    end
    
    properties
        
        % computational grid (size of initial_focused_field)
        Nx;
        Ny;
        Nz;
        
        % discretization steps = (lambda0/NA)*(N_disk/N_detection_window)
        drx;
        dry;
        drz;
        
        % physical dimensions
        Lx;
        Ly;
        Lz;
        
        % domain coordinate vectors
        rx;
        ry;
        rz;
        
        % 2d meshgrid
        Rxx;
        Ryy
        
        % discretization in the spatial frequency domain: dkx = 2*pi/Lx
        dkx;
        dky;
        dkz;
        
        % spatial frequency vectors
        kx;
        ky;
        kz;
        
        % phase factor for Fresnel diffraction
        dphi;
    end
    
    methods
        function this = PlaneWave3D(Lx, Ly, Lz, Nx, Ny, Nz) % calculate the initial incident field 
            
            this.Lz = Lz;
            this.Lx = Lx;
            this.Ly = Ly;
            
            this.Nx = Nx;
            this.Ny = Ny;
            this.Nz = Nz;
            
            this.drx = this.Lx/this.Nx;
            this.dry = this.Ly/this.Ny;
            this.drz = this.Lz/this.Nz;
            
            this.rx = this.drx * (-this.Nx/2+1:this.Nx/2)';
            this.ry = this.dry * (-this.Ny/2+1:this.Ny/2)';
            this.rz = this.drz * (1:this.Nz);
            
            %[this.Rx, this.Ry, this.Rz] = meshgrid(this.rx,this.ry,this.rz);
            [this.Rxx, this.Ryy] = meshgrid(this.rx, this.ry);
            
            this.dkx = 2*pi/this.Lx;
            this.dky = 2*pi/this.Ly;
            this.dkz = 2*pi/this.Lz;
            
            this.kx = this.dkx * [0:this.Nx/2-1, -this.Nx/2:-1]';
            this.ky = this.dky * [0:this.Ny/2-1, -this.Ny/2:-1]';
            this.kz = this.dkz * [0:this.Nz/2-1, -this.Nz/2:-1]';
            
            [Kxx, Kyy] = meshgrid(this.kx, this.ky);
            
            K2 = Kxx.^2+Kyy.^2; % 2D frequency meshgrid
            
            this.dphi = real(K2./(this.k+sqrt(this.k^2-K2))); % diffraction phase factor
        end
        
        function [gout, gtot] = iforward(this, x, gin, theta) % initial forward model(Baoliang)
            
            gtot = 0.*ones(this.Ny, this.Nx, this.Nz);
            
            u = gin;
            
            dz = this.drz/cos(0);
            
            for ind_z = 1:this.Nz
                u = ifft2(fft2(u).*exp(-1i*dz*this.dphi)); % diffraction step
                u = u.*exp(1i*this.k0*dz*x(:,:,ind_z)); % refraction step
                gtot(:,:,ind_z) = u;
            end
            
            %%% % Backpropagate to the center
          u = ifft2(fft2(u).*exp(-1i*(-this.Lz/2)*this.dphi));
            
            gout = u;
        end
        
        function [grad, gout, gtot] = icomputeGrad(this, gi, x, gin, theta)% gi should be the measured field
            
            phi = this.dphi;
            
            [gout, gtot] = this.iforward(x, gin, theta);
            
            grad = zeros(this.Ny,this.Nx,this.Nz);
            
            dz = this.drz/cos(0);
            
            res = gout - gi;
            res = ifft2(fft2(res).*exp(-1i*(this.Lz/2)*this.dphi));
            
            for ind_z = this.Nz:-1:2
                
                s = ifft2(fft2(gtot(:,:,ind_z-1)).*exp(-1i*dz*phi));
                s = conj(s).*res.*conj(1i*this.k0*dz...
                    *exp(1i*this.k0*dz*x(:,:,ind_z)));
                
                grad(:,:,ind_z) = real(s);
                
                res = res.*conj(exp(1i*this.k0*dz*x(:,:,ind_z)));
                res = ifft2(fft2(res) .* exp(1i*dz*phi));
            end
            
            s = ifft2(fft2(gin).*exp(-1i*dz*phi));
            s = conj(s).*res...
                .*conj(1i*this.k0*dz*exp(1i*this.k0*dz*x(:,:,1)));
            grad(:,:,1) = real(s);
        end
        
        function gouthat = computeFullForward(this, fhat, gin)
            gouthat = zeros(size(gin));
            Ntheta = size(gouthat, 3);
            thetas = [0,ones(1,Ntheta-1).*pi/3];
                        
            parfor ind_theta = 1:Ntheta
                obj = this;
                igin = gin(:,:,ind_theta);
                igout = obj.iforward(fhat, igin, thetas(ind_theta));
                gouthat(:,:,ind_theta) = igout;
                fprintf('computeFullForward: %d of %d\n',...
                    ind_theta, Ntheta);
            end                        
        end
        
        function [grad, gouthat] = computeFullGrad(this, gout, fhat, gin)
            
            grad = zeros(size(fhat));
            gouthat = zeros(size(gout));
            Ntheta = size(gout,3);
            thetas = [0,ones(1,Ntheta-1).*pi/3];
                        
            parfor ind_theta = 1:Ntheta
                obj = this;
                igin = gin(:,:,ind_theta);
                igout = gout(:, :, ind_theta);
                [igrad, igouthat] = obj.icomputeGrad(igout, fhat, igin, thetas(ind_theta));
                grad = grad + igrad;
                gouthat(:,:,ind_theta) = igouthat;                
                fprintf('computeFullGradient: %d of %d\n',...
                    ind_theta, Ntheta);
            end            
            grad = grad/Ntheta;            
        end
    end
end