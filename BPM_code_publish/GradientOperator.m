classdef GradientOperator < handle
    %%% This class contains methods that allow the computation of
    %%% horizontal and vertical gradients as well as the wrapping operator.
    
    properties
        sigSize; % size of the 2D signal
        epsilon = 1e-4; % small constant
    end
    
    methods
        function obj = GradientOperator(sigSize)
            obj.sigSize = sigSize;
        end
        
        function [q, pos, neg] = computeResiduals(obj, x)
            %%% Function computes the residuals of the wrapped signal x
            
            N = obj.sigSize(1); % rows
            M = obj.sigSize(2); % cols
            
            q = zeros(N-1, M-1);
            
            for m = 1:M-1
                for n = 1:N-1
                    q(n,m) = obj.W(x(n+1,m)-x(n,m))...
                        + obj.W(x(n+1,m+1)-x(n+1,m))...
                        + obj.W(x(n,m+1)-x(n+1,m+1))...
                        + obj.W(x(n,m)-x(n,m+1));
                end
            end
            
            pos = q > obj.epsilon;
            neg = q < -obj.epsilon;
            
            fprintf('Found %d (+) and %d (-) residuals.\n',...
                sum(pos(:)), sum(neg(:)));
        end
        
        function z = H(obj,x)
            % Second derivative difference
            
            z = obj.Lx(obj.Ly(x))-obj.Ly(obj.Lx(x));
        end
        
        function x = HT(obj,z)
            % Second derivative difference transpose
            
            x = obj.Ly_t(obj.Lx_t(z))-obj.Lx_t(obj.Ly_t(z));
        end
        
        function y = L(obj,x)
            % Gradient
            
            y = zeros([size(x), 2]);
            y(:,1:end-1,1) = obj.Lx(x);
            y(1:end-1,:,2) = obj.Ly(x);
        end
        
        function x = LT(obj, y)
            % Gradient transpose
            
            x_dx = squeeze(y(:,1:end-1,1));
            x_dy = squeeze(y(1:end-1,:,2));
            
            gx = obj.Lx_t(x_dx);
            gy = obj.Ly_t(x_dy);
            
            x = gx+gy;
        end        
    end
    
    methods(Static)
        function y = W(x)
            %%% Wrap the data
            y = mod(x+pi,2*pi)-pi;
        end
        
        function x_dx = Lx(x)
            % horizontal difference
            
            x_dx = x(:, 2:end) - x(:,1:end-1);
        end
        
        function x_dy = Ly(x)
            % vertical difference
            
            x_dy = x(2:end,:) - x(1:end-1,:);
        end
        
        function gx = Lx_t(diff_x)
            % horizontal difference transpose
            
            M = size(diff_x,1);
            gx = [zeros(M,1), diff_x, zeros(M,1)];
            gx = gx(:,1:end-1)-gx(:, 2:end);
        end
        
        function gy = Ly_t(diff_y)
            N = size(diff_y,2);
            gy = [zeros(1,N); diff_y; zeros(1,N)];
            gy = gy(1:end-1,:)-gy(2:end,:);
        end
    end
end