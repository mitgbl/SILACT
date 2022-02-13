% function [S] = sparse_3d(A)
% 
% [M,N,L]=size(A);
% 
% for k = 1:L
%     layer = squeeze(A(:,:,k));
%     vector = squeeze(reshape(layer,M*N,1));
%     A_2d(:,k) = vector;
% end
% 
% [I,J,V] = find(A_2d>0);

clear;clc;close all

str0 = 'D:\HEK_cell_training_set\Four_angle_multiplex\two ways of angle illumination_20200418\20200423_both_3T3\True_phases';
paths = dir(str0);
L = length(paths);

for m = 3:L
str1 = [str0,'\',paths(m).name];

load(str1);
[M,N,L] = size(Phi_stack);

Phi_stack(Phi_stack<-2*pi) = -2*pi;
Phi_stack(Phi_stack>2*pi) = 2*pi;

for k = 1:L
    
    Phi_sub = imresize(squeeze(Phi_stack(:,:,k)),[M/2,N/2]);
    Phi_sparse{k} = (sparse(Phi_sub));
    
end

% save([str0,'\Phase_sparse_stack.mat'],'Phi_sparse');
clear Phi_stack

for l = 1:L
   
    Phi_stack(:,:,l) = single(full(Phi_sparse{l}));
    
%     figure(666);imagesc(squeeze(Phi_stack_2(:,:,m)),[-2*pi,2*pi]);colormap jet;colorbar
%     drawnow
end

save(str1,'Phi_stack');

end