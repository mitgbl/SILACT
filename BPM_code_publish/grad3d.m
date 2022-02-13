function d = grad3d(f)
% f    : 3D image (double)

bc = 'circular';

d = zeros([size(f) 3]);

filter1 = cat(1,0,-1,1);
filter2 = cat(2,0,-1,1);
filter3 = cat(3,0,-1,1);

d(:,:,:,1) = imfilter(f,filter1,bc);
d(:,:,:,2) = imfilter(f,filter2,bc);
d(:,:,:,3) = imfilter(f,filter3,bc);