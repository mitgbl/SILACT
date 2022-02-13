`function f = gradAdj3d(d)

bc = 'circular';

filter1 = cat(1,1,-1,0);
filter2 = cat(2,1,-1,0);
filter3 = cat(3,1,-1,0);

fx = d(:,:,:,1);
fy = d(:,:,:,2);
fz = d(:,:,:,3);

fx = imfilter(fx,filter1,bc);
fy = imfilter(fy,filter2,bc);
fz = imfilter(fz,filter3,bc);

f = fx+fy+fz;