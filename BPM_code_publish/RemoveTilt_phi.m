function [Phi2] = RemoveTilt_phi(x_c,y_c,Phi)

 [xsize ysize] = size(Phi);
% cphi_ang = unwrap2(angle(cphi));

% xx = [1:xsize]';
x = [1:xsize]';
disp(size(x));
fx = polyfit(x,Phi(min(x):max(x),y_c),1);
res_tilt_comp_x = fx(1)*x + fx(2);

Phi1 = Phi - repmat(res_tilt_comp_x,[1 ysize]);
% cphi_ang = unwrap2(angle(cphi));

 y = 1:ysize;
fy = polyfit(y,Phi1(x_c,min(y):max(y)),1);
res_tilt_comp_y = fy(1)*y + fy(2);

Phi2 = Phi1 - repmat(res_tilt_comp_y,[xsize 1]);

end