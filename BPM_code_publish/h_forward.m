function r = h_forward(f, angs, Nt)
r = radon(f, angs);
Nt2 = size(r);
tini = floor((Nt2-Nt)/2+1);
r = r(tini+1:tini+Nt,:);
r = r(:);