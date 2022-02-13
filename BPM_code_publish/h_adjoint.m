function x = h_adjoint(r, angs, Nt, Nthetas)

r = reshape(r, Nt, Nthetas);
x = iradon(r, angs, 'linear', 'None', Nt);