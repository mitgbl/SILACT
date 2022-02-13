function fhat_roi = radonRoiEst(gem, gob, thetas)


[Ny, Nx, Ntheta] = size(gem);
Nz = Nx;

angs = thetas(end:-1:1)*180/pi-90; % for iradon in degrees

x = 1:Nx;
y = 1:Ny;
z = 1:Nz;

[~, mid_index_y] = min(abs(y-Ny/2)); % midpoint index along x

phaseData = zeros(Ny, Nx, Ntheta);

scrsz = get(0,'ScreenSize');

figWidth =  scrsz(3)/3;
figHeight = scrsz(4)/2.5;

for ind_theta = 1:Ntheta
    imgem = gem(:,:,ind_theta);
    imgob = gob(:,:,ind_theta);
    
    %%% Extract raw amplitudes and phases
    ampem = abs(imgem);
    ampob = abs(imgob);
    phiem = angle(imgem);
    phiob = angle(imgob);
    
    %%% Plot fields
    figure(303);
    set(303,...
        'Name', sprintf('[%d/%d] Measured Field: Amplitude and Phase', ind_theta, Ntheta),...
        'Position', [1, scrsz(4)/2, figWidth, figHeight]);
    subplot(2, 4, 1);
    imagesc(x, y, ampem);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Empty-Amp');
    
    subplot(2, 4, 2);
    plot(x, squeeze(ampem(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('|gem|');
    xlim([1, Nx]);
    
    subplot(2, 4, 3);
    imagesc(x, y, ampob);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Object-Amp');
    
    subplot(2, 4, 4);
    plot(x, squeeze(ampob(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('|gob|');
    xlim([1, Nx]);
    
    subplot(2, 4, 5);
    imagesc(x, y, phiem);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Empty-Phase');
    
    subplot(2, 4, 6);
    plot(x, squeeze(phiem(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('angle(gem)');
    xlim([1, Nx]);
    
    subplot(2, 4, 7);
    imagesc(x, y, phiob);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Object-Phase');
    
    subplot(2, 4, 8);
    plot(x, squeeze(phiob(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('angle(gob)');
    xlim([1, Nx]);
    drawnow;
    
    %%% Unwrap the phase
    phiem1 = goldsteinUnwrap2d(imgem);
    phiob1 = goldsteinUnwrap2d(imgob);
    
    figure(304);
    set(304,...
        'Name', sprintf('[%d/%d] Unwrapped Phase', ind_theta, Ntheta),...
        'Position', [scrsz(3)/3, scrsz(4)/2, figWidth, figHeight]);
    
    subplot(2, 2, 1);
    imagesc(x, y, phiem1);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Empty-Phase');
    
    subplot(2, 2, 2);
    plot(x, squeeze(phiem1(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('angle(gem)');
    xlim([1, Nx]);
    
    subplot(2, 2, 3);
    imagesc(x, y, phiob1);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    title('Object-Phase');
    
    subplot(2, 2, 4);
    plot(x, squeeze(phiob1(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('angle(gob)');
    xlim([1, Nx]);
    drawnow;
    
    phi = phiob1-phiem1;
    phi(isnan(phi)) = 0;
    %phi = ain.*phi;
    phaseData(:,:,ind_theta) = phi;
    
    figure(305);
    set(305,...
        'Name', sprintf('[%d/%d] Final Phase', ind_theta, Ntheta),...
        'Position', [2*scrsz(3)/3, scrsz(4)/2, figWidth, figHeight]);
    
    subplot(1, 2, 1);
    imagesc(x, y, phi);
    hold on;
    plot([x(1) x(end)], [Ny/2 Ny/2], 'r--');
    hold off;
    axis square;
    xlabel('x');
    ylabel('y');
    
    subplot(1, 2, 2);
    plot(x, squeeze(phi(mid_index_y,:)), 'LineWidth', 1.5);
    grid on;
    xlabel('x');
    ylabel('phase(x)');
    xlim([1, Nx]);
    
    drawnow;
end

fhat_roi = zeros(Ny, Nx, Nz);

for ind_y = 1:Ny
    
    t = x; % coordinate orthogonal to the propagation direction
    Nt = Nx; % number of samples in the ortho direction
    dt = 1; % discretization step
    
    [Theta, X] = meshgrid(thetas, x); % Meshgrid for interp2
    
    %%% extract data
    rdata = squeeze(phaseData(ind_y, :, :)); % extract data for a given y
    rdata(rdata<0) = 0;
    
    rdata2 = zeros(Nx, Ntheta); % initialize the cosine transformed data
    
    %%% Transform (x, theta) to (t, theta)
    for ind_theta = 1:Ntheta
        for ind_t = 1:Nt
            theta = thetas(ind_theta);
            xt = t(ind_t)/cos(theta); % = t/cos(theta)
            rdata2(ind_t,ind_theta) =...
                interp2(Theta,X,rdata,theta,xt,'linear',0);
        end
    end
    
    %%% median filter
    rdata3 = medfilt2(rdata2, [3,3]);
    
    %%% Reconstruct with iradon
    ifhat_roi = iradon(rdata3, angs, 'linear','han',Nx);
    ifhat_roi(ifhat_roi<0) = 0;
    
    fhat_roi(ind_y,:,:) = ifhat_roi;
    
    figure(306);
    set(306, 'Name', 'reconstruction');
    subplot(1, 3, 1);
    imagesc(x, y, squeeze(mean(fhat_roi, 3)));    
    axis square;
    xlabel('x');
    ylabel('y');
    title('Pz');
    subplot(1, 3, 2);
    imagesc(z, y, squeeze(mean(fhat_roi, 2)));    
    axis square;
    xlabel('z');
    ylabel('y');
    title('Px');
    subplot(1, 3, 3);
    imagesc(z, x, squeeze(mean(fhat_roi, 1)));    
    axis square;
    xlabel('z');
    ylabel('x');
    title('Py');
end

close 303 304 305 306;