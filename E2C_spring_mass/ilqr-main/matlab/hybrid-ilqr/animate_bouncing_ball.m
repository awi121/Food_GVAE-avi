function animate_bouncing_ball(xout, dt,utrj)

bRecord = 1;  % Uncomment this to save a video
%  bRecord = 0;
if bRecord
    % Define video recording parameters
    Filename = ['bouncing_ball_animation'];
    v = VideoWriter(Filename, 'MPEG-4');
    myVideo.Quality = 100;
    open(v);
end

% xout: collection of state vectors at each time, output from ode45
% dt: (difference in time between each row of xout, generated by calling
% ode45 with the argument [tstart:dt:tfinal];)

% Define axis window
xmin = -5;
xmax = 5;
ymin = -1;
ymax = 6;

% Draw contact surfaces
x_a = linspace(xmin, xmax,500);
y_a = linspace(ymin, ymax,500);
[X,Y] = meshgrid(x_a,y_a);
a1 = Y;
% coeff = calc_constraint_coeff();
% a1 = coeff(1)*X+coeff(2)*Y - ones(size(Y))*coeff(3);
% Only want 1 contact mode for now
contour(X,Y,a1,[0,0], 'k'); hold on;
% contour(X,Y,a3,[0,0], 'k')

% Create trace of trajectory and particle object
h = animatedline('LineStyle', ':', 'LineWidth', 1.5);
particle = [];

% Set up axes
axis equal
axis([xmin xmax ymin ymax])
xlabel('x')
ylabel('y')
if(size(xout,1)>=1000)
   skip = 10;
else
    skip = 1;
end
% draw
force_handle = [];
for ii = 1:skip:size(utrj,1)
    a = tic;
    addpoints(h,0,xout(ii,1));
    drawnow limitrate
    delete(particle) % Erases previous particle
    delete(force_handle);
    particle = scatter(0,xout(ii,1),'ro','filled');
    if(nargin>2)
        force_handle = quiver(0,xout(ii,1),0,utrj(ii,1)/10,'b');
    else
        force_handle = [];
    end
    if bRecord
        frame = getframe(gcf);
        writeVideo(v,frame);
    else
        pause(dt - toc(a)); % waits if drawing frame took less time than anticipated
    end
end

if bRecord
    close(v);
end