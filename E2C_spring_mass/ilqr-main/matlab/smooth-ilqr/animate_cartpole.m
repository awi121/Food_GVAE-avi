function animate_cartpole(x,inputs, dt)
% x: collection of state vectors at each time, similar to output from ode45
% dt: (difference in time between each row of xout, generated by calling
% Written by Nathan Kong

bRecord = 1;  % Uncomment this to save a video
framerecord = 0;
if bRecord
    % Define video recording parameters
    Filename = 'cartpole_animation';
    v = VideoWriter(Filename, 'MPEG-4');
    myVideo.Quality = 100;
    open(v);
end

% Define axis window
xmin = -1;
xmax = 1;
ymin = -0.5;
ymax = 0.5;

Fig = figure('Color', 'w');

% Create trace of trajectory and particle object
h = animatedline('LineStyle', ':', 'LineWidth', 1.5,'MaximumNumPoints',1000);
hh = animatedline('LineStyle', ':', 'LineWidth', 1.5,'MaximumNumPoints',1000);
manipulator = [];
endeffector = [];
force = [];

% Set up axes
axis equal
axis([xmin xmax ymin ymax])
xlabel('x')
ylabel('y')

% draw

L = 0.25;
for ii = 1:length(x)
    % Too lazy to comment the rest
    a = tic;
    
    set(gcf,'DoubleBuffer','on');
    
    q1 = x(ii,1);
    q2 = x(ii,2);
    % Theres no input on the last state
    if(ii<length(x))
        u = inputs(ii);
    end
    
    delete(manipulator);
    delete(endeffector);
    delete(force);
    manipulator = line([q1, q1+L*sin(q2)], [0, -L*cos(q2)], 'Color', [0;0;0],'LineStyle','-');
    endeffector = line(q1, 0, 'Color', [1;0;0],'Marker','.', 'MarkerSize', 50);
    hold on
    if(ii<length(x))
        force = quiver(q1,0,u,0,0.25,'b','LineWidth',1.5);
    end
    addpoints(h,q1, 0);
    addpoints(hh,q1+L*sin(q2),-L*cos(q2));
    title(['Time = ',num2str(ii*dt),'s'])
    drawnow limitrate
    if(mod(ii,10) == 1 && framerecord == 1)
        saveas(gcf,['cartpoleframes\frame',num2str(ii),'.png'])
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