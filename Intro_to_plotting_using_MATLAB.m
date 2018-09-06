%% Introduction to MATLAB
% For third year UCT control students
% Please give feedback on the document (error corrections + things that
% were explained badly)

% Author:       Alexander Knemeyer
% Date:         3 September, 2018
% Contact:      KNMALE002@myuct.ac.za

%% NOTE: Complete the Intro_to_MATLAB.m guide before doing this one

%% 2: plotting
%% 2.1: basic plotting
% the general format is
% >> plot(Y)
% or
% >> plot(X, Y)
t = 0:0.1:10;
y = sin(t);
plot(t, y)  % don't close the plot window yet!

%% 2.2: adding descriptions to plots
% figures are accessible and can be updated after they have been made
% for example,
title('this is the title. Look at all the options above me')
legend('this is the legend - you can drag me around')
xlabel('xlabel'); ylabel('ylabel')
grid on;
shg;  % shows the graph

%% 2.3: putting multiple graphs on the same plot
% create a new plot (so that we don't mess with the old one)
figure()

% by default, a new plot is made each time the plot command is called
% you can change this using the 'hold' command
y = sin(0:0.1:10);
plot(y); hold on; plot(y + 1); plot(y + 2);
legend('y', 'y + 1', 'y + 2'); grid on;

%% 2.4: making subplots
y = sin(0:0.1:100);
y = y .* linspace(1, 10, length(y));

% the general format is
% >> subplot(num_vertical, num_horizontal, which_plot)
% for example,
subplot(3, 2, 1); % three high, 2 wide, first plot
plot(y);
title('1: top left');

subplot(3, 2, 2); plot(y + 10); title('2: top right')
subplot(3, 2, 3); plot(y - 10); title('3: middle left')
subplot(3, 2, 4); plot(y * 5); title('4: middle right')
subplot(3, 2, 5); plot(y / 5); title('5: you get the idea?')
shg;

% note that we didn't call figure() before, so the previous plot (if it
% wasn't closed) was overwritten

%% 2.5: closing plots
% either click on the window or type,
close all

%% Finished!
% good work!
% now do 'Intro_to_control_using_MATLAB.m'
