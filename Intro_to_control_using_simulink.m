%% Introduction to Control using MATLAB
% For third year UCT control students
% Please give feedback on the document (error corrections + things that
% were explained badly)

% Author:       Alexander Knemeyer
% Date:         3 September, 2018
% Contact:      KNMALE002@myuct.ac.za

%% NOTE: Complete the Intro_to_control_using MATLAB.m guide before doing
%% this one

%% 4: SIMULINK
% simulink may well be the best feature of matlab.
% it allows you to create block diagrams for your system, which means you
% can work visually.
% it has loads of functionality and there are good tutorials to learn from
% http://bfy.tw/Jh6X

% anyway, start by typing
simulink
% and then come back here for more instructions. It may take a while to
% load

% choose 'blank model'.
% you should see a blank canvas. click on it, and then start typing to
% search for blocks. Start with 'tranfser fcn'. Press enter.
% now, double click on the block and enter some coefficients. Press enter.

% add another block. This time we'll define it in a different way.
% simulink has access to your workspace, where the variables you're
% programming with have been saved
A = 9.5; % do the 'A' value for your controller
T = 12.2; % do the 'T' value for your controller
% click on the controller, and type 'A' and 'T' for coefficients where
% appropriate. You'll likely have
    % numerator = [A]
    % denominator = [1 T 0]
% note that the transfer function for your helicopter is:
    % G_velocity = A/(sT + 1)
    % G_position = A/(sT + 1)/s = A/(s^2 * T + s)
% it is important that you realise this!
% click enter to save this model.

% move the blocks to be in line with each other. Then, click on the output
% arrow of one block and drag it to the input arrow of the next.

% add a 'step' block. Adjust the parameters as needed.

% add a 'sum' block where appropriate (we're trying to build the classic
% negative feedback control system). Double click on it, and change the
% second plus to a minus. Ie: |+-

% add a 'scope' block with two inputs. Right click on it > Signals and
% Ports > Number of Input Ports.

% scroll your mouse to zoom in and out as needed.
% press space, and then click and drag to pan around.

% make your block diagram look like this:

%      +-------------------------------+
%      |                               v
% step --> (sum) --> tf1 --> tf2 --> scope
%            ^                    |
%            +--------------------|

% I encourage everyone to not adopt a defeatest attitude, and try figure
% things out yourself if you get stuck. Then, ask for help or look online.

% next, click the green play button thing to simulate the system

% once it has been simulated, double click on the scope block to look at
% the outputs

% click on the settings gear thing to adjust paramters like simulation
% time, simulation time step size, etc

% some useful blocks are as follows:
%   gain
%   saturation (very very useful! you can't input u = 1000 in your system!)

% it is also useful to simulate a step input/output disturbance to your
% model, usually at a later time (like t = 10)
%      +-------------------------------+
%      |                   step2       |
%      |  +                |           v
% step --> (sum) --> K(s) --> P(s) --> scope
%            ^ -                   |
%            +-----=---------------|

%% Finished!
% and you've made it!
% well done!
% hope you read everything, because I spent most of my Sunday making these
% good luck for your projects
