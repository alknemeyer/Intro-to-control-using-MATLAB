%% Introduction to Control using MATLAB
% For third year UCT control students
% Please give feedback on the document (error corrections + things that
% were explained badly)

% Author:       Alexander Knemeyer
% Date:         3 September, 2018
% Contact:      KNMALE002@myuct.ac.za

%% NOTE: Complete the Intro_to_plotting_using_MATLAB.m guide before doing
%% this one

%% 3: basics of control
%% 3.1 creating transfer funtions
disp('------------------------------------------------')
% there are a couple ways of doing this. One way is to define 's' as a
% transfer function variable and then work with it
s = tf('s');
tf_1 = (s + 1)/(s^2 + 2*s + 5)

% another way is to pass in the coefficients of the transfer function
numerator_coeffs = [1, 2, 3];
denominator_coeffs = [6, 4, 3, 2];
tf_2 = tf(numerator_coeffs, denominator_coeffs)

% a third is to define the zeros, poles and gains of the transfer function
tf_3 = zpk(-1, [2, 2], 5)

%% 3.2: basic transfer function arithmetic
disp('------------------------------------------------')
% multiplication etc works as you'd expect
tf1 = 1/s;
tf2 = 1/(s + 3);

tf_prod = tf1 * tf2
tf_div = tf1 / tf2
tf_add = tf1 + tf2

%% 3.3.1: getting step responses
disp('------------------------------------------------')
% the 'step' command gives the OPEN LOOP step response of the system
sys = 3.5/(s + 2)

step(sys);
hold on;
step(sys/(1 + sys));  % manually 'close' the loop

legend('open loop -> T(s)', 'closed loop -> T(s)/(1+T(s))')

% don't think of sys/T(s) as a plant or any specific thing. It's just 'a
% transfer function' which relates some input to some output.

%% 3.3.2: more step responses
disp('------------------------------------------------')
% now, think back about how transfer functions work in the orignal block
% diagram. For example, we usually work with T_y/r (transfer function from
% input r to output y) but it's also useful to see other things, like the
% input to the plant. Let's make a simple controller - just a gain of 5
plant = 3/(s + 3);
controller = 5;
L = plant * controller % the loop gain

% transfer function from reference to output, closed loop
T_y_r_cl = L/(1 + L)

% transfer function from input disturbance to output, closed loop.
% look at the block diagram if this doesn't make sense!
T_y_u_cl = plant/(1 + L)

hold off % don't add to the previous plot
step(plant, T_y_r_cl, T_y_u_cl); grid on;
legend('r -> y, open loop', 'r -> y, closed loop', 'd_in -> y, closed loop')
shg;

%% 3.3.3: how do we interpret this?
disp('------------------------------------------------')
% closing the loop and adding a gain of five greatly increases the plant's
% performance! However, we still have two issues -

% first, while the plant is stable, there is a steady state error when
% tracking a unit step input

% second, if there is a step disturbance in between the controller and the
% plant input (say, some electronic component starts acting up and adds 1
% to all plant inputs) there will be a constant error. Our controller isn't
% smart enough to detect this and remove it. Look at the d_in -> y graph
% and think about this.

% what we need is an integrator.
% (how do we know this? control theory! go study)
% Look at the tut question about having an r(s) factor in your loop gain to
% ensure zero steady state error
plant = 3/(s + 3);
controller_v2 = 5 + 3/s 
L = plant * controller_v2;

T_y_r_cl = L/(1 + L);
T_y_u_cl = plant/(1 + L);
step(plant, T_y_r_cl, T_y_u_cl); grid on; shg;
legend('plant - open loop', 'r -> y, closed loop', 'd_in -> y, closed loop')

% now, if there is a step input disturbance, the plant initially goes a bit
% haywire BUT the controller accounts for this. The steady state error to a
% step input disturbance is zero

% how did I come up with the values for controller_v2? are they the best
% possible values? later on we'll look at some control design techniques
% (such as root locus) and try to answer these questions

%% 3.4.1: adding delays
disp('------------------------------------------------')
% just as in the maths interpretation, use an exponential
tf_delayed = 1/(s + 1) * exp(-3*s) % 3 second delay
step(tf_delayed); grid on; shg

%% 3.4.2: Pade's approximation for delays
disp('------------------------------------------------')
% we can replace the exponential with a polynomial using Pade's
% approximation
tf_delayed_pade = pade(tf_delayed)  % first order

step(tf_delayed, tf_delayed_pade); shg; grid on;
legend('actual delay', 'pade approximated delay')

% it's quite close, but not ideal. Why use Pade's approximation? The
% control design methods we know usually require things to be represented
% as a ratio of polynomials and often can't factor in the exponential part.
% So, we approximate the exponential with a ratio of polynomials

% higher order pade approximations can be found using
% >> pade(transfer_function_with_delay, order_of_approximation)

%% 3.5.1: System Identification - getting data and visualisation
close all;
% say we're given the step response data for a system, and we don't know
% what the model of the system is. How do we find the parameters which best
% model the plant?

% Let's do an example.

% Don't try to understand the code below (or at least pretend you didn't
% read it).
Ts = 0.2;  % sampling time
t = Ts:Ts:30;
t_step = 3;  % seconds
noise = randn(1, length(t))/10;  % normally distributed pseudorandom nums
step_data = 1.3*7.3215*(1 - exp(- max(t - t_step+Ts, 0)/3.916)) + noise;

% (step_data will have different values each time you run this cell due to
%  the added random noise)

% all you know is that you went to the lab, did a step response and got
% some data.
% the first step is to put the data into matlab/excel/whatever and plot it
plot(t, step_data); grid on; title('step response of ~~unknown system~~')

% the input signal was as follows:
step_size = 1.3;
step_index = t_step/Ts - 1;
U = [zeros(1, step_index), step_size*ones(1, length(t) - step_index)];
hold on; plot(t, U)
legend('plant output (from lab)', 'plant input (from lab)')

% don't worry too much if you don't understand the code used to create this
% step data. Just keep going through the guide and try to understand

% looking at the graph, it should be pretty clear that we're dealing with
% the classic ol' first order low pass system. Ie,
%       P(s) = A/(sT + 1)
% given an input U(s) = B/s to this model results in,
%       Y(s) = A*B/s/(sT + 1)
%       y(t) = A*B*(1 - exp(-t/T))

%% 3.5.2: System Identification - get data ready
% the next step is to normalise the data - the step didn't happen at t = 0,
% and the step wasn't a unit step input (it was B = 1.3)

% find the step time and input size by looking at the data
step_data = step_data / (max(U) - min(U));  % account for size of step input
step_data = step_data(step_index:end);  % start from the step time

t = t(step_index:end);
t = t - t(1);  % make time start at 0
plot(t, step_data); grid on; title('step response - normalised')

% notice how a plot is made at each point in the system ID process - it's
% much easier to tell whether you're doing the right thing by looking at a
% place than by looking at numbers

%% 3.5.3: System Identification - first approach: 63% method
disp('------------------------------------------------')
% letting t -> infinity, the step response becomes,
%       lim t->infty of y(t) = A*(1 - exp(-t/T))
%                            = A    as t -> infty
% hope that made sense - writing maths in matlab is horrible
% the point is that the final value of the normalised plant is (close to) A
% however, there is some noise. Let's take the average of the last five
% samples to mitigate that effect:
A = mean(step_data(end-5:end))

% we still don't know what the value of T is.
% however, if we look at where t = T, then,
%        y(T) = A*(1 - exp(-T/T)) = A*(1 - exp(-1)) = A*0.6321
% so, let's find the point where the graph is roughly equal to 0.6321A and
% take the time value at that point as our T.
% we'll do this by taking the absolute value of the difference between the
% step data and the 0.6321A point, and finding the index which corresponds
% to that point
[val, idx] = min(abs(step_data - A*(1-exp(-1))));
% look up documentation for 'min' if you don't understand the line above
T = t(idx)

% and now we have our A and T values!

%% 3.5.4: System Identification - verification
% are we sure that we followed that process correctly?
% perhaps the noise messed things up, or there was something funny about
% that reading? maybe we made some sort of indexing error!
% it's good practice to (somehow) verify that your model is accurate.
% let's do it as follows:
%       1. get another step response (we'll skip the part about normalising
%          by step size + shifting the time so that the step is at t = 0).
%          For the purposes of this tutorial, we'll create a 'new' reading
%          with new noise. In reality, it should be a new step from the lab
noise = randn(1, length(t))/15;  % normally distributed pseudorandom nums
step_data2 = 7.3215*(1 - exp(- t/3.916)) + noise;

%       2. create our own step response using the A and T values from the
%          previous step response data. We can use our knowledge of what
%          the step response for a first order low pass system SHOULD look
%          like
predicted_step_response = A*(1 - exp(-t/T));

%       3. finally, let's compare them
plot(t, step_data2); grid on; hold on;
plot(t, predicted_step_response);
legend('raw data from lab', 'what our model suggests')

% that looks pretty good!
% however, some of you may have noticed this method is limited in how
% accurate it can be.

% The maximum resolution of our T value is limited by our sampling time of
% Ts = 200ms. We could increase the accuracy by taking the average of
% multiple steps, though there are situations in which you only have one or
% two steps available (say you're short on time or money)

% another issue is that taking the A value as the final value isn't
% incredibly accurate either - what if the plant was still settling, or if
% the noise mucked things up?

% the actual values of A and T are 7.3215 and 3.916 respectively - let's
% see if we can do better!

%% 3.5.5: System Identification - grid search
disp('------------------------------------------------')
% let's think about how we verified the A and T values in the previous step

% we could, in principle, try out a whole bunch of A and T values and
% compare them to the lab data each time. We'd then choose the values of A
% and T which minimize the difference between our predicted model and the
% actual model

% Let's use our previous A and T values as a starting point (after all,
% they were pretty close) and then try out a large number of combinations
% until we get a pair that work really well

% we'll also need a numeric result which indicates whether we're close or
% not - how about the mean square error? (aka difference between the model
% and the real thing, squared). You could try other metrics! There's
% nothing magical about the the mse.
fprintf('A_before = %d, T_before = %d\n', A, T)

A_range = A + (-0.4:0.001:0.4); % vector of values above and below A
T_range = T + (-0.4:0.001:0.4);

A_best = A;
T_best = T;
error_best = inf;

for A_test = A_range
    for T_test = T_range 
        model_test = A_test*(1 - exp(-t/T_test));
        error_test = mean((step_data - model_test).^2);
        
        if error_test < error_best
            A_best = A_test;
            T_best = T_test;
            error_best = error_test;
        end
    end
end

plot(t, step_data); hold on;
plot(t, A*(1 - exp(-t/T)));
plot(t, A_best*(1 - exp(-t/T_best)));
grid on;
legend('lab data', 'previous A and T', 'new A and T')
fprintf('A_after = %d, T_after = %d\n', A_best, T_best)
fprintf('A_actual = %d, T_actual = %d\n', 7.3215, 3.916)

% are the new A and T values better? are they worse? try comparing them to
% step_data2

%% 3.6.1: Bode plots
close all;
% Bode plots (plots of the magnitude and phase response of a system to a
% given input frequency) can be found as follows:
bode(1/(s + 1), 1/(s^2 + 2*s + 1)); grid on;
legend('first order system', 'second order system')

%% 3.6.2: Using Bode plots
disp('------------------------------------------------')
% Bode plots are pretty and all, but how do we actually use them?
tf_complex_poles = 1/(s^2 + 0.1*s + 4)  % poles at s = -0.05 +- 2i
bode(tf_complex_poles); grid on;

% look at figure 1 first

% right click on the plot -> characteristics to view more options

% note how if the system has complex poles, there will be a spike in the
% magnitude of the plot at the associated frequency. The phase also starts
% changing rapidly at that point

% takeaways from reading the graph:
%   1.
%       given an input sinusoid with a frequency of 2 rad/s, the system
%       will oscillate with a magnitude of 14dB, with a phase lag of about
%       90 degrees
%   2.
%       there is low gain to a DC input (ie frequency of zero). Roughly
%       -12dB. Convert from dB's: 10^(-12/20) = 0.25

% let's look at figure 2 to see whether these conclusions are true -
figure(2); step(tf_complex_poles)
% high frequency component --> true
% low DC gain of 0.25 --> true

figure(1); shg

%% 3.7.1: Root locus - simple visulation
disp('------------------------------------------------')
% the rlocus() function takes a plant/system as input, and shows how the
% poles of the system move as the gain changes. Type 'help rlocus' for more
% info and a quick visualisation of the block diagram
sys = zpk(-5, [-2, -3], 2)
rlocus(sys); shg;

%% 3.7.2: Root locus - understanding what it is
disp('------------------------------------------------')
% each colour shows how a pole 'moves' as the gain changes.
% as a simple illustration, consider the following: a plant with,
P = (s + 10)/(s + 2);
% there is a zero at s = -10, and a pole at s = -2

% now, close the loop and insert a simple controller with K(s) = k
% the loop gain is then,
%       L = K*P = k*(s + 1)/(s + 2)
% the closed loop transfer function from input r to output y is,
%       T_y_r = KP/(1 + KP) = k(s + 1) / (s + 2 + k(s + 1))
% note how, no matter what value k is, the system's zeros stays at s = -1
% however, let's try some values of k and see how the poles change:
K = 1;
T = K*P/(1 + K*P);
p1 = pole(T);  % returns the poles of the system

K = 10;
T = K*P/(1 + K*P);
p2 = pole(T);

K = 0.1;
T = K*P/(1 + K*P);
p3 = pole(T);

fprintf('k = 1: pole = %d\n', p1(1))
fprintf('k = 10: pole = %d\n', p2(1))
fprintf('k = 0.1: pole = %d\n', p3(1))

% now look at the root locus plot. Click on the line and drag your mouse
% around. The gain/pole values in the popup should correspond to what we
% calculated
rlocus(P)

% finally, notice how the poles of the closed loop the system are close to
% the plant's open loop poles when the gain is close to zero, and close to
% the plant's open loop zeros when the gain is very high

%% 3.7.3: Root locus - using SISOTOOL
disp('------------------------------------------------')
% this is a GUI interface which helps you design controllers using the root
% locus method. There are many ways of launching it, but this is my
% preferred method:
% define your plant, G:
G = 22.2/s/(s + 9.8)

% launch SISOTOOL, while telling it what your plant is:
sisotool(G)
% it may take a while to load

% root locus editor:
%       click and drag on the purple dots to move the poles around. Note
%       that by doing this, what you're REALLY doing is adjusting the gain
%       of the controller.
%       you can add poles and zeros (ie define a controller) by right
%       clicking on the plot.
%       note that you can also add design requirements - you get these by
%       interpreting your '2% response' type specs

% bode editor:
%       similar to the previous plot - right click for more options

% step response:
%       this is initially the output response for a step input.
%       you can add design requirements and view peak response, etc.
%       be aware that your real world performance will almost be worse due
%       to component tolerances, modelling inaccuracies, etc

% Responses:
%       this is the list thing on the left.
%       use this to look at other types of step responses - for example,
%       right click 'IOTransfer_r2u' to see how the input to the plant (u)
%       reacts to a step change in the reference signal. Watch this
%       carefully - the value shouldn't go above the maximum input you can
%       give to your helicopter system. It is unrealistic to think that you
%       could have eg. 10000000W of power as an input. Everything has a
%       limit!
%       another useful one is 'IOTransfer_du2y'. This shows how the output
%       responds to a step input disturbance. For disturbance rejection,
%       this should go to zero! Think about this and make sure you
%       understand it. If you don't, ask Sammy or a tutor.

% Controllers and Fixed Blocks:
%       use this to input and existing controller, or read off your current
%       controller

% These are just the essentials - you should be able to figure out how to
% do everything else in SISOTOOL. Otherwise, ask for help

% It is NOT suggested that you just mess around with SISOTOOL until you get
% something that works. Use the theory, equations, etc to guide your
% choices and explain how you got to your answer. I can guarantee you that
% the more you spend time understanding the theory and your system, the
% less time you'll spend frustrated in the lab

%% 3.8: Nyquist plots
disp('------------------------------------------------')
% Next up are Nyquist plots. Their main use is tell you more about the
% stability of a given transfer function.

% This isn't the best place to try explain how the other parts of Nyquist
% work (comparing open loop poles, circulations, etc) so I'd suggest
% reading up on that to properly understand what's going on

% I'll be honest - I haven't used Nyquist plots all that much myself (bear
% in mind that I'm only one year ahead of you, at time of writing). So,
% I'll do things slightly differently and refer you to the wikipedia page:
% https://en.wikipedia.org/wiki/Nyquist_stability_criterion#Nyquist_plot

% Don't worry trying to understand everything - just read the intro
% paragraphs. This part,
% https://en.wikipedia.org/wiki/Nyquist_stability_criterion#The_Nyquist_criterion
% and the part after are also quite useful, though there are other methods
% (such as shifting the poles of the system to be off the j omega axis)

% anyway - on to plotting:
tf1 = tf(1, [1, 0.3, -0.1])
nyquist(tf1); grid on; shg

% notice how the grid corresponds to gain, and that the gain gets higher
% closer to the (-1, 0) point. Using the theory - is this system stable?

% also, note that you can right click on the plot to get more info

%% 3.9.1: State Space - introduction
disp('------------------------------------------------')
% the best place to start with state space modelling is to read through
% some of the documentation
help ss

%% 3.9.2: State Space - Creating state space models
disp('------------------------------------------------')
% from A, B, C, D matrices:
A = [-1.3333 -1.6667
     1.0000  0];
B = [1
     0];
C = [0.3333 0.6667];
D = [0];
sys = ss(A, B, C, D)

% you could get A, B, C, D from modelling your plant (eg differential
% equations) or from numerator and denominator coefficients:
[A, B, C, D] = tf2ss([1, 2], [3, 4, 5]);

% get them from an existing transfer function as follows:
t = tf([1, 2], [3, 4, 5]);
[A, B, C, D] = tf2ss(t.num{:}, t.den{:});

% convert a state space model to a transfer function:
[num, den] = ss2tf(A, B, C, D);
t = tf(num, den) *3/3  % need to put this here to normalise for some reason

%% 3.9.3: State Space - Creating a controller
disp('------------------------------------------------')
% say that, after having done all your calculations, you know where you
% want to place your poles in a state-feedback controller.
% for this, you can use the place() command:
help place

% the documentation for acker() is better, though. Try,
% >> help acker
% (I'm not sure whether there's a difference between the two)

% for example: say you want to put the poles of your system at s = -5, -7
K = place(A, B, [-5, -7])

%% Finished!
% good work!
% now do 'Intro_to_contro_using_simulink.m'
