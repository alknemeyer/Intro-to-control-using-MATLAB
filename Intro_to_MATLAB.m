%% Introduction to MATLAB
% For third year UCT control students
% Please give feedback on the document (error corrections + things that
% were explained badly)

% Author:       Alexander Knemeyer
% Date:         3 September, 2018
% Contact:      KNMALE002@myuct.ac.za
E
%% 0: how to use this document
%% 0.1: run things line by line
% This document is NOT meant to be run as a script in one go.
% Instead, it should be run line-by-line.
% Put the cursor in a cell to highlight a block (it should turn yellow).
% Then, press 'ctrl+enter' to run the code in the block.
% Try it now - the code below should run, and print out 'test' in the
% Command Window on the right.
disp('test')

% Also, you should read all the text in the cell BEFORE running it. That's
% vital! Otherwise you'll see stuff in the Command Window and not know what
% it is

% if the cell has the plot() command in it, it's usually best to read the
% text until you get to the plot() command, then run it, then carry on
% reading the text in the cell

%% 0.2: add or remove semicolons to make things display or not
disp('------------------------------------------------')
% Lines which end without a semicolon will have the result printed to the
% screen - see the difference between the two lines below:
this_will_display = 123
this_wont_display = 123;

%% 0.3: how to get help/documentation for any function
disp('------------------------------------------------')
% Type 'help' and then the function name into the Command Window
% Ie the general format is:
% >> help function_name

% Eg the 'linspace' command is useful for creating linearly spaced values
help linspace

% Otherwise, use the matlab website:
% https://uk.mathworks.com/help/matlab/ref/linspace.html

%% 1: basic MATLAB usage
%% 1.0: the MATLAB workspace
% MATLAB is a bit different from some other types of programming languages.
% You can work with it interactively.
% When you save a variable, it gets put into the Workspace
x = 1;
% You can then access it later.
% This guide should be run one cell at a time, and may not work properly if
% you skip cells or run the same cell multiple times.
% If this turns out to be an issue, message me and I'll update the guide
% accordingly.

%% 1.1: creating column vectors
disp('------------------------------------------------')
% rows are separated using semicolons or new lines
cols1 = [1; 2; 3]; % same as below
cols2 = [1
         2
         3]
% cols1 == cols2

%% 1.2: creating row vectors
disp('------------------------------------------------')
% columns are separated using commas or spaces
rows1 = [1, 2, 3]; % same as below
rows2 = [1  2  3]


%% 1.3: creating matrices
disp('------------------------------------------------')
matrix1 = [1, 2, 3; 4, 5, 6; 7, 8, 9];
matrix2 = [1 2 3
           4 5 6
           7 8 9]

identity_matrix = eye(3)  % '3' is the size of the matrix
matrix_of_zeros = zeros(2)
matrix_of_ones = ones(4)

%% 1.3: indexing vectors and matrices
disp('------------------------------------------------')
rows3 = ['a', 'b', 'c', 'd'];

% vectors are one-indexed in MATLAB, which means you access the ith element
% using variable(i). For example,
rows3(1)  % returns 'a', the first element

% you can access the last element using the keyword 'end'
rows3(end) % returns 'd', the last element

% or the second to last
rows3(end - 1)  % returns 'c', the second last element

% access multiple elements at once
rows3([1, 3])  % returns 'a', 'c'

%% 1.4: multiplying matrices
disp('------------------------------------------------')

x = [1, 2; 3, 4];
y = [1 2
     0 3];

% regular matrix-matrix multiplication
regular_matrix_multiplication = x * y

% element-wise matrix multiplication - add a dot before the operator
element_wise_multiplication = x .* y

% in general, adding a dot before any operator should make it act
% element-wise

%% 1.5: transposing vectors
disp('------------------------------------------------')
% vectors can be transposed using the ' operator
row_vector = [1, 2, 3]
col_vector = row_vector'


%% 1.6: creating linearly spaced data
disp('------------------------------------------------')
% the format is start:step:stop
linearly_spaced_values = 0.1:0.2:0.9  % note that this is a row vector

% the format is linspace(start, stop, number of values)
linearly_spaced_values2 = linspace(0.5, 10.5, 5)

%% 1.7: working with text
disp('------------------------------------------------')
% this won't be covered in detail as you'll likely only need the basics.

% strings are denoted by single quotes, as in
disp('I am text')

% they can be printed to the screen with the 'disp' command.
% concatenate strings by popping them into an array.
% numbers should be converted to strings using the num2str command.
disp(['The number is ', num2str(123), '!'])

% however, there are better ways of working with text.
% for example, look up 'sprintf'.
% >> help sprintf
% note that it doesn't actually print the string - it just creates it.
my_str = sprintf('decimal number: %d, float: %.2f', 12, 12.345);
disp(my_str)
% what are 'd' and 'f'? Find out from documentation instead of asking
% someone else!

% another option is to use 'fprintf', though you have to manually add a new
% line character at the end:
fprintf('test of fprintf: %i. New line character-->\n', 5)

%% 1.8: write multiple commands on one line
disp('------------------------------------------------')
x = 1; y = 2; disp(sprintf('x = %d, y = %d, in one line', x, y));

%% 1.9: complex numbers
disp('------------------------------------------------')
% these can be written by multiplying by '1i'
complex_1 = 1 + 2*1i
complex_2 = 3 - 2*1i;

% arithmeic, matrices, etc - they all work as expected
prod_of_complex_nums = complex_1 * complex_2

%% 1.10: flow control (for, if, elseif, else, while)
disp('------------------------------------------------')
% don't worry to much about understanding/memorizing this - just know that
% they exist, and that you can refer back to here/online if need be.
for i = 1:5
    if (i == 3)
        disp(num2str(i))
    elseif (i == 4)
        disp('i is 4')
    else
        disp('i isn''t 3 or 4')
    end % if and for statements must finish with 'end'
end

j = 3;
while (j > 0)
    j = j - 1;
    fprintf('j = %i\n', j)
end

%% 1.11: some other useful commands
disp('------------------------------------------------')
x = [1, 2, 3, 4, 5];
length(x)  % number of items in x

y = [1 2 3
     4 5 6];
size(y)  % dimensions of the vector

% have any suggestions?

%% Finished!
% good work!
% now do 'Intro_to_plotting_using_MATLAB.m'
