%CHOICE Stratified random binary matrix.
%   CHOICE(M, N, CNT) returns M-by-N matrix consisting of {0,1} where
%   the count of ones is given by CNT.
%
%   The nice property of this generator is that it is stratified by 
%   both, rows and columns. In other words, the count of ones in each
%   row/column differs at most by one. This property is desirable,
%   for example, when we want to be sure that there isn't any "zero" row
%   (guaranteed when CNT >= M). Or when we want to be sure that each
%   row is unique (guaranteed when CNT >= M-1 && M <= N. Whenever
%   applicable, these constraints are enforced for both, rows and columns,
%   at the same time.
%
%   Example call:
%       mask = choice(6,4,7)  % Select 7 items from 6x4 matrix
% 
%   Example of usage:
%      x = rand(6,4);  % Some matrix
%      mask = choice(6,4,7)  % Get the mask
%      x(mask>0)  % A vector of the selected items
%
%   See also RAND.

%   Jan Motl, 2019.

function result = choice(m, n, cnt)

% Argument validation
validateattributes(m, {'numeric'}, {'integer', 'scalar', '>', 0})
validateattributes(n, {'numeric'}, {'integer', 'scalar', '>', 0})
validateattributes(cnt, {'numeric'}, {'integer', 'scalar', '>=', 0, '<=', m*n})

% Create a banded rectangular matrix, where the bands overflow.
% The overflow is reminiscent of Sarrus' rule (the textbook calculation
% of determinant on 3x3 matrix). The matrix may contain multiple bands.
result = zeros(m, n);
row = 1;
col = 1;
for i=1:cnt
    if result(row, col) == 1
       row=row+1; 
    end
    result(row, col) = 1;   
    row = mod(row, m)+1;
    col = mod(col, n)+1;
end
    
% Add randomness
result = result(randperm(m), randperm(n));

% Validate the contract
assert(sum(result(:)) == cnt, 'The count of ones is as given')
assert(max(sum(result)) - min(sum(result)) <= 1, 'The count of ones in each column differs at most by one')
assert(max(sum(result,2)) - min(sum(result,2)) <= 1, 'The count of ones in each row differs at most by one')
