% Like choice, but returns booleans.
% And respects mtry (not just at row but at matrix level).

function result = choiceMtry(n, mtry, repeats)

% Get something
k = ceil(mtry*n);
proposal = choice(n, k, repeats);

% To boolean indexing
result = nan(repeats, n);
for row=1:repeats
    result(row, :) = ismember(1:n, proposal(row, :));
end

% Remove some samples to get mtry
removeCount = round(repeats*(k-mtry*n));

for row = 1:removeCount
    choices = find(result(row,:)==1);
    col = choices(randi(length(choices)));
    result(row, col) = 0;
end
