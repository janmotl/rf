% Test choice function

% OK
choice(5,5,15);
choice(10,5,15);
choice(5,10,15);

% OK edge scenarios
choice(5,10,0);
choice(5,10,50);

% Assert non-zero rows (with cnt>=m)
a = choice(10,5,10);
assert(all(sum(a,2)>0))

% Assert non-zero cols (with cnt>=n)
a = choice(10,5,5);
assert(all(sum(a,1)>0))

% Assert row uniqueness (with cnt>=(n-1) && m<=n)
a = choice(5,10,12);
assert(size(unique(a,'rows'), 1) == size(a, 1))

% Assert col uniqueness (with cnt>=(n-1) && m>=n)
a = choice(10,5,12);
assert(size(unique(a','rows'), 1) == size(a, 2))

% Bad (expected to fail because of invalid arguments)
% choice('a',10,15)
% choice(5.1,10,15)
% choice(0,10,15)
% choice(5,0,15)
% choice(5,10,-1)
% choice(5,10,100)




