e = gexpp_fmin(p,x,s,a,b)

n = length(x);
e = s - (a + n) / (b + sum(x.^(1/p)))
e = e*e;
