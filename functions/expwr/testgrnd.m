dx = 0.1;
x = 0:dx:15;
hx = x + dx/2;
n = 10000;

s = 1;
p = 0.75;
y = gexppdf(x,s,p);
r = gexprnd(s,p,1,n);

figure(1);
plot(x,y);

figure(2);
hist(r,hx);
