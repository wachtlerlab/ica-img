ds = 0.01;
s = [-10:ds:10];

beta=1;
theta=1;

cosh_pdf = coshpdf(s,beta,theta);

figure(1);
plot(s,cosh_pdf);

z = ds*trapz(cosh_pdf);

fprintf('integral z=%f\n', z);
