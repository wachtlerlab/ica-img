function gb = gamrnd2(m,k,n1,n2)
%RANDGAMMA(n1,n2,m,k) Returns (n1 x n2) matrix of random gamma(m,k) deviates
%                     All inputs are scalar.
%
%         NB: Some authors differ on how parameters enter into the
%             gamma cdf.  Here, mean(gamma)=m*k, var(gamma)=m*(k^2).
%
%  Program by Michael Gordy, 15 Sept 1993
%             mbgordy@athena.mit.edu
%
%  Source: Luc Devroye, Non-Uniform Random Variate Generation, 
%                   New York: Springer Verlag, 1986, ch 9.3-6.
%
%  Modified by Lewicki to use same parameter convention as matlab gamrnd.
%  Note: The matlab gamrnd can take a very long time for large matrices
%        This program does not appear to have to same problem.


% use same convention as matlab for gamma params.
k = 1/k;

% check for special case of an exponential distribution
if (m == 1)
  gb = exprnd(k,n1,n2);
  return
end

gb=zeros(n1,n2);
if m<1
  % Use RGS algorithm by Best, p. 426
  c=1/m; 
  t=0.07+0.75*sqrt(1-m);
  b=1+exp(-t)*m/t;
  for i1=1:n1
    for i2=1:n2
       accept=0;
       while accept==0
          u=rand; w=rand; v=b*u;
          if v<=1
             x=t*(v^c);
             accept=((w<=((2-x)/(2+x))) | (w<=exp(-x)));
          else
             x=-log(c*t*(b-v));
             y=x/t;
             accept=(((w*(m+y-m*y))<=1) | (w<=(y^(m-1))));
          end
       end
       gb(i1,i2)=x;
    end
  end
else
  % Use Best's rejection algorithm XG, p. 410
  b=m-1;
  c=3*m-0.75;
  for i1=1:n1
    for i2=1:n2
       accept=0;
       while accept==0
          u=rand;  v=rand;
          w=u*(1-u);  y=sqrt(c/w)*(u-0.5);
          x=b+y;
          if x>=0
             z=64*(w^3)*v*v;
             accept=(z<=(1-2*y*y/x)) ...
                    | (log(z)<=(2*(b*log(x/b)-y)));
          end
       end
       gb(i1,i2)=x;
    end
  end
end
gb=gb/k;    
    

