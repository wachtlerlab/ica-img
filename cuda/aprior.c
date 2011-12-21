#include "cube.h"
#include "cube_blas.h"
#include "cube_matlab.h"
#include "cube_ica.h"
#include "cube_ica_kernels.h"

#include <matrix.h>
#include <mat.h>

#include <string.h>

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

double
dnorm_centered (double *x, int n, int incx, double p, double mu, double sigma)

{
  int i;
  double s;

  s = 0;
  for (i = 0; i < n; i += incx)
    {
      double u = (x[i] - mu) / sigma;
      double t;

      t = pow (fabs(u), p);
      s += t;
    }

  return s;
}

double
exp_pwr_l_beta (double beta, double *y, int n, double incy, double mu, double sigma, double a, double b)
{
  double u, c, logw, uas, l;
  double p;

  p = 2.0/(1+beta);

  c = pow ((tgamma(3.0/p)/tgamma(1/p)), (p/2.0));
  logw = 0.5 * lgamma (3.0/p) - 1.5 * lgamma (1/p) - log (1+beta);

  uas = dnorm_centered (y, n, incy, p, mu, sigma);
  l = (-1*lgamma(a)) - (a*log(b)) + ((a-1.0)*log(1.0+beta)) +
    n*logw - n*log(sigma) - ((1.0+beta)/b) - (c * uas);

  return l;
}


typedef struct _exp_param
{
  double *y;
  int    n;
  int    incy;
  
  double mu;
  double sigma;
  double a;
  double b;

} exp_param;

double
exp_pwrlbeta_fmin (double x, exp_param params)
{
  double *y;
  int n, incy;
  double mu, sigma, a, b;
  double l, beta;

  y     = params.y;
  n     = params.n;
  incy  = params.incy;
  mu    = params.mu;
  sigma = params.sigma;
  a     = params.a;
  b     = params.b;

  beta = exp (x) - 1;

  l = -1 *  exp_pwr_l_beta (beta, y, n, incy, mu, sigma, a, b);
  
  return l;
}

double hsign (double v)
{
  if (v > 0)
    return 1;
  else if (v < 0)
    return -1;
  else
    return 0;
}

/* No input checking, ax > bx required  */
double
f_min_bound (double ax, double bx, double tol, exp_param fparams)
{
  double tol1, tol2;
  int maxfun,  maxiter;
  double seps;
  double a, b, c, d, e;
  double v, fv, w, fw, x, xf, fx, xm;
  int count, iter;

  count = 0.0;

  maxfun = 500.0;
  maxiter = 500.0;
  
  seps = 1.4901e-08; /* machine espilon, squared, FIXME */
  
  c = 0.5 * (3.0 - sqrt (5.0));
  a = ax;
  b = bx;
  v = a + c * (b - a);
  w = v;
  xf = v;
  d = 0.0;
  e = 0.0;
  x = xf;

  fx = exp_pwrlbeta_fmin (x, fparams);
  count++;

  fv = fx;
  fw = fx;
  xm = 0.5 * (a + b);
  tol1 = seps * fabs (xf) + tol/3.0;
  tol2 = 2.0 * tol1;

  while (fabs (xf- xm) > (tol2 - 0.5*(b-a)))
    {
      double r, q, p, si, fu;
      int gs = 1;

      if (fabs(e) > tol1)
	{
	  gs = 0;

	  r = (xf-w)*(fx-fv);
	  q = (xf-v)*(fx-fw);
	  p = (xf-v)*q-(xf-w)*r;
	  q = 2.0*(q-r);
	  
	  if (q > 0.0)
	    p = -p;

	  q = fabs(q);
	  r = e; e = d;

	  if ((fabs(p) < fabs(0.5*q*r)) &&
	      (p > q*(a-xf)) &&
	      (p < q*(b-xf)))
	    {
	      d = p/q;
	      x = xf + d;

	      if (((x-a) < tol2) || ((b-x) < tol2))
		{
		  si = hsign (xm-xf) + ((xm-xf) == 0);
		  d = tol1 * si;
		}
	    }
	  else
	    {
	      gs = 1;
	    }
	}

      if (gs)
	{
	  if (xf >= xm)
	    e = a-xf;
	  else
	    e = b-xf;

	  d = c*e;
	}

      si = hsign(d) + (d == 0);
      x = xf + si * fmax (fabs(d), tol1);

      fu = exp_pwrlbeta_fmin (x, fparams);
      count++;
      iter++;

      if (fu <= fx)
	{
	  if (x >= xf)
	    a = xf;
	  else 
	    b = xf;

	  v = w; fv = fw;
	  w = xf; fw = fx;
	  xf = x; fx = fu;
	}
      else // fu > fx
	{
	  if (x < xf)
	    a = x;
	  else
	    b = x;
	       
	  if ((fu <= fw) || (w == xf))
	    {
	      v = w; fv = fw;
	      w = x; fw = fu;
	    }
	  else if ((fu <= fv) || (v == xf) || (v == w))
	    {
	      v = x; fv = fu;
	    }
	  
	}

      xm = 0.5 * (a+b);
      tol1 = seps * fabs(xf) + tol/3.0;
      tol2 = 2.0 * tol1;
      
      if (count > maxfun || iter > maxiter)
	{
	  break;
	}
    }

  return xf;
}


/* Maximum aposteriori estimate of exponential power parameters */
double
exp_pwr_map_beta (double tol, exp_param params)
{
  double betamin, betamax;
  double xmin, xmax;
  double beta;
  double x;

  betamin = -0.9;
  betamax = 20.0;

  xmin = log (1 + betamin);
  xmax = log (1 + betamax);

  x = f_min_bound (xmin, xmax, tol, params);

  beta = exp(x) - 1;
  return beta;
}


#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))

void
mat_array_dump (mxArray *matrix, int m_max, int n_max)
{
  double *A;
  int m;
  int n;
  int row, col;

  A = (double *) mxGetData (matrix);

  m = MIN (mxGetM (matrix), m_max);
  n = MIN (mxGetN (matrix), n_max);

  for (row = 0; row < m; row++)
    {
      for (col = 0; col < n; col++)
	{
	  int pos = (col * mxGetM(matrix)) + row;
	  printf ("%lf ", A[pos]); //A[IDX2F(row, col, n)]);
	}

      printf ("\n");
    }
  printf ("\n");
}

int
main(int argc, char **argv)
{
  cube_t  *ctx;
  cube_matfile_t *fd;
  char *filename;
  mxArray *y, *p, *mu, *sigma, *uas_r, *a, *b, *beta, *x;
  double uas, l, fmin, be;
  exp_param params;
  cudaEvent_t start, stop;
  float elpased;

  ctx = cube_context_new ();

  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  filename = argv[1];

  fd = cube_matfile_open (ctx, filename);

  cube_matfile_get_vars (ctx, fd, "y", &y, "p", &p, "sigma", &sigma, "mu", &mu, "uas", &uas_r, "a", &a, "b", &b, "beta", &beta, "x", &x, NULL);

  mat_array_dump (y, 10, 10);
  mat_array_dump (p, 10, 10);
  mat_array_dump (mu, 10, 10);
  mat_array_dump (sigma, 10, 10);

  uas = dnorm_centered (mxGetPr(y), mxGetN (y), 1, mxGetPr(p)[0], mxGetPr(mu)[0], mxGetPr(sigma)[0]);

  printf ("uas: %e (%e)\n", uas, mxGetPr(uas_r)[0]);

  uas = gpu_sumpower (ctx, mxGetPr(y), mxGetN (y), mxGetPr(p)[0]);
  printf ("uas(gpu): %e \n", uas);

  l = exp_pwr_l_beta (mxGetPr(beta)[0], mxGetPr(y), mxGetN (y), 1, mxGetPr(mu)[0], mxGetPr(sigma)[0], mxGetPr(a)[0], mxGetPr(b)[0]);
  
  printf ("l: %e\n", l);

  params.y = mxGetPr(y);
  params.n = mxGetN (y);
  params.incy = 1;

  params.mu = mxGetPr(mu)[0];
  params.sigma = mxGetPr(sigma)[0];
  params.a = mxGetPr(a)[0];
  params.b = mxGetPr(b)[0];

  fmin = exp_pwrlbeta_fmin (mxGetPr(x)[0], params);
  printf ("fmin: %e\n", fmin);

  be = exp_pwr_map_beta (0.1, params);
  
  printf ("beta: %e\n", be);

  cudaEventRecord(start, 0);
  be = gpu_aprior (ctx, params.y, params.n, 0.1, params.a, params.b);
  cudaEventRecord(stop, 0);
  cudaEventElapsedTime(&elpased, start, stop);

  printf ("beta: %e [%f in ms (%f)]\n", be, elpased, elpased/1000.0);

  return 0;
}
