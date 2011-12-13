
#ifdef __cplusplus
extern "C" {
#endif


void
gpu_calc_Z (const double *S,
	    int           m,
	    int           n,
	    const double *mu,
	    const double *beta,
	    const double *sigma,
	    double       *Z);

#ifdef __cplusplus
}
#endif



