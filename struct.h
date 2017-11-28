/*!
 \file  struct.h
 \brief This file contains structures needed in the program
 */

#ifndef _HMM_STRUCT_H_
#define _HMM_STRUCT_H_

#include "includes.h"

/*-------------------------------------------------------------
 * The following data structure stores a HMM model
 *-------------------------------------------------------------*/

typedef struct sg_hmm_t {
	long N, M;
	double **A, **B, *pi;
} sg_hmm_t;

/*-------------------------------------------------------------
 * The following data structure stores an observation sequence
 *-------------------------------------------------------------*/

typedef struct sg_obs_t {
	long N, T;
	long *seq;
	double **alpha, **beta, **gamma, ***digamma, *scale;
} sg_obs_t;

#endif
