/*
 * \file proto.h
 * \brief This file contains function prototypes
 */

#ifndef _SRC_PROTO_H_
#define _SRC_PROTO_H_

#include "includes.h"

void sg_hmm_Init(sg_hmm_t *mod);
sg_hmm_t* sg_hmm_Create(void);
void sg_hmm_InitRandom();
void sg_hmm_InitRandomWithoutA();
void sg_hmm_Print(sg_hmm_t *mod);
void sg_obs_Init(sg_obs_t *obs);
sg_obs_t* sg_obs_Create(void);
void sg_obs_Allocate(sg_obs_t *obs);
void sg_obs_Print(sg_obs_t *obs);
void sg_train(sg_hmm_t *mod, sg_obs_t *obs, int max_itr);
void sg_trainWithoutA(sg_hmm_t *mod, sg_obs_t *obs, int max_itr);
double _computeLogScore(sg_obs_t *obs);
void _calculateAlphaPass(sg_hmm_t *mod, sg_obs_t *obs);
void _calculateBetaPass(sg_hmm_t *mod, sg_obs_t *obs);
void _calculateGammas(sg_hmm_t *mod, sg_obs_t *obs);
void _reEstimateModel(sg_hmm_t *mod, sg_obs_t *obs);
void _reEstimateModelWithoutA(sg_hmm_t *mod, sg_obs_t *obs);
long sg_hmm_predict(sg_hmm_t *mod, sg_obs_t *obs);

#endif /* SRC_PROTO_H_ */
