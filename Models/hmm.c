/*!
 * \file  hmm.c
 * \brief Functions for dealing with HMM model
 */
#include "includes.h"

/*************************************************************************/

/*! Initializes the model
 * \param mod is the model to be initialized.
 */
/*************************************************************************/
void
sg_hmm_Init(sg_hmm_t * mod)
{
    memset(mod, 0, sizeof(sg_hmm_t));
    mod->N = mod->M = -1;
}

/*************************************************************************/

/*! Allocate memory for a HMM model and initializes it
 * \returns the allocated model. The various fields are set to NULL.
 */
/**************************************************************************/
sg_hmm_t *
sg_hmm_Create(void)
{
    sg_hmm_t * mod;

    mod = (sg_hmm_t *) malloc(sizeof(sg_hmm_t));
    sg_hmm_Init(mod);
    return mod;
}

/*************************************************************************/

/*! After updating model with values of N and M, initialize matrices A, B
 * and π.
 * \returns the allocated model. The various fields are set to NULL.
 */
/**************************************************************************/
void
sg_hmm_InitRandom(sg_hmm_t * mod, long seed)
{
    if (mod->N < 1 || mod->M < 1) {
        printf("Values of N and/or M are invalid! Returning...\n");
        return;
    }
    double baseN = 1.0 / mod->N, baseM = 1.0 / mod->M;
    if (seed > 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }
    double total = 0, r = 0, v = 0;

    /* Initialize π */
    mod->pi = (double *) malloc(mod->N * sizeof(double));
    for (int i = 0; i < mod->N; ++i) {
        r = rand() % 1000;
        while (r < 700) {
            r += 100;
        }
        r = r / 10000;
        v = baseN + r;
        mod->pi[i] = v;
        total     += v;
    }

    for (int i = 0; i < mod->N; ++i) {
        mod->pi[i] = mod->pi[i] / total;
    }

    /* Initialize A */
    mod->A    = (double **) malloc(mod->N * sizeof(double *));
    mod->A[0] = (double *) malloc(mod->N * mod->N * sizeof(double));
    for (int i = 0; i < mod->N; i++)
        mod->A[i] = (*mod->A + mod->N * i);

    for (int i = 0; i < mod->N; ++i) {
        total = 0;
        for (int j = 0; j < mod->N; ++j) {
            r = rand() % 1000;
            while (r < 700) {
                r += 100;
            }
            r = r / 10000;
            v = baseN + r;
            mod->A[i][j] = v;
            total       += v;
        }

        for (int j = 0; j < mod->N; ++j) {
            mod->A[i][j] = mod->A[i][j] / total;
        }
    }

    /* Initialize B */
    mod->B    = (double **) malloc(mod->N * sizeof(double *));
    mod->B[0] = (double *) malloc(mod->N * mod->M * sizeof(double));
    for (int i = 0; i < mod->N; i++)
        mod->B[i] = (*mod->B + mod->M * i);

    for (int i = 0; i < mod->N; ++i) {
        total = 0;
        for (int j = 0; j < mod->M; ++j) {
            r = rand() % 1000;
            while (r < 700) {
                r += 100;
            }
            r = r / 10000;
            v = baseM + r;
            mod->B[i][j] = v;
            total       += v;
        }

        for (int j = 0; j < mod->M; ++j) {
            mod->B[i][j] = mod->B[i][j] / total;
        }
    }
} /* sg_hmm_InitRandom */

/*************************************************************************/

/*! After updating model with values of N and M, initialize matrices B
 * and π.
 * \returns the allocated model. The various fields are set to NULL.
 */
/**************************************************************************/
void
sg_hmm_InitRandomWithoutA(sg_hmm_t * mod, long seed)
{
    if (mod->N < 1 || mod->M < 1) {
        printf("Values of N and/or M are invalid! Returning...\n");
        return;
    }
    // if(seed > 0) {
    //  srand(seed);
    // }else {
    //	srand(time(NULL));
    // }
    double total = 0, r = 0;

    /* Initialize π */
    mod->pi = (double *) malloc(mod->N * sizeof(double));
    for (int i = 0; i < mod->N; ++i) {
        r = rand() % 10000;
        while (r < 9000) {
            r += 1000;
        }
        mod->pi[i] = r;
        total     += r;
    }

    for (int i = 0; i < mod->N; ++i) {
        mod->pi[i] = mod->pi[i] / total;
    }

    /* Initialize B */
    mod->B    = (double **) malloc(mod->N * sizeof(double *));
    mod->B[0] = (double *) malloc(mod->N * mod->M * sizeof(double));
    for (int i = 0; i < mod->N; i++)
        mod->B[i] = (*mod->B + mod->M * i);

    for (int i = 0; i < mod->N; ++i) {
        total = 0;
        for (int j = 0; j < mod->M; ++j) {
            r = rand() % 10000;
            while (r < 9000) {
                r += 1000;
            }
            mod->B[i][j] = r;
            total       += r;
        }

        for (int j = 0; j < mod->M; ++j) {
            mod->B[i][j] = mod->B[i][j] / total;
        }
    }
} /* sg_hmm_InitRandomWithoutA */

/**************************************************************************/

/*! Reads a HMM model from the supplied file and stores it the model's
 * forward structure.
 * \param src_file is the file pointer that stores the HMM model.
 * \returns the model that was read.
 */
/**************************************************************************/
void
sg_hmm_Print(sg_hmm_t * mod)
{
    for (int i = 0; i < 10; i++) {
        printf("*");
    }

    printf("\nN = %ld, M = %ld\n", mod->N, mod->M);

    printf("\nπ = [ ");
    for (int i = 0; i < mod->N; i++) {
        printf("%f ", mod->pi[i]);
    }
    printf("]\n");

    printf("\nA = [\n");
    for (int i = 0; i < mod->N; i++) {
        printf("      ");
        for (int j = 0; j < mod->N; j++) {
            // printf("%f\t", mod->A[mod->N*i + mod->N*j]);
            printf("%f ", mod->A[i][j]);
        }
        printf("\n");
    }
    printf("    ]\n");

    printf("\nB = [\n");
    double total[mod->N];
    for (int j = 0; j < mod->M; j++) {
        printf("%c     ", 'a' + j);
        for (int i = 0; i < mod->N; i++) {
            // printf("%f\t", mod->B[mod->N*i + mod->M*j]);
            printf("%f ", mod->B[i][j]);
            total[i] += mod->B[i][j];
        }
        printf("\n");
    }
    printf("\n      ");
    for (int i = 0; i < mod->N; i++) {
        printf("%f ", total[i]);
    }
    printf("\n    ]\n");

    for (int i = 0; i < 10; i++) {
        printf("*");
    }
    printf("\n");
} /* sg_hmm_Print */

/*************************************************************************/

/*! Initializes the observation array
 * \param obs is the array to be initialized.
 */
/*************************************************************************/
void
sg_obs_Init(sg_obs_t * obs)
{
    memset(obs, 0, sizeof(sg_obs_t));
    obs->N = obs->T = -1;
}

/*************************************************************************/

/*! Allocate memory for a HMM model and initializes it
 * \returns the allocated model. The various fields are set to NULL.
 */
/**************************************************************************/
sg_obs_t *
sg_obs_Create(void)
{
    sg_obs_t * obs;

    obs = (sg_obs_t *) malloc(sizeof(sg_obs_t));
    sg_obs_Init(obs);
    return obs;
}

void
sg_obs_Allocate(sg_obs_t * obs)
{
    if (obs->N < 1 || obs->T < 1) {
        printf("Values of N and/or T are invalid! Returning...\n");
        return;
    }

    /* Allocate space for observation sequence */
    obs->seq = (long *) malloc(obs->T * sizeof(long));
    for (int i = 0; i < obs->T; ++i) {
        obs->seq[i] = -1;
    }

    /* Allocate space for alpha pass */
    obs->alpha    = (double **) malloc(obs->T * sizeof(double *));
    obs->alpha[0] = (double *) malloc(obs->T * obs->N * sizeof(double));
    for (int i = 0; i < obs->T; i++)
        obs->alpha[i] = (*obs->alpha + obs->N * i);

    /* Allocate space for beta pass */
    obs->beta    = (double **) malloc(obs->T * sizeof(double *));
    obs->beta[0] = (double *) malloc(obs->T * obs->N * sizeof(double));
    for (int i = 0; i < obs->T; i++)
        obs->beta[i] = (*obs->beta + obs->N * i);

    /* Allocate space for gamma pass */
    obs->gamma    = (double **) malloc(obs->T * sizeof(double *));
    obs->gamma[0] = (double *) malloc(obs->T * obs->N * sizeof(double));
    for (int i = 0; i < obs->T; i++)
        obs->gamma[i] = (*obs->gamma + obs->N * i);

    /* Allocate space for digamma pass */
    obs->digamma = (double ***) malloc(obs->T * sizeof(double **));
    for (int i = 0; i < obs->T; ++i) {
        obs->digamma[i] = (double **) malloc(obs->N * sizeof(double *));
        for (int j = 0; j < obs->N; ++j) {
            obs->digamma[i][j] = (double *) malloc(obs->N * sizeof(double));
        }
    }

    /* Allocate space for scaling factor */
    obs->scale = (double *) malloc(obs->T * sizeof(double));
    for (int i = 0; i < obs->T; ++i) {
        obs->scale[i] = -1;
    }
} /* sg_obs_Allocate */

void
sg_obs_Print(sg_obs_t * obs)
{
    printf("Observations:\n");
    for (int i = 0; i < obs->T; ++i) {
        printf("%ld ", obs->seq[i]);
    }
    printf("\n");
}

void
sg_train(sg_hmm_t * mod, sg_obs_t * obs, int max_itr)
{
    int iters = 0, flag = 1;
    double old_log_prob, log_prob;

    // old_log_prob = -DBL_MAX;
    // log_prob = old_log_prob+1;
    while (max_itr-- > 0 || flag) {
        _calculateAlphaPass(mod, obs);
        _calculateBetaPass(mod, obs);
        _calculateGammas(mod, obs);
        _reEstimateModel(mod, obs);
        old_log_prob = log_prob;
        log_prob     = _computeLogScore(obs);

        iters = iters + 1;

        double delta = fabs(log_prob - old_log_prob);
        if (delta > 0.001) {
            flag = 1;
        } else {
            flag = 0;
        }
    }

    // while (flag) {
    //         _calculateAlphaPass(mod, obs);
    //         _calculateBetaPass(mod, obs);
    //         _calculateGammas(mod, obs);
    //         _reEstimateModel(mod, obs);
    //         log_prob = _computeLogScore(obs);
    //         iters = iters + 1;
    //
    //         if(log_prob <= old_log_prob) {
    //                 break;
    //         }
    //         old_log_prob = log_prob;
    // }

    printf("HMM trained in %d iterations\n", iters);
} /* sg_train */

void
sg_trainWithoutA(sg_hmm_t * mod, sg_obs_t * obs, int max_itr)
{
    int iters = 0;
    double old_log_prob, log_prob;

    old_log_prob = -DBL_MAX;
    log_prob     = old_log_prob + 1;

    while (iters < max_itr) {
        _calculateAlphaPass(mod, obs);
        _calculateBetaPass(mod, obs);
        _calculateGammas(mod, obs);
        _reEstimateModelWithoutA(mod, obs);
        log_prob = _computeLogScore(obs);
        iters    = iters + 1;
        // printf("Interation %d complete, log_prob = %f\n", iters, log_prob);
        if (log_prob <= old_log_prob) {
            break;
        }
        old_log_prob = log_prob;
    }

    printf("Iterations: %d ", iters);
}

double
_computeLogScore(sg_obs_t * obs)
{
    double log_prob = 0;

    for (int t = 0; t < obs->T - 1; ++t) {
        log_prob += log(obs->scale[t]);
    }
    log_prob = -log_prob;
    return log_prob;
}

void
_calculateAlphaPass(sg_hmm_t * mod, sg_obs_t * obs)
{
    /* compute alpha[0][i] */
    obs->scale[0] = 0;
    for (int i = 0; i < obs->N; ++i) {
        obs->alpha[0][i] = mod->pi[i] * mod->B[i][obs->seq[0]];
        obs->scale[0]   += obs->alpha[0][i];
    }

    /* scale obs->alpha[0][i] */
    obs->scale[0] = 1 / obs->scale[0];
    for (int i = 0; i < obs->N; ++i) {
        obs->alpha[0][i] *= obs->scale[0];
    }

    /* compute a[t][i] */
    for (int t = 1; t < obs->T; ++t) {
        obs->scale[t] = 0;
        for (int i = 0; i < obs->N; ++i) {
            obs->alpha[t][i] = 0;
            for (int j = 0; j < obs->N; ++j) {
                obs->alpha[t][i] = obs->alpha[t][i] + obs->alpha[t - 1][j] * mod->A[j][i];
            }
            obs->alpha[t][i] *= mod->B[i][obs->seq[t]];
            obs->scale[t]    += obs->alpha[t][i];
        }

        /* scale obs->alpha[t][i] */
        obs->scale[t] = 1 / obs->scale[t];
        for (int i = 0; i < obs->N; ++i) {
            obs->alpha[t][i] *= obs->scale[t];
        }
    }
} /* _calculateAlphaPass */

void
_calculateBetaPass(sg_hmm_t * mod, sg_obs_t * obs)
{
    /* let beta[T-1][i] be 1, scaled by c[T-1] */
    for (int i = 0; i < obs->N; ++i) {
        obs->beta[obs->T - 1][i] = obs->scale[obs->T - 1];
    }

    /* Beta pass */
    for (int t = obs->T - 2; t >= 0; --t) {
        for (int i = 0; i < obs->N; ++i) {
            obs->beta[t][i] = 0;
            for (int j = 0; j < obs->N; ++j) {
                obs->beta[t][i] += mod->A[i][j] * mod->B[j][obs->seq[t + 1]] * obs->beta[t + 1][j];
            }

            /* scale b[t][i] by same factor as a[t][i] */
            obs->beta[t][i] *= obs->scale[t];
        }
    }
}

void
_calculateGammas(sg_hmm_t * mod, sg_obs_t * obs)
{
    for (int t = 0; t < obs->T - 1; ++t) {
        double denom = 0;
        for (int i = 0; i < obs->N; ++i) {
            for (int j = 0; j < obs->N; ++j) {
                denom += obs->alpha[t][i] * mod->A[i][j] * mod->B[j][obs->seq[t + 1]] * obs->beta[t + 1][j];
            }
        }

        for (int i = 0; i < obs->N; ++i) {
            obs->gamma[t][i] = 0;
            for (int j = 0; j < obs->N; ++j) {
                obs->digamma[t][i][j] = (obs->alpha[t][i] * mod->A[i][j]
                  * mod->B[j][obs->seq[t + 1]] * obs->beta[t + 1][j]) / denom;
                obs->gamma[t][i] += obs->digamma[t][i][j];
            }
        }
    }

    /* special case for g[T-1][i] */
    double denom2 = 0;
    for (int i = 0; i < obs->N; ++i) {
        denom2 += obs->alpha[obs->T - 1][i];
    }
    for (int i = 0; i < obs->N; ++i) {
        obs->gamma[obs->T - 1][i] = obs->alpha[obs->T - 1][i] / denom2;
    }
}

void
_reEstimateModel(sg_hmm_t * mod, sg_obs_t * obs)
{
    /* reestimate pi */
    for (int i = 0; i < obs->N; ++i) {
        mod->pi[i] = obs->gamma[0][i];
    }

    /* reestimate A */
    for (int i = 0; i < obs->N; ++i) {
        for (int j = 0; j < obs->N; ++j) {
            double numer = 0;
            double denom = 0;
            for (int t = 0; t < obs->T - 1; ++t) {
                numer += obs->digamma[t][i][j];
                denom += obs->gamma[t][i];
            }
            mod->A[i][j] = numer / denom;
        }
    }

    /* reestimate B */
    for (int i = 0; i < obs->N; ++i) {
        for (int j = 0; j < mod->M; ++j) {
            double numer2 = 0;
            double denom2 = 0;
            for (int t = 0; t < obs->T; ++t) {
                if (obs->seq[t] == j) {
                    numer2 += obs->gamma[t][i];
                }
                denom2 += obs->gamma[t][i];
            }
            mod->B[i][j] = numer2 / denom2;
        }
    }
} /* _reEstimateModel */

void
_reEstimateModelWithoutA(sg_hmm_t * mod, sg_obs_t * obs)
{
    /* reestimate pi */
    for (int i = 0; i < obs->N; ++i) {
        mod->pi[i] = obs->gamma[0][i];
    }

    /* reestimate B */
    for (int i = 0; i < obs->N; ++i) {
        for (int j = 0; j < mod->M; ++j) {
            double numer2 = 0;
            double denom2 = 0;
            for (int t = 0; t < obs->T; ++t) {
                if (obs->seq[t] == j) {
                    numer2 += obs->gamma[t][i];
                }
                denom2 += obs->gamma[t][i];
            }
            mod->B[i][j] = numer2 / denom2;
        }
    }
}

long
sg_hmm_predict(sg_hmm_t * mod, sg_obs_t * obs)
{
    sg_obs_t * obs2 = sg_obs_Create();

    obs2->N = obs->N;
    obs2->T = 11;
    sg_obs_Allocate(obs2);

    for (size_t i = 0; i < 10; i++) {
        /* code */
        obs2->seq[i] = obs->seq[i + obs->T - 10];
    }

    int best_observation1 = -1;
    // int best_observation2   = -1;
    double best_model_score = 0;

    for (size_t i = 0; i < mod->M; i++) {
        /* code */
        obs2->seq[(obs2->T - 1)] = i;

        _calculateAlphaPass(mod, obs2);
        double score = 0;
        for (size_t j = 0; j < obs2->N; j++) {
            /* code */
            // printf("alpha[%lu][%zu]: %lf\n", obs2->T - 1, j, obs2->alpha[(obs2->T - 1)][j]);
            score += obs2->alpha[(obs2->T - 1)][j];
        }
        printf("Observation: %zu, Score: %lf\n", i, score);
        if (score > best_model_score) {
            best_model_score  = score;
            best_observation1 = i;
            // best_observation2 = k;
        }
    }
    // printf("Best Prediction is %d %d with score %lf\n", best_observation1, best_observation2, best_model_score);
    printf("Best Prediction is %d with score %lf\n", best_observation1, best_model_score);
    return best_observation1;
} /* sg_hmm_predict */
