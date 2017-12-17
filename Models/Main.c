#include "includes.h"

void
obtain_observations(sg_obs_t * obs, long * observations, char * file_name, int m);
long
format(char * line, int m);

int
main(int argc, char * argv[])
{
    char * file_name, * pEnd;
    int m, n;

    if (argc < 4) {
        fprintf(stderr, "usage: %s filename observations states\n", argv[0]);
        exit(1);
    }
    file_name = argv[1];
    m         = strtol(argv[2], &pEnd, 10);
    n         = strtol(argv[3], &pEnd, 10);

    /* Obtain observation sequence */
    long * observations = (long *) malloc(1655 * sizeof(long));

    sg_obs_t * obs = sg_obs_Create();
    obs->N = n;
    obs->T = 1648;
    sg_obs_Allocate(obs);
    obtain_observations(obs, observations, file_name, m);

    /* Create random model */
    sg_hmm_t * mod = sg_hmm_Create();
    mod->N = n;
    mod->M = m + 2;
    sg_hmm_InitRandom(mod);

    sg_train(mod, obs, 200);

    sg_hmm_Print(mod);

    long prediction = sg_hmm_predict(mod, obs);

    printf("Actual Value: %lu\n", observations[obs->T]);
    // printf("Predicted Value: %lu\n", prediction);

    /* clean up */
    exit(EXIT_SUCCESS);
} /* main */

void
obtain_observations(sg_obs_t * obs, long * observations, char * file_name, int m)
{
    /* Initialize */
    FILE * fp;
    char * line = NULL;
    size_t len  = 0;
    ssize_t read;
    int i = 0;

    /* Validate */
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        fprintf(stderr, "Unable to open file\n");
        exit(EXIT_FAILURE);
    }

    /* Ignore file header */
    getline(&line, &len, fp);

    /* Obtain observations */
    while ((read = getline(&line, &len, fp)) != -1) {
        // obs->seq[i++] = format(line, m);
        long value = format(line, m);
        if (i < obs->T) {
            obs->seq[i] = value;
        }
        observations[i++] = value;
    }

    /* Clean up */
    fclose(fp);
    if (line) {
        free(line);
    }
} /* obtain_observations */

long
format(char * line, int m)
{
    char * pEnd;
    double percent = strtod(line, &pEnd);

    if (percent < -20)
        return 0;

    double window = 60 / m;
    int i         = 1;

    while (percent > (-20 + window)) {
        i++;
        percent -= window;
    }

    return i;
}
