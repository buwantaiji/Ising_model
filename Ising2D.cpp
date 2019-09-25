#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <math.h>
#include <map>
#include "Ising2D.h"

using namespace std;

Ising2D::Ising2D(const int size) {
    L = size;
    printf("Initialize 2D Ising model of %d*%d lattice...\n", L, L);
    N = L * L;
    S = (int **)malloc(L * sizeof(int *));
    for(int row = 0; row < L; row++)
        S[row] = (int *)malloc(L * sizeof(int));
    Ss = (int ***)malloc(conf_num * sizeof(int **));
    for(int i = 0; i < conf_num; i++) {
        Ss[i] = (int **)malloc(L * sizeof(int *));
        for(int row = 0; row < L; row++)
            Ss[i][row] = (int *)malloc(L * sizeof(int));
    }

    gsl_rng_env_setup();
    Type = gsl_rng_default;
    r = gsl_rng_alloc(Type);
    gsl_rng_set(r, (unsigned long)time(NULL));
    printf("Initialize random number generator %s...\n", gsl_rng_name(r));

    M_samples = NULL;
    E_samples = NULL;

    distances = NULL;
    distances_pairnum = NULL;
    disconnected_correlation = NULL;
}
Ising2D::~Ising2D() {
    for(int row = 0; row < L; row++)
        free(S[row]);
    free(S);
    for(int i = 0; i < conf_num; i++) {
        for(int row = 0; row < L; row++)
            free(Ss[i][row]);
        free(Ss[i]);
    }
    free(Ss);

    gsl_rng_free(r);

    free(M_samples);
    free(E_samples);
    free(distances);
    free(distances_pairnum);
    free(disconnected_correlation);
    printf("The destructor is called. Bye!\n");
}
void Ising2D::init_conf(const int choice) {
    int ud[] = {1, -1};
    if(choice == 0) {
        printf("Initialize the spin configuration of T = 0...\n");
        int orientation;
        gsl_ran_sample(r, &orientation, 1, ud, 2, sizeof(int));
        for(int row = 0; row < L; row++)
            for(int col = 0; col < L; col++)
                S[row][col] = orientation;
    }
    else if(choice == 1) {
        printf("Initialize the spin configuration of T = inf...\n");
        for(int row = 0; row < L; row++)
            gsl_ran_sample(r, S[row], L, ud, 2, sizeof(int));
    }
    return;
}
void Ising2D::print_conf() {
    printf("configuration: \n");
    for(int row = 0; row < L; row++) {
        for(int col = 0; col < L; col++)
            printf("%2d ", S[row][col]);
        printf("\n");
    }
    return;
}

int Ising2D::site_field(const int row, const int col) {
    int up   = (row + L - 1) % L, down  = (row + 1) % L;
    int left = (col + L - 1) % L, right = (col + 1) % L;
    int h = S[row][left] + S[row][right] + S[up][col] + S[down][col];
    return h;
}

void Ising2D::Metropolis(const int equil_steps, const int measure_steps, const int measure_interval, const unsigned long seed) {
    printf("Using seed = %lu\n", seed);
    if(seed != 0)
        gsl_rng_set(r, seed);
    double beta = 1.0 / T;
    int total_steps = equil_steps + measure_steps;
    int num_samples = measure_steps / measure_interval;
    int conf_measure_interval = measure_steps / conf_num;
    printf("equil_steps = %d, measure_steps = %d, measure_interval = %d, conf_measure_interval = %d\n", 
            equil_steps, measure_steps, measure_interval, conf_measure_interval);
    printf("num_samples = %d, conf_num = %d\n", num_samples, conf_num);
    int sample = 0, conf_sample = 0;
    free(M_samples);
    free(E_samples);
    M_samples = (int *)malloc(num_samples * sizeof(int));
    E_samples = (int *)malloc(num_samples * sizeof(int));
    int magnetization = 0, energy = 0;
    for(int row = 0; row < L; row++)
        for(int col = 0; col < L; col++) {
            magnetization += S[row][col];
            energy -= S[row][col] * site_field(row, col);
        }
    energy /= 2;
    int site, row, col;
    int deltaM, deltaE;
    for(int step = 0; step < total_steps; step++) {
        site = gsl_rng_uniform_int(r, N);
        row = site / L;
        col = site % L;
        deltaM = - 2 * S[row][col];
        deltaE = 2 * S[row][col] * site_field(row, col);
        if( gsl_rng_uniform(r) < exp(- beta * deltaE) ) {
            S[row][col] *= -1;
            magnetization += deltaM;
            energy += deltaE;
        }
        if(step >= equil_steps) {
            int measure_step = step - equil_steps;
            if(measure_step % measure_interval == 0) {
                M_samples[sample] = magnetization;
                E_samples[sample] = energy;
                sample++;
            }
            if(measure_step % conf_measure_interval == 0) {
                for(int row = 0; row < L; row++)
                    for(int col = 0; col < L; col++)
                        Ss[conf_sample][row][col] = S[row][col];
                conf_sample++;
            }
        }
    }
    printf("iteration done. Total number of samples: %d. Total number of conf samples: %d\n", sample, conf_sample);
    if(seed != 0)
        gsl_rng_set(r, (unsigned long)time(NULL));
    return;
}

double Ising2D::distance(const int row1, const int col1, const int row2, const int col2) {
    double dist;
    int delta_row = abs(row1 - row2);
    delta_row = min(delta_row, L - delta_row);
    int delta_col = abs(col1 - col2);
    delta_col = min(delta_col, L - delta_col);
    dist = sqrt(delta_row * delta_row + delta_col * delta_col);
    return dist;
}

void Ising2D::generate_distances_dict() {
    int s1, row1, col1, s2, row2, col2;
    double dist;
    for(s1 = 0; s1 < N; s1++) {
        row1 = s1 / L;
        col1 = s1 % L;
        for(s2 = s1 + 1; s2 < N; s2++) {
            row2 = s2 / L; 
            col2 = s2 % L;
            dist = distance(row1, col1, row2, col2);
            if(distances_dict.find(dist) != distances_dict.end())
                distances_dict[dist]++;
            else
                distances_dict[dist] = 1;
        }
    }
    size = distances_dict.size();
    distances = (double *)malloc(size * sizeof(double));
    distances_pairnum = (int *)malloc(size * sizeof(int));
    map<double, int>::iterator iter;
    int idx = 0;
    for(iter = distances_dict.begin(); iter != distances_dict.end(); iter++) {
        distances[idx] = iter->first;
        distances_pairnum[idx] = iter->second;
        idx++;
    }
    return;
}

void Ising2D::generate_disconnected_correlation() {
    map<double, int> disconnected_correlation_dict;
    map<double, int>::iterator iter;
    for(iter = distances_dict.begin(); iter != distances_dict.end(); iter++)
        disconnected_correlation_dict[iter->first] = 0;
    int s1, row1, col1, s2, row2, col2;
    double dist;
    for(s1 = 0; s1 < N; s1++) {
        row1 = s1 / L;
        col1 = s1 % L;
        for(s2 = s1 + 1; s2 < N; s2++) {
            row2 = s2 / L; 
            col2 = s2 % L;
            dist = distance(row1, col1, row2, col2);
            for(int i = 0; i < conf_num; i++)
                disconnected_correlation_dict[dist] += Ss[i][row1][col1] * Ss[i][row2][col2];
        }
    }
    if(disconnected_correlation == NULL)
        disconnected_correlation = (double *)malloc(size * sizeof(double));
    int idx = 0;
    for(iter = disconnected_correlation_dict.begin(); iter != disconnected_correlation_dict.end(); iter++) {
        disconnected_correlation[idx] = (double)(iter->second) / (distances_pairnum[idx] * conf_num);
        idx++;
    }
    return;
}

/*
int main(int argc, char *argv[]) {
    int L = 100;
    int N = L * L;
    Ising2D ising(L);
    ising.init_conf(1);
    ising.print_conf();
    ising.T = 3.0;

    int equil_steps_persite = 2000;
    int equil_steps = equil_steps_persite * N;
    int measure_steps_persite = 20000;
    int measure_steps = measure_steps_persite * N;
    int measure_interval_persite = 1;
    int measure_interval = int(measure_interval_persite * N);
    ising.Metropolis(equil_steps, measure_steps, measure_interval);
    ising.generate_distances_dict();
    printf("Size of generate_distances_dict: %d\n", ising.distances_dict.size());
    map<double, int>::iterator iter;
    int distance_num = 0;
    for(iter = ising.distances_dict.begin(); iter != ising.distances_dict.end(); iter++) {
        printf("%f:   %d\n", iter->first, iter->second);
        distance_num += iter->second;
    }
    printf("distance_num = %d\n", distance_num);
    return 0;
}
*/
