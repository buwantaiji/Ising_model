#include <gsl/gsl_rng.h>
#include <map>

using namespace std;

class Ising2D {
    private:
        const gsl_rng_type *Type;
        gsl_rng *r;
    public:
        int L, N;
        double T;
        int **S;
        const int conf_num = 20;
        int ***Ss;

        Ising2D(const int size);
        ~Ising2D();
        void init_conf(const int choice);
        void print_conf();

        int site_field(const int row, const int col);
        int *M_samples, *E_samples;
        void Metropolis(const int equil_steps, const int measure_steps, const int measure_interval, const unsigned long seed);

        double distance(const int row1, const int col1, const int row2, const int col2);
        map<double, int> distances_dict;
        double *distances;
        int *distances_pairnum;
        int size;
        void generate_distances_dict();
        double *disconnected_correlation;
        void generate_disconnected_correlation();
};

