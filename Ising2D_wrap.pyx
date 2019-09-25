import numpy as np
cimport numpy as np

cdef extern from "Ising2D.h":
    cdef cppclass Ising2D:
        int L, N
        double T
        int **S
        const int conf_num;
        int ***Ss;

        Ising2D(const int size)
        void init_conf(const int choice)
        void print_conf()

        int site_field(const int row, const int col)
        int *M_samples, *E_samples
        void Metropolis(const int equil_steps, const int measure_steps, const int measure_interval, const unsigned long seed)

        double *distances
        int size
        void generate_distances_dict()
        double *disconnected_correlation
        void generate_disconnected_correlation()

cdef class Ising_2D:
    cdef Ising2D *ising
    def __cinit__(self, const int size):
        self.ising = new Ising2D(size)
    def __dealloc__(self):
        del self.ising

    def init_conf(self, const int choice):
        self.ising.init_conf(choice)
    def print_conf(self):
        self.ising.print_conf()

    def Metropolis(self, const int equil_steps, const int measure_steps, const int measure_interval, const unsigned long seed=0):
        cdef int L = self.ising.L
        self.ising.Metropolis(equil_steps, measure_steps, measure_interval, seed)
        cdef int num_samples = measure_steps / measure_interval
        if(num_samples == 0):
            return
        cdef np.ndarray[np.int32_t, ndim=1] M_samples = np.empty(num_samples, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] E_samples = np.empty(num_samples, dtype=np.int32)
        cdef int i
        for i in range(num_samples):
            M_samples[i] = self.ising.M_samples[i]
            E_samples[i] = self.ising.E_samples[i]

        cdef int conf_num = self.ising.conf_num
        cdef np.ndarray[np.int32_t, ndim=3] confs = np.empty((conf_num, L, L), dtype=np.int32)
        cdef int ***Ss = self.ising.Ss
        cdef int row, col
        for i in range(conf_num):
            for row in range(L):
                for col in range(L):
                    confs[i, row, col] = Ss[i][row][col]
        return M_samples, E_samples, confs

    @property
    def L(self): return self.ising.L
    @property
    def N(self): return self.ising.N
    @property
    def T(self): return self.ising.T
    @T.setter
    def T(self, value): self.ising.T = value

    @property
    def S(self):
        cdef int L = self.ising.L
        cdef int **S = self.ising.S
        cdef np.ndarray[np.int32_t, ndim=2] conf = np.empty((L, L), dtype=np.int32)
        cdef int row, col
        for row in range(L):
            for col in range(L):
                conf[row, col] = S[row][col]
        return conf
    @property
    def conf_num(self): return self.ising.conf_num

    @property
    def size(self): return self.ising.size
    def generate_distances_dict(self):
        self.ising.generate_distances_dict()
    @property
    def distances(self):
        cdef int size = self.ising.size
        cdef np.ndarray[np.float32_t, ndim=1] distances = np.empty(size, dtype=np.float32)
        cdef int i
        for i in range(size):
            distances[i] = self.ising.distances[i]
        return distances
    def generate_disconnected_correlation(self):
        self.ising.generate_disconnected_correlation()
    @property
    def disconnected_correlation(self):
        cdef int size = self.ising.size
        cdef np.ndarray[np.float32_t, ndim=1] disconnected_correlation = np.empty(size, dtype=np.float32)
        cdef int i
        for i in range(size):
            disconnected_correlation[i] = self.ising.disconnected_correlation[i]
        return disconnected_correlation


"""
    cdef dict __dict__
    def generate_distances(self):
        cdef int L = self.ising.L
        cdef int N = self.ising.N
        cdef double dist
        cdef int s1, row1, col1, s2, row2, col2
        distances_dict = {}
        for s1 in range(N):
            if(s1 % 1000 == 0):
                print(s1)
            row1, col1 = s1 // L, s1 % L
            for s2 in range(s1 + 1, N):
                row2, col2 = s2 // L, s2 % L
                dist = distance(L, row1, col1, row2, col2)
                if(dist in distances_dict):
                    distances_dict[dist] += 1
                else:
                    distances_dict[dist] = 1
        self.distances_dict = distances_dict
        self.distances = list(distances_dict.keys())
        self.distances.sort()
        self.distances = np.array(self.distances)
        self.distances_pairnum = np.array([distances_dict[d] for d in self.distances])
        return

cdef double distance(int L, int row1, int col1, int row2, int col2):
    cdef double dist
    cdef int delta_row = abs(row1 - row2)
    delta_row = min(delta_row, L - delta_row)
    cdef int delta_col = abs(col1 - col2)
    delta_col = min(delta_col, L - delta_col)
    dist = np.sqrt(delta_row ** 2 + delta_col ** 2)
    return dist
def correlation_function(Ising_2D ising, np.ndarray[np.int32_t, ndim=3] confs, double m):
    cdef int L = ising.L
    cdef int N = ising.N
    cdef int num_samples = confs.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] conf = np.empty((L, L), dtype=np.int32)
    cdef int s1, row1, col1, s2, row2, col2
    cdef double dist
    correlation = {}
    for i in range(num_samples):
        conf = confs[i, :, :]
        for s1 in range(N):
            if(s1 % 1000 == 0):
                print(s1)
            row1, col1 = s1 // L, s1 % L
            for s2 in range(s1 + 1, N):
                row2, col2 = s2 // L, s2 % L
                dist = distance(L, row1, col1, row2, col2)
                if(dist in correlation):
                    correlation[dist] = correlation[dist] + conf[row1, col1] * conf[row2, col2] - m ** 2
                else:
                    correlation[dist] = conf[row1, col1] * conf[row2, col2] - m ** 2
    correlation = np.array([correlation[d] for d in ising.distances])
    correlation = correlation / (num_samples * ising.distances_pairnum)
    return correlation
"""
