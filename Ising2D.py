import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import Ising2Dlib

def equilibrium_properties(ising, T_list, 
                            overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite):
    ising.generate_distances_dict()
    print(ising.distances)
    print("Total number of distances: %d" % ising.size)

    print("Calculate the equilibrium properties of 2D Ising model in a %d*%d lattice. ==================================================================\n"
            % (ising.L, ising.L))
    overall_equil_steps = int(overall_equil_steps_persite * ising.N)
    equil_steps = int(equil_steps_persite * ising.N)
    measure_steps = int(measure_steps_persite * ising.N)
    measure_interval = int(measure_interval_persite * ising.N)

    ising.init_conf(0)
    ising.T = T_list[0]
    ising.Metropolis(overall_equil_steps, 0, measure_interval)
    E_perspin_T, M_abs_perspin_T, c_T, chi_T = [], [], [], []
    confs_T = np.empty((T_list.size, ising.conf_num, ising.L, ising.L))
    correlation_T = np.empty((T_list.size, ising.size + 1))
    idx = 0
    for T in T_list:
        print("T = %.2f" % T)
        ising.T = T
        beta = 1.0 / ising.T
        M_samples_T, E_samples_T, confs = ising.Metropolis(equil_steps, measure_steps, measure_interval) 

        M_abs_samples_T = np.abs(M_samples_T)
        m_T = np.mean(M_abs_samples_T) / ising.N
        print("T = %.2f, m = %.2f" % (T, m_T))
        M_abs_perspin_T.append(m_T)
        E_perspin_T.append( np.mean(E_samples_T) / ising.N )
        c_T.append( beta ** 2 * np.var(E_samples_T) / ising.N )
        chi_T.append( beta * np.var(M_abs_samples_T) / ising.N )

        confs_T[idx, :, :, :] = confs
        ising.generate_disconnected_correlation()
        correlation_T[idx, :] = np.concatenate( (np.array([1]), ising.disconnected_correlation) ) - m_T ** 2
        idx += 1

    folder = "thermo_properties/L_%d/" % ising.L
    filename = folder + "equilibrium_properties.npz"
    np.savez(filename, E_perspin_T=np.array(E_perspin_T), M_abs_perspin_T=np.array(M_abs_perspin_T), 
                       c_T=np.array(c_T), chi_T=np.array(chi_T), 
                       T_list=T_list, confs_T=confs_T, 
                       distances = np.concatenate( (np.array([0.0]), ising.distances) ), correlation_T=correlation_T)
    return

def plot_equilibrium_properties(ising, T_list, 
                            overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, 
                            direct_result=None):
    if(direct_result):
        filename = direct_result
        if(os.path.isfile(filename)):
            print("Read results from file %s ..." % filename)
            results = np.load(filename)
            T_list_direct = results["T_list"]
            E_perspin_T_direct = results["E_perspin_T"]
            M_abs_perspin_T_direct = results["M_abs_perspin_T"]
            c_T_direct = results["c_T"]
            chi_T_direct = results["chi_T"]
        else:
            print("Error reading file %s: the file doesn't exist." % filename)
            direct_result = None
    folder = "thermo_properties/L_%d/" % ising.L
    filename = folder + "equilibrium_properties.npz"
    if(not os.path.isfile(filename)):
        print("Error reading file %s: the file doesn't exist.\nCalculating the equilibrium properties from scratch..." % filename)
        equilibrium_properties(ising, T_list, 
                            overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite)
        print("Done...")
    results = np.load(filename)
    T_list= results["T_list"]
    E_perspin_T= results["E_perspin_T"]
    M_abs_perspin_T= results["M_abs_perspin_T"]
    c_T= results["c_T"]
    chi_T= results["chi_T"]
    confs_T = results["confs_T"]
    distances = results["distances"]
    correlation_T = results["correlation_T"]

    fig_size = (9, 7)
    label_size = "xx-large"
    title_size = "x-large"
    title_weight = "bold"
    line_marker = "o-"

    plt.figure(figsize=fig_size)
    plt.plot(T_list, E_perspin_T, line_marker, label="Metropolis MC algorithm")
    if(direct_result):
        plt.plot(T_list_direct, E_perspin_T_direct, label="Direct summation of partition function")
    plt.xlabel("$T$", size=label_size)
    plt.ylabel("$\\frac{<E>}{N}$", size=label_size)
    plt.title("Average energy per spin of 2D Ising model as function of temperature $T$\n"
                + "Square lattice: $%d \\times %d$" % (ising.L, ising.L), size=title_size, weight=title_weight)
    plt.legend()
    plt.savefig(folder + "Energy.eps")
    plt.close()

    plt.figure(figsize=fig_size)
    plt.plot(T_list, M_abs_perspin_T, line_marker, label="Metropolis MC algorithm")
    if(direct_result):
        plt.plot(T_list_direct, M_abs_perspin_T_direct, label="Direct summation of partition function")
    plt.xlabel("$T$", size=label_size)
    plt.ylabel("$\\frac{<|M|>}{N}$", size=label_size)
    plt.ylim(0.0, 1.05)
    plt.title("Average (absolute value of) magnetization per spin of 2D Ising model\nas function of temperature $T$\n"
                + "Square lattice: $%d \\times %d$" % (ising.L, ising.L), size=title_size, weight=title_weight)
    plt.legend()
    plt.savefig(folder + "Magnetization.eps")
    plt.close()

    plt.figure(figsize=fig_size)
    plt.plot(T_list, c_T, line_marker, label="Metropolis MC algorithm")
    if(direct_result):
        plt.plot(T_list_direct, c_T_direct, label="Direct summation of partition function")
    plt.xlabel("$T$", size=label_size)
    plt.ylabel("$c$", size=label_size)
    plt.title("Specific heat per spin of 2D Ising model as function of temperature $T$\n"
                + "Square lattice: $%d \\times %d$" % (ising.L, ising.L), size=title_size, weight=title_weight)
    plt.legend()
    plt.savefig(folder + "Specific_heat.eps")
    plt.close()

    plt.figure(figsize=fig_size)
    plt.plot(T_list, chi_T, line_marker, label="Metropolis MC algorithm")
    if(direct_result):
        plt.plot(T_list_direct, chi_T_direct, label="Direct summation of partition function")
    plt.xlabel("$T$", size=label_size)
    plt.ylabel("$\chi$", size=label_size)
    plt.title("Magnetic susceptibility per spin of 2D Ising model\nas function of temperature $T$\n"
                + "Square lattice: $%d \\times %d$" % (ising.L, ising.L), size=title_size, weight=title_weight)
    plt.legend()
    plt.savefig(folder + "Magnetic_susceptibility.eps")
    plt.close()

    idx = 0
    print(distances[:10])
    for T in T_list:
        print("T = %.2f: %s" % (T, correlation_T[idx, :10]))
        plt.figure(figsize=fig_size)
        plt.plot(distances, correlation_T[idx, :])
        plt.xlabel("$r$", size=label_size)
        plt.ylabel("$C(r)$", size=label_size)
        plt.title("Connected spin-spin correlation function of 2D Ising model at temperature $T$\n"
                    + "Square lattice: $%d \\times %d$   $T$ = %.2f" % (ising.L, ising.L, T), size=title_size, weight=title_weight)
        plt.ylim(-0.05, 1.0)
        plt.savefig(folder + "correlation_function/T_%.2f.png" % T)
        plt.close()

        confs = confs_T[idx, :, :, :]
        for i in range(confs.shape[0]):
            plt.figure(figsize=fig_size)
            plt.imshow(confs[i, :, :], extent=[0, ising.L, 0, ising.L], cmap="hot", interpolation='nearest')
            plt.title("Typical equilibrium configuration of 2D Ising model at temperature $T$\n"
                    + "Square lattice: $%d \\times %d$   $T$ = %.2f" % (ising.L, ising.L, T), size=title_size, weight=title_weight)
            plt.savefig(folder + "snapshots/T_%.2f_%02d.png" % (T, i))
            plt.close()
        idx += 1

def equilibration_time(ising, T_list, measure_steps_persite, measure_interval_persite):
    print("Investigate the equilibration time of Metropolis simulation of 2D Ising model in a %d*%d lattice. ===========================================\n"
            % (ising.L, ising.L))
    fig_size = (9, 7)
    label_size = "xx-large"
    title_size = "x-large"
    title_weight = "bold"
    folder = "equilibration_time/L_%d/" % ising.L
#    folder = "equilibration_time/L_%d_test/" % ising.L

    equil_steps = 0
    measure_interval = int(measure_interval_persite * ising.N)

    for T in T_list:
        print("T = %.2f" % T)
#        if(T <= 2.5):
        measure_steps = int(measure_steps_persite[1] * ising.N)
#        else:
#            measure_steps = int(measure_steps_persite[0] * ising.N)
        num_samples = measure_steps / measure_interval

        ising.T = T
        seed = np.random.randint(1, 10000)
        ising.init_conf(1)
        M_samples1_T, E_samples1_T = ising.Metropolis(equil_steps, measure_steps, measure_interval, seed=seed)
        M_samples1_T, E_samples1_T = np.abs(M_samples1_T) / ising.N, E_samples1_T / ising.N
        ising.init_conf(1)
        M_samples2_T, E_samples2_T = ising.Metropolis(equil_steps, measure_steps, measure_interval, seed=seed)
        M_samples2_T, E_samples2_T = np.abs(M_samples2_T) / ising.N, E_samples2_T / ising.N
        step_axis = np.arange(num_samples) * measure_interval_persite 

        plt.figure(figsize=fig_size)
        plt.plot(step_axis, E_samples1_T, label="Simulation 1")
        plt.plot(step_axis, E_samples2_T, label="Simulation 2")
        plt.xlabel("Monte Carlo steps per site", size=label_size)
        plt.ylabel("$\\frac{E}{N}$", size=label_size)
        plt.title("Energy per spin of 2D Ising model as function of Monte Carlo step\nin two simulations with different initial configurations\n"
                + "Square lattice: $%d \\times %d$   $T$ = %.2f" % (ising.L, ising.L, ising.T), size=title_size, weight=title_weight)
        lower_bound = -2.0
        plt.ylim(lower_bound - 0.1, 0.2)
        plt.legend()
        plt.savefig(folder + "Energy_T_%.2f.png" % ising.T)
        plt.close()

        plt.figure(figsize=fig_size)
        plt.plot(step_axis, M_samples1_T, label="Simulation 1")
        plt.plot(step_axis, M_samples2_T, label="Simulation 2")
        plt.xlabel("Monte Carlo steps per site", size=label_size)
        plt.ylabel("$\\frac{|M|}{N}$", size=label_size)
        plt.title("Magnetization per spin of 2D Ising model as function of Monte Carlo step\nin two simulations with different initial configurations\n"
                + "Square lattice: $%d \\times %d$   $T$ = %.2f" % (ising.L, ising.L, ising.T), size=title_size, weight=title_weight)
        plt.ylim(0.0, 1.1)
        plt.legend()
        plt.savefig(folder + "Magnetization_T_%.2f.png" % ising.T)
        plt.close()

def time_displaced_autocorrelation(ising, T_list, 
                            overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, 
                            tmax_persite):
    overall_equil_steps = int(overall_equil_steps_persite * ising.N)
    equil_steps = int(equil_steps_persite * ising.N)
    measure_steps = int(measure_steps_persite * ising.N)
    measure_interval = int(measure_interval_persite * ising.N)
    num_samples = int(measure_steps / measure_interval)

    tmax = int(tmax_persite / measure_interval_persite)
    autocorrelation_T = np.empty(tmax + 1)
    t_axis = np.arange(tmax + 1) * measure_interval_persite
    autocorrelations = np.empty((T_list.size, tmax + 1))

    folder = "timedisplaced_autocorrelation/L_%d/" % ising.L
    filename = folder + "autocorrelations.npz"

    ising.init_conf(0)
    ising.Metropolis(overall_equil_steps, 0, measure_interval)
    idx = 0
    for T in T_list:
        print("T = %.2f" % T)
        ising.T = T
        M_samples_T, E_samples_T = ising.Metropolis(equil_steps, measure_steps, measure_interval) 
        M_samples_T = np.abs(M_samples_T) / ising.N

        for t in range(tmax + 1):
            m = M_samples_T[:(num_samples - t)]
            m_t = M_samples_T[t:]
            autocorrelation_T[t] = np.mean(m * m_t) - np.mean(m) * np.mean(m_t)
        if(autocorrelation_T[0] != 0.0):
            autocorrelation_T = autocorrelation_T / autocorrelation_T[0]
        autocorrelations[idx, :] = autocorrelation_T
        idx += 1
    np.savez(filename, T_list=T_list, t_axis=t_axis, autocorrelations=autocorrelations)
        
def plot_time_displaced_autocorrelation(ising, T_list, 
                            overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, 
                            tmax_persite):
    print("Calculate the time-displaced magnetization autocorrelation function of Metropolis simulation of 2D Ising model in a %d*%d lattice. ================\n"
            % (ising.L, ising.L))
    folder = "timedisplaced_autocorrelation/L_%d/" % ising.L
    filename = folder + "autocorrelations.npz"
    movie_filename = folder + "movie.gif"
    if(not os.path.isfile(filename)):
        print("Error reading the file %s: the file doesn't exist!\nCalculate the autocorrelation from scratch..." % filename)
        time_displaced_autocorrelation(ising, T_list, overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, tmax_persite)
        print("Done......")
    print("Read autocorrelation data from file %s ..." % filename)
    results = np.load(filename)
    T_list = results["T_list"]
    t_axis = results["t_axis"]
    autocorrelations = results["autocorrelations"]

    fig_size = (9, 7)
    label_size = "xx-large"
    title_size = "x-large"
    title_weight = "bold"
    idx = 0
    for T in T_list:
        plt.figure(figsize=fig_size)
        plt.plot(t_axis, autocorrelations[idx, :])
        plt.xlabel("time t (Monte Carlo steps per site)", size=label_size)
        plt.ylabel("$\\frac{\chi(t)}{\chi(0)}$", size=label_size)
        plt.title("Time-displaced magnetization autocorrelation function\nof a Metropolis simulation of 2D Ising model\n"
                + "Square lattice: $%d \\times %d$   $T$ = %.2f" % (ising.L, ising.L, T), size=title_size, weight=title_weight)
        plt.ylim(-0.2, 1.1)
        plt.savefig(folder + "T_%.2f.png" % T)
        plt.close()
        idx += 1
    os.system("convert -delay 1 -dispose Background +page " + folder + "*.png -loop 0 " + movie_filename)

L = 100
ising = Ising2Dlib.Ising_2D(L)

"""
T_list = np.concatenate( (np.linspace(0.1, 2.0, num=9, endpoint=False),
                          np.linspace(2.0, 3.0, num=50, endpoint=False), 
                          np.linspace(3.0, 5.0, num=11)) )
overall_equil_steps_persite = 10000
equil_steps_persite = 2000
measure_steps_persite = 20000
measure_interval_persite = 0.1
tmax_persite = 800
plot_time_displaced_autocorrelation(ising, T_list, overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, tmax_persite)
sys.exit(123)

T_list = np.concatenate( (np.linspace(0.1, 2.0, num=9, endpoint=False),
                          np.linspace(2.0, 2.5, num=25, endpoint=False), 
                          np.linspace(2.5, 5.0, num=13)) )
measure_steps_persite = (2000, 20000)
measure_interval_persite = 1
equilibration_time(ising, T_list, measure_steps_persite, measure_interval_persite)
"""

T_list = np.concatenate( (np.linspace(0.1, 2.0, num=19, endpoint=False),
                          np.linspace(2.0, 2.5, num=50, endpoint=False), 
                          np.linspace(2.5, 5.0, num=26)) )
overall_equil_steps_persite = 20000
equil_steps_persite = 2000
measure_steps_persite = 20000
measure_interval_persite = 1
plot_equilibrium_properties(ising, T_list, 
                       overall_equil_steps_persite, equil_steps_persite, measure_steps_persite, measure_interval_persite, 
                       direct_result="results_direct_L_%d.npz" % ising.L)
