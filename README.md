# Polymer_analysis
python code for analyzing polymer simulation trajectories


# Project plan

1) Reading DCD or XTC trajectories via MDAnalysis
2) Assign molecule info
3) Calculating radial distirbution functions or MSD via MDanalyiss or MPI


# Memo
Multiprocessing vs MPI4py
the Multiprocessing module does not support parallelization over multiple compute nodes.
MPI4pi does.

- Multiprocessing via python
  example from ygyoon 22. 04.08

    from functools import partial
    import multiprocessing
    
    list = []
    
    n_cpu = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(processes=n_cpu)
    m = multiprocessing.Manager()
    l = m.Lock() # Lock list variables to prevent list mixing ?
    func = partial(funcc, l)
    pool.map(func, list)
    pool.close()
    pool.join()
