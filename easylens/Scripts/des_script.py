__author__ = 'sibirrer'

#this file is ment to be a shell script to be run with Monch cluster

# set up the scene
from cosmoHammer.util.MpiUtil import MpiPool
import time
import sys
import pickle
import dill

start_time = time.time()

#path2load = '/mnt/lnec/sibirrer/input.txt'

path2load = str(sys.argv[1])
f = open(path2load, 'rb')

[lensDES, walkerRatio, n_burn, n_run, mean_start, sigma_start, lowerLimit, upperLimit, path2dump] = dill.load(f)
f.close()

end_time = time.time()
#print end_time - start_time, 'time used for initialisation'
# run the computation

from lensDES.Fitting.mcmc import MCMC_sampler
sampler = MCMC_sampler(lensDES, fix_center=False)
samples = sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, lowerLimit, upperLimit, threadCount=1, init_pos=None, mpi_monch=True)
# save the output
pool = MpiPool(None)
if pool.isMaster():
    f = open(path2dump, 'wb')
    pickle.dump(samples, f)
    f.close()
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('Result saved in:', path2dump)
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')


