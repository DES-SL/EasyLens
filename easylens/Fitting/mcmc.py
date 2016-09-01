__author__ = 'sibirrer'


from cosmoHammer import ParticleSwarmOptimizer
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import MpiUtil
from cosmoHammer.util import InMemoryStorageUtil

import emcee
import numpy as np
import time
import tempfile
import os
import shutil


class MCMC_chain(object):
    """
    this class contains the routines to run a MCMC process
    """

    def __init__(self, lensDES, fix_center=True):
        """
        initializes all the classes needed for the chain
        """
        self.lensDES = lensDES
        self.data_vector = self.lensDES.get_data_vector()
        self.C_D_inv_vector = self.lensDES.get_C_D_inv_vector()
        self.lens_type = lensDES.lens_type
        self.fix_center = fix_center

    def X2_chain(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        kwargs_lens = self.lensDES.get_lens()

        #extract parameters
        if self.lens_type == 'SIS':
            if self.fix_center is True:
                [phi_E] = args
            else:
                [phi_E, center_x, center_y] = args
                kwargs_lens["center_x_sis"] = center_x
                kwargs_lens["center_y_sis"] = center_y
            # update lensDES
            kwargs_lens["phi_E_sis"] = phi_E
        elif self.lens_type == 'SPEMD':
            if self.fix_center is True:
                [phi_E, gamma, q, phi_G] = args
            else:
                [phi_E, gamma, q, phi_G, center_x, center_y] = args
                kwargs_lens['center_x'] = center_x
                kwargs_lens['center_y'] = center_y
            kwargs_lens['phi_E'] = phi_E
            kwargs_lens['gamma'] = gamma
            kwargs_lens['q'] = q
            kwargs_lens['phi_G'] =phi_G

        else:
            raise ValueError("Lens type %s is not valid!" % (self.lens_type))
        self.lensDES.add_lens(kwargs_lens, print_statement=False)
        # generate image
        A = self.lensDES.get_response()
        param_array, model_array = self.lensDES.get_inverted(A, self.C_D_inv_vector, self.data_vector)

        # compute X^2
        chi2 = np.sum((model_array-self.data_vector)**2*self.C_D_inv_vector)
        logL = - chi2/2
        return logL, None

    def __call__(self, a):
        return self.X2_chain(a)

    def likelihood(self, a):
        return self.X2_chain(a)

    def computeLikelihood(self, ctx):
        likelihood, _ = self.X2_chain(ctx.getParams())
        return likelihood

    def setup(self):
        pass


class MCMC_sampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, easyLens, fix_center=True):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = MCMC_chain(easyLens, fix_center)

    def pso(self, n_particles, n_iterations, lowerLimit, upperLimit, threadCount=1, init_pos=None, mpi_monch=False):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
        if mpi_monch is True:
            pso = MpiParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=1)
        else:
            pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=threadCount)
        if not init_pos is None:
            pso.gbest.position = init_pos
            pso.gbest.velocity = [0]*len(init_pos)
            pso.gbest.fitness, _ = self.chain.likelihood(init_pos)
        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()
        if pso.isMaster():
            pass
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness*2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
        if mpi_monch is True:
            result = MpiUtil.mpiBCast(pso.gbest.position)
        else:
            result = pso.gbest.position
        if pso.isMaster() and mpi_monch is True:
            print(pso.gbest.fitness*2/(self.chain.lensDES.get_pixels_unmasked()), 'reduced X^2 of best position')
            print(result, 'result')
            time_end = time.time()
            print(time_end - time_start, 'time used for PSO')
        return result, [X2_list, pos_list, vel_list]

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        numParam = len(mean_start)
        sampler = emcee.EnsembleSampler(n_walkers, numParam, self.chain.X2_chain)
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        new_pos, _, _, _ = sampler.run_mcmc(p0, n_burn)
        sampler.reset()

        store = InMemoryStorageUtil()
        for pos, prob, _, _ in sampler.sample(new_pos, iterations=n_run):
            store.persistSamplingValues(pos, prob, None)
        return store.samples

    def mcmc_CH(self, walkerRatio, n_run, n_burn, mean_start, sigma_start, lowerLimit, upperLimit, threadCount=1, init_pos=None, mpi_monch=False):
        """
        runs mcmc on the parameter space given parameter bounds with CosmoHammerSampler
        returns the chain
        """
        params = np.array([mean_start, lowerLimit, upperLimit, sigma_start]).T

        chain = LikelihoodComputationChain(
            min=lowerLimit,
            max=upperLimit)

        temp_dir = tempfile.mkdtemp("Hammer")
        file_prefix = os.path.join(temp_dir, "logs")

        # chain.addCoreModule(CambCoreModule())
        chain.addLikelihoodModule(self.chain)
        chain.setup()

        store = InMemoryStorageUtil()
        if mpi_monch is True:
            sampler = MpiCosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=file_prefix,
            walkersRatio=walkerRatio,
            burninIterations=n_burn,
            sampleIterations=n_run,
            threadCount=1,
            initPositionGenerator=init_pos,
            storageUtil=store)
        else:
            sampler = CosmoHammerSampler(
                params=params,
                likelihoodComputationChain=chain,
                filePrefix=file_prefix,
                walkersRatio=walkerRatio,
                burninIterations=n_burn,
                sampleIterations=n_run,
                threadCount=threadCount,
                initPositionGenerator=init_pos,
                storageUtil=store)
        time_start = time.time()
        #if sampler.isMaster():
        #    print('Computing the MCMC...')
        #    print('Number of walkers = ', len(mean_start)*walkerRatio)
        #    print('Burn-in itterations: ', n_burn)
        #    print('Sampling itterations:', n_run)
        sampler.startSampling()
        #if sampler.isMaster():
        #    time_end = time.time()
        #    print(time_end - time_start, 'time taken for MCMC sampling')
        # if sampler._sampler.pool is not None:
        #     sampler._sampler.pool.close()
        try:
            shutil.rmtree(temp_dir)
        except Exception as ex:
            print(ex)
            pass
        return store.samples

