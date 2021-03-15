import sys
import os
import random
import time
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# Get useful paths and add swirlgenerator modules folder to python path
scriptpath = os.path.split(os.path.realpath(__file__))[0]
idx = scriptpath.find('Testing')
parentpath = scriptpath[:idx]
sys.path.append(os.path.join(parentpath,'swirlgenerator'))

# Import swirlgenertor modules
import core as sg
import pre
import post

def make_vortex_and_get_flux(args):
    # Extract arguments from input tuple
    caseNum, maxNumVort, maxStrengthVort, maxCoreVort, meshfile, figuresfolder = args

    # Initialise rng here with case specific seeds - order of execution not guaranteed so need to be careful with the random number stream
    random.seed(a=caseNum*9999)

    # Set up descritisation
    flowfield = sg.FlowField(pre.Input.extractMesh(meshfile))

    # Get duct radius
    duct_radius = np.max(np.abs(flowfield.coords))

    numVortices     = random.randint(1,maxNumVort)
    vortCentres     = [None]*numVortices
    vortStrengths   = [None]*numVortices
    vortCores       = [None]*numVortices

    for j in range(numVortices):
        vortCentres[j]      = [random.uniform(-duct_radius+0.001,duct_radius-0.001),random.uniform(-duct_radius,duct_radius)]
        vortStrengths[j]    = random.uniform(-maxStrengthVort,maxStrengthVort)
        vortCores[j]        = random.uniform(0,maxCoreVort)

    vortexDefs = sg.Vortices(model='lo', centres=vortCentres, strengths=vortStrengths, radius=vortCores)
    flowfield.computeDomain(vortexDefs, axialVel=1)

    if ((caseNum+1) % 100 == 0) or (caseNum == 0):
        # Get flux across boundary
        fluxOut = flowfield.checkBoundaries(plot=True)

        # Plot the tangential and radial flow angles
        plots = post.Plots(flowfield)
        plots.plotSwirl()

        # Plot streamlines as well for interest
        plots.plotVelocity()

        # Save all figs
        post.Plots.saveFigsToPdf(os.path.join(figuresfolder,f'{caseNum}.pdf'))

    else:
        # Get flux across boundary
        fluxOut = flowfield.checkBoundaries()

    return fluxOut


if __name__ == '__main__':

    resultsFile = 'results'
    meshfile = os.path.join(parentpath, 'cylinder.su2')
    figuresfolder = 'results'

    maxNumVort = 10
    maxStrengthVort = 2
    maxCoreVort = 0.5

    try:
        numSamples = int(sys.argv[1])
    except:
        raise RuntimeError('Valid number of samples not provided for first argument')

    print(f'Proceeding with numSamples = {numSamples}')

    try:
        numProcesses = int(sys.argv[2])
    except:
        numProcesses = mp.cpu_count()-1
    #numProcesses = 5

    print(f'\nDoing flux across boundary tests with parallel pool of {numProcesses} workers...')

    if not os.path.exists(figuresfolder):
            os.mkdir(figuresfolder)

    # Make list of argument tuples for parallel processing
    args_list = [None]*numSamples
    for j in range(numSamples):
        args_list[j] = (j, maxNumVort, maxStrengthVort, maxCoreVort, meshfile, figuresfolder)

    script_start_time = time.time()
    pool = mp.Pool(processes = (numProcesses))
    results = list(tqdm(pool.imap(make_vortex_and_get_flux, args_list), total=numSamples))
    pool.close()
    pool.join()

    script_end_time = time.time()

    print(f'\nDone. Elapsed time: {script_end_time-script_start_time}')
    #print(results)

    # Save as numpy file
    np.savez(resultsFile, results=results)
    # Save as csv
    np.savetxt(f'{resultsFile}.csv', results)

    npzfile = np.load(f'{resultsFile}.npz', allow_pickle=True)

    results = npzfile['results']

    print(f'Mean specific flux out per unit perimeter = {results.mean()} m/s;  Variance = {results.var()}')