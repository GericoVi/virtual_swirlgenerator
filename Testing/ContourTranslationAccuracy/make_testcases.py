# -------------------------------------------------------------------------------------------------------------------
#
# Creates test cases with the vortex method and saves both the contour plot images and the numerical data
# Using a range of random vortex configurations
#
# -------------------------------------------------------------------------------------------------------------------

import sys
import os
import time
import numpy as np
import multiprocessing as mp
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import NamedTuple

# Get useful paths and add swirlgenerator modules folder to python path
scriptpath = os.path.split(os.path.realpath(__file__))[0]
idx = scriptpath.find('Testing')
parentpath = scriptpath[:idx]
sys.path.append(os.path.join(parentpath,'swirlgenerator'))

# Import swirlgenertor modules
import core as sg
import pre
import post

'''
Named Tuple classes for storing arguements better
'''
class TestCase(NamedTuple):
    caseNum: int = None                # Case identifier - for consitent rng seed
    maxNumVort: int = None             # Max number of vortices to randomly generate
    maxStrengthVort: float = None      # Max strength of a vortex when randomly generating
    maxCoreVort: float = None          # Max core radius of a vortex when randomly generating
    bl: float = 0                      # Reference length for the boundary layer model
    meshfile: str = None               # Name of mesh file to extract inlet mesh from
    outFolder: str = None              # Name of folder to output images and data

def makeTestCase(args: TestCase):
    # Initialise rng here with case specific seeds - order of execution not guaranteed so need to be careful with the random number stream
    random.seed(a=args.caseNum*9999)

    # Set up descritisation
    flowfield = sg.FlowField(pre.Input.extractMesh(args.meshfile))

    # Get duct radius
    duct_radius = np.max(np.abs(flowfield.coords))

    numVortices     = random.randint(1,args.maxNumVort)
    vortCentres     = [None]*numVortices
    vortStrengths   = [None]*numVortices
    vortCores       = [None]*numVortices

    for j in range(numVortices):
        vortCentres[j]      = [random.uniform(-duct_radius,duct_radius),random.uniform(-duct_radius,duct_radius)]
        vortStrengths[j]    = random.uniform(-args.maxStrengthVort,args.maxStrengthVort)
        vortCores[j]        = random.uniform(0,args.maxCoreVort)

    vortexDefs = sg.Vortices(model='lo', centres=vortCentres, strengths=vortStrengths, radius=vortCores)
    flowfield.computeDomain(vortexDefs, axialVel=1)

    # Model a boundary layer within the inlet condition if requested
    if args.bl > 0:
        flowfield.makeBoundaryLayer(args.bl)

    # Save flowfield variables so we can get the 'correct' values later
    flowfield.save(os.path.join(args.outFolder,f'{args.caseNum}'))

    # Plot the tangential and radial flow angles and save as images
    plots = post.Plots(flowfield)
    plots.plotSwirl()
    tanRange = post.Plots.__getContourRange__(plots.swirlAngle)
    radRange = post.Plots.__getContourRange__(plots.radialAngle)
    f = plt.figure(1)
    f.savefig(os.path.join(args.outFolder,f'{args.caseNum}_tangential_{tanRange}.png'))
    f = plt.figure(2)
    f.savefig(os.path.join(args.outFolder,f'{args.caseNum}_radial_{radRange}.png'))

    # Save streamlines plot as well for interest
    plots.plotVelocity()
    f = plt.figure(4)
    f.savefig(os.path.join(args.outFolder,f'{args.caseNum}_streamlines.png'))
    plt.close('all')

    return None

if __name__ == '__main__':

    imgfolder = 'images'
    meshfile = os.path.join(parentpath, 'cylinder.su2')

    # Bounds for randomly generated vortex cases
    maxNumVort = 10
    maxStrengthVort = 0.1
    maxCoreVort = 0.05

    try:
        numCases = int(sys.argv[1])
    except:
        raise RuntimeError('Valid number of samples not provided for first argument')

    print(f'Proceeding with numCases = {numCases}')

    try:
        numProcesses = int(sys.argv[1])
    except:
        numProcesses = mp.cpu_count()-1

    script_start_time = time.time()

    # Create the necessary folders if they're not present
    if not os.path.exists(imgfolder):
        os.mkdir(imgfolder)

    # Make list of argument tuples for parallel processing
    args_list = [None]*numCases
    for j in range(numCases):
        args_list[j] = TestCase(caseNum=j, maxNumVort=maxNumVort, maxStrengthVort=maxStrengthVort, maxCoreVort=maxCoreVort, meshfile=meshfile, outFolder=imgfolder)

    # Make all test images using pool of processes
    print(f'Making flowfields for test images with parallel pool of {numProcesses} workers...')
    startTime = time.time()
    pool = mp.Pool(processes = (numProcesses))
    _ = list(tqdm(pool.imap(makeTestCase, args_list), total=numCases))
    pool.close()
    pool.join()
    endTime = time.time()
    print(f'{numCases} test cases created in {imgfolder}')

    print(f'\n\nElapsed time: {time.time()-script_start_time}')