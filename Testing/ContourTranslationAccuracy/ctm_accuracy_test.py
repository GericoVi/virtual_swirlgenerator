# -------------------------------------------------------------------------------------------------------------------
#
# For testing the accuracy of the flow field reconstruction with different conditions from contour plots
# Using contour plots created from a flow field calculated with the vortex method
# - so that we have the exact values to compare to
# Using a range of random vortex configurations
#
# -------------------------------------------------------------------------------------------------------------------

import sys
import os
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from typing import NamedTuple

# Get useful paths and add swirlgenerator modules folder to python path
scriptpath = os.path.split(os.path.realpath(__file__))[0]
idx = scriptpath.find('Testing')
parentpath = scriptpath[:idx]
sys.path.append(os.path.join(parentpath,'swirlgenerator'))

# Import swirlgenertor modules
import core as sg
import contour_translation as ct
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
    bl: float = None                   # Reference length for the boundary layer model
    meshfile: str = None               # Name of mesh file to extract inlet mesh from
    outFolder: str = None              # Name of folder to output images and data

class Test(NamedTuple):
    caseNum: int = None                # Case identifier
    cmap: str = None                   # Colour map name to use, or none if extracting from image
    samplingMode: int = None           # Mode for creating the sampling distribution
    samplingParams: tuple = None       # Sampling params relevant to the sampling mode chosen - details in contour_translation.samplePoints()
    shrinkPlotMax: int = 10            # Maximum number to try and shrink the plot to attempt to crop border
    files: list = None                 # List containing the name of the two files needed to reconstruct the flow field, [tangential, radial]
    meshfile: str = None               # Name of mesh file to extract inlet mesh from
    dataFolder: str = None             # Folder containing the input images and flow field data for comparison
    outputFolder: str = None           # Folder to output images of the reconstructed flow field
    savePlots: bool = False            # Do we save the plots for this case


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


def doTest(args: Test):

    if args.cmap is not None:
        getColourbar = False
        cmapName = args.cmap
        cmap = cm.get_cmap(args.cmap)
        cmap = cmap(range(cmap.N))[:,0:3]
    else:
        getColourbar = True
        cmapName = 'None'
        cmap = args.cmap

    # For creating pandas dataframe
    columns = ['Colourmap','SamplingMode','SamplingParams','ShrinkPlotMax','CaseNum','RMSE', 'Time']

    # All flowfields are using the same mesh so we can reuse the same discretisation
    flowfield = sg.FlowField(pre.Input.extractMesh(args.meshfile))

    # Load original flow field variables
    flowfield.load(os.path.join(args.dataFolder,f'{args.caseNum}.npz'))
    # Protection for old format, which didn't have radial angles saved
    flowfield.getSwirl()

    correctTangential = flowfield.swirlAngle
    correctRadial     = flowfield.radialAngle

    # Get image files for this case
    tanImg = args.files[0]
    radImg = args.files[1]

    # Get ranges
    tanRng = list(float(numString) for numString in ('.'.join(tanImg.split('_')[-1].split('.')[:-1])[1:-1]).split(','))
    radRng = list(float(numString) for numString in ('.'.join(radImg.split('_')[-1].split('.')[:-1])[1:-1]).split(','))

    start = time.time()
    tangential = ct.Contour(os.path.join(args.dataFolder,tanImg), tanRng, cmap=cmap, doTranslation=False)
    radial = ct.Contour(os.path.join(args.dataFolder,radImg), radRng, cmap=cmap, doTranslation=False)

    if args.samplingMode == 3:
        samplingParams = flowfield.coords
    else:
        samplingParams = args.samplingParams
    
    tangential.translateContourPlot(getColourbar=getColourbar, samplingMode=args.samplingMode, samplingParams=samplingParams)
    radial.translateContourPlot(getColourbar=getColourbar, samplingMode=args.samplingMode, samplingParams=samplingParams)

    tangentialValues = tangential.getValuesAtNodes(flowfield.coords, interpolation=args.interp)
    radialValues = radial.getValuesAtNodes(flowfield.coords, interpolation=args.interp)

    # Do the flowfield reconstruction anyway even though the values won't be used, just for the representative time
    flowfield.reconstructDomain(tangentialValues, radialValues, axialVel=1)

    end = time.time()

    # Save a couple samples
    if args.savePlots:
        plots = post.Plots(flowfield)
        label = f'{args.caseNum}_cmap_{cmapName}_samplingMode_{args.samplingMode}_shrink_{args.shrinkPlotMax}.pdf'
        plots.plotInletNodes(show=False)
        plots.plotAll(pdfName=os.path.join(args.outputFolder,label), swirlAxisRange=[tanRng[0],tanRng[1],radRng[0],radRng[1]], swirlAxisNTicks=11)

    # Concatenate the arrays to get the RMSE across both tangential and radial 
    values          = np.hstack([tangentialValues, radialValues])
    correctValues   = np.hstack([correctTangential, correctRadial])

    rmse = post.SwirlDescriptors.getError(correctValues, values)

    return pd.DataFrame([[cmapName,args.samplingMode,args.samplingParams,args.shrinkPlotMax,args.caseNum,rmse,end-start]],
                        columns=columns)

if __name__ == '__main__':

    resultsFile = 'results.csv'
    meshfile = os.path.join(parentpath, 'cylinder.su2')
    numWorkers = mp.cpu_count()-1
    #numWorkers = 5

    # Tuple for params of each sampling mode 
    sampling_modes = ((1, (34)), (2, (34, 3.3)), (3, (None)), (1, (20)), (2, (20, 5.4)), (1, (10)), (2, (10, 10)), (1, (5)), (2, (5, 18.4)))

    #cmaps = [None, 'jet']
    #shrink = [0,10]
    cmaps = ['jet']
    shrink = [0]
    
    # Bounds for randomly generated vortex cases
    numCases = 100
    maxNumVort = 10
    maxStrengthVort = 0.1
    maxCoreVort = 0.05

    script_start_time = time.time()

    imgfolder = 'images'
    outfolder = 'results'

    numTests = len(cmaps)*len(shrink)*len(sampling_modes)*numCases

    # Create the necessary folders if they're not present
    if not os.path.exists(imgfolder):
        os.mkdir(imgfolder)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    # Make list of argument tuples for parallel processing
    args_list = [None]*numCases
    for j in range(numCases):
        args_list[j] = TestCase(caseNum=j, maxNumVort=maxNumVort, maxStrengthVort=maxStrengthVort, maxCoreVort=maxCoreVort, meshfile=meshfile, outFolder=imgfolder)

    # Make all test images using pool of processes
    print(f'Making flowfields for test images with parallel pool of {numWorkers} workers...')
    startTime = time.time()
    pool = mp.Pool(processes = (numWorkers))
    _ = list(tqdm(pool.imap(makeTestCase, args_list), total=numCases))
    pool.close()
    pool.join()
    endTime = time.time()
    print(f'{numCases} test cases created in {imgfolder}')


    # Get list of files in test case folder
    files = [f for f in os.listdir(imgfolder) if (os.path.isfile(os.path.join(imgfolder, f)) and f.split('.')[-1].lower() == 'png')]

    # Create list of arguments for each test case
    args_i = 0
    args_list = [None]*numTests
    for sampling_tuple in sampling_modes:
        for cmap in cmaps:
            for sMax in shrink:
                for k in range(numCases):
                    # Get image files for this case
                    tanImg = [f for f in files if f.find(f'{k}_tan') == 0][0]
                    radImg = [f for f in files if f.find(f'{k}_rad') == 0][0]

                    # Put into tuple
                    img_files = (tanImg, radImg)

                    # Save plots or not
                    if (k == 0) or (k == 49) or (k == 99):
                        savePlots = True
                    else:
                        savePlots = False

                    args_list[args_i] = Test(caseNum=k, cmap=cmap, samplingMode=sampling_tuple[0], samplingParams=sampling_tuple[1], shrinkPlotMax=sMax, files=img_files, meshfile=meshfile, dataFolder=imgfolder, outputFolder=outfolder, savePlots=savePlots)

                    args_i += 1

    # Do all test cases using a pool of processes
    print(f'\nStarting tests on parallel pool of {numWorkers} workers...')
    startTime = time.time()
    pool = mp.Pool(processes = (numWorkers))
    results = list(tqdm(pool.imap(doTest, args_list), total=len(args_list)))
    pool.close()
    pool.join()

    results_df = pd.concat(results)

    endTime = time.time()

    print(results_df)
    
    print(f'\n\nDone. Elapsed time: {time.time()-script_start_time}')
    
    try:
        results_df.to_csv(resultsFile,index=False)
    except:
        filename = time.time()
        print(f'Permission denied for {resultsFile}. Saving to {filename}.csv')
        results_df.to_csv(f'{filename}.csv',index=False)

 
