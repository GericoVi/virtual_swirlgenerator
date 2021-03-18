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

class Test(NamedTuple):
    caseNum: int = None                # Case identifier
    cmap: str = None                   # Colour map name to use, or none if extracting from image
    interp: str = 'linear'
    samplingMode: int = None           # Mode for creating the sampling distribution
    samplingParams: tuple = None       # Sampling params relevant to the sampling mode chosen - details in contour_translation.samplePoints()
    shrinkPlotMax: int = 10            # Maximum number to try and shrink the plot to attempt to crop border
    files: list = None                 # List containing the name of the two files needed to reconstruct the flow field, [tangential, radial]
    meshfile: str = None               # Name of mesh file to extract inlet mesh from
    dataFolder: str = None             # Folder containing the input images and flow field data for comparison
    outputFolder: str = None           # Folder to output images of the reconstructed flow field
    savePlots: bool = False            # Do we save the plots for this case


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
    columns = ['Colourmap','SamplingMode','SamplingParams','NumNodes','ShrinkPlotMax','CaseNum','NRMSE', 'Time']

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
    
    try:
        tangential.translateContourPlot(getColourbar=getColourbar, samplingMode=args.samplingMode, samplingParams=samplingParams)
        radial.translateContourPlot(getColourbar=getColourbar, samplingMode=args.samplingMode, samplingParams=samplingParams)

        tangentialValues = tangential.getValuesAtNodes(flowfield.coords, interpolation=args.interp)
        radialValues = radial.getValuesAtNodes(flowfield.coords, interpolation=args.interp)

        # Do the flowfield reconstruction anyway even though the values won't be used, just for the representative time
        flowfield.reconstructDomain(tangentialValues, radialValues, axialVel=1)

        end = time.time()

        # Get number of nodes
        numNodes = tangential.samples.size

        # Save a couple samples
        if args.savePlots:
            plots = post.Plots(flowfield)
            label = f'{args.caseNum}_cmap_{cmapName}_samplingMode_{args.samplingMode}_shrink_{args.shrinkPlotMax}.pdf'
            plots.plotInletNodes(show=False)
            plots.plotAll(pdfName=os.path.join(args.outputFolder,label), swirlAxisRange=[tanRng[0],tanRng[1],radRng[0],radRng[1]], swirlAxisNTicks=11)

        # Get RMSE of extracted tangential and radial flow angles separately
        rmse_tan = post.SwirlDescriptors.getError(correctTangential, tangentialValues)
        rmse_rad = post.SwirlDescriptors.getError(correctRadial, radialValues)

        # Get range of observations 
        rng_tan = np.nanmax(tangentialValues) - np.nanmin(tangentialValues)
        rng_rad = np.nanmax(radialValues) - np.nanmin(radialValues)

        # Get normalised root mean square errors using mean
        nrmse_tan = rmse_tan / rng_tan
        nrmse_rad = rmse_rad / rng_rad

        # Get avg of these
        nrmse = (nrmse_tan + nrmse_rad) / 2

        return pd.DataFrame([[cmapName,args.samplingMode,args.samplingParams,numNodes,args.shrinkPlotMax,args.caseNum,nrmse,end-start]],
                            columns=columns)

    except BaseException as e:
        print(f'Case number {args.caseNum} failed with cmap={cmapName}, sampling mode={args.samplingMode}, sampling params={args.samplingParams}, shrink plot={args.shrinkPlotMax}')
        print(f'Error message: {e}')

        return pd.DataFrame([[None]*8],
                            columns=columns)

if __name__ == '__main__':

    resultsFile = 'results.csv'
    meshfile = os.path.join(parentpath, 'cylinder.su2')

    # Tuple for params of each sampling mode 
    sampling_modes = ((1, (40)), (2, (40, 2.8)), (1, (34)), (2, (34, 3.3)), (3, (None)), (1, (20)), (2, (20, 5.4)), (1, (10)), (2, (10, 10)), (1, (5)), (2, (5, 18.4)))

    #cmaps = [None, 'jet']
    #shrink = [0,10]
    cmaps = ['jet']
    shrink = [0]

    try:
        numProcesses = int(sys.argv[1])
    except:
        numProcesses = mp.cpu_count()-1

    script_start_time = time.time()

    imgfolder = 'images'
    outfolder = 'results'

    # Create the necessary folders if they're not present
    if not os.path.exists(imgfolder):
        os.mkdir(imgfolder)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    # Get list of files in test case folder
    files = [f for f in os.listdir(imgfolder) if (os.path.isfile(os.path.join(imgfolder, f)) and f.split('.')[-1].lower() == 'png')]
    numCases = int(len(files)/3)

    numTests = len(cmaps)*len(shrink)*len(sampling_modes)*numCases

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
    print(f'\nStarting tests on parallel pool of {numProcesses} workers...')
    startTime = time.time()
    pool = mp.Pool(processes = (numProcesses))
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

 
