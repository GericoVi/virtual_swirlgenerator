# -------------------------------------------------------------------------------------------------------------------
#
# For running batches of tests for the 4 idealised test cases from K.N. Smith paper
#
# -------------------------------------------------------------------------------------------------------------------

import sys
import os
import time

from matplotlib.cm import get_cmap

# Get path of the swirlgenerator modules
scriptpath = os.path.realpath(__file__)
modulespath = '\\'.join(scriptpath.split('\\')[:5])
modulespath = os.path.join(modulespath, 'swirlgenerator')

# Insert path to to swirlgenerator during runtime so we can import them
sys.path.insert(1, modulespath)

# Import swirlgenertor modules
import core as sg
import maketestdomain as domain
import contour_translation as ct
import pre
import post


# Testcases
testcases = ['bulkswirl','twinswirl','offsetswirl1','offsetswirl2']
names = ['Bulk Swirl', 'Twin Swirl', 'Offset Swirl 1', 'Offset Swirl 2']
makemesh = [True, True, False, False]

with open('results.txt', 'w') as f:
    for i,case in enumerate(testcases):
        print(f'\nDoing case: {case}')
        f.write(f'\n{names[i]}\n')

        # Form filenames
        configfile = f'{case}/{case}.config'
        highres_tangential = f'{case}/{case}_tangential.png'
        lowres_tangential = f'{case}/{case}_tangential_lowres.png'
        highres_radial = f'{case}/{case}_radial.png'
        lowres_radial = f'{case}/{case}_radial_lowres.png'

        # Get default configuration for this test case
        inputdata = pre.Input(configfile)

        # Make the domain
        if makemesh[i]:
            domain.testDomain(inputdata, inputdata.meshfilename)

        # Initialise flow field object with nodes from mesh
        print(f'Extracting nodes...')
        flowfield = sg.FlowField(inputdata.getNodes())

        print(f'Vortex method test...')
        # Create the inlet velocity field using discrete vortices
        start = time.time()
        vortexDefs = sg.Vortices(inputObject=inputdata)
        flowfield.computeDomain(vortexDefs, axialVel=inputdata.axialVel)
        end = time.time()

        # Save plots
        if not os.path.exists(f'{case}/results'):
            os.makedirs(f'{case}/results')
        plots = post.Plots(flowfield)
        plots.plotAll(pdfName=f'{case}/results/VM.pdf', swirlAxisRange=inputdata.swirlPlotRange, swirlAxisNTicks=inputdata.swirlPlotNTicks)

        # Construct arrays so we can iterate
        tan_images = [highres_tangential, lowres_tangential, highres_tangential, lowres_tangential]
        rad_images = [highres_radial, lowres_radial, highres_radial, lowres_radial]
        labels = ['high res image', 'low res image', 'high res image with cmap specified', 'low res image with cmap specified']
        cmaps = [None, None, 'jet', 'jet']

        # Run the different tests for recontructing the velocity fields using the contour plots
        print(f'Reconstruction method test...')
        for j,label in enumerate(labels):
            # Get colour map
            if cmaps[j] is not None:
                cmap = get_cmap(cmaps[j])
                cmap = cmap(range(cmap.N))[:,0:3]
            else:
                cmap = cmaps[j]

            print(f'{label}...')
            start = time.time()
            tangential = ct.Contour(tan_images[j], inputdata.tanRng, flowfield.coords, cmap=cmap)
            radial = ct.Contour(rad_images[j], inputdata.radRng, flowfield.coords, cmap=cmap)

            flowfield.reconstructDomain(tangential.values, radial.values, inputdata.axialVel)
            end = time.time()

            # Save plots
            if not os.path.exists(f'{case}/results'):
                os.makedirs(f'{case}/results')
            plots = post.Plots(flowfield)
            plots.plotAll(pdfName=f'{case}/results/{label}.pdf', swirlAxisRange=inputdata.swirlPlotRange, swirlAxisNTicks=inputdata.swirlPlotNTicks)



