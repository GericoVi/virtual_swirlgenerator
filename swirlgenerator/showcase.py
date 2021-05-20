import core
import contour_translation as ct
import pre
import post
import writeBC

import os
import time

# Get the path the repo has been cloned to
scriptpath = os.path.dirname(os.path.realpath(__file__))
parentpath = os.path.dirname(scriptpath)

# Mesh file path
meshfile = os.path.join(parentpath, 'cylinder.su2')

# Extract nodes from mesh
nodes = pre.Input.extractMesh(meshfile)

##### 
##### Vortex method
#####
print('Doing vortex method...')
start = time.time()

# Initialise FlowField object with the nodes
flowfield_vm = core.FlowField(nodes)

# Specify the individual vortices to be placed in the domain
vortexDefs = core.Vortices(model='lo', centres=[[0.083, 0.0], [-0.083, 0.0]], strengths = [1.243, -1.243], radius=[0.25,0.25])
# Place vortices into domain and calculate velocity effects
flowfield_vm.computeDomain(vortexDefs, axialVel=1)

# Write boundary condition file
writeBC.writeSU2(flowfield_vm, 'twinvortex_vm.dat')

print(f'Elapsed time: {time.time()-start}s\n')

# Visualise flowfield and save to pdf
print('Making plots...')
start = time.time()

plots = post.Plots(flowfield_vm)
plots.plotAll(pdfName='twinvortex_vm.pdf')      # Shows figures on screen if pdfName parameter is not supplied

print(f'Elapsed time: {time.time()-start}s\n')


##### 
##### Contour Translation Method
#####
print('Doing contour translation method...')
start = time.time()

# Initialise FlowField object with the nodes
flowfield_ctm = core.FlowField(nodes)

# Path to images
tangentialImage = os.path.join(parentpath, 'Testing', 'IdealisedSwirlCases', 'twinswirl_tangential.PNG')
radialImage = os.path.join(parentpath, 'Testing', 'IdealisedSwirlCases', 'twinswirl_radial.PNG')

# Initialise contour translation objects but don't translate yet
tangential = ct.Contour(tangentialImage, [-20,20], doTranslation=False)
radial = ct.Contour(radialImage, [-20,20], doTranslation=False)

# Translate contour plots (with some given parameters) to data
tangential.translateContourPlot(samplingMode=1, samplingParams=(10), circleParams=(200,30), minLevels=100, shrinkPlotMax=10)
radial.translateContourPlot(samplingMode=1, samplingParams=(10), circleParams=(200,30), minLevels=100, shrinkPlotMax=10)

# Reconstruct flow field from contour plot data
flowfield_ctm.reconstructDomain(tangential.getValuesAtNodes(flowfield_ctm.coords), \
                                radial.getValuesAtNodes(flowfield_ctm.coords), axialVel=1)

# Write boundary condition file
writeBC.writeSU2(flowfield_ctm, 'twinvortex_ctm.dat')

print(f'Elapsed time: {time.time()-start}s\n')

# Visualise flowfield and save to pdf
print('Making plots...')
start = time.time()

plots = post.Plots(flowfield_ctm)
plots.plotAll(pdfName='twinvortex_ctm.pdf')

print(f'Elapsed time: {time.time()-start}s\n')



#####
##### Extras
#####

# Approximate a wall boundary layer
print('Modelling wall boundary layer...')
flowfield_vm.makeBoundaryLayer(ref_len = 5)

print('Making plots...')
plots = post.Plots(flowfield_vm)
plots.plotAll(pdfName='twinvortex_vm_bl.pdf')
