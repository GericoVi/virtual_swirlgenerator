# Virtual_StreamVane
 Python tool for creating arbitrary swirling inflow boundary conditions for use with a CFD framework. Created as part of a Master's dissertation for MEng Aerospace Engineering.
 
 Inspired by **'A New Method for Generating Swirl Inlet Distortion for Jet Engine Research'** paper by K. Hoopes (2013). This is an attempt to develop the same capability for aiding work in the numerical (CFD) domain.

## Dependencies
- numpy (1.19.3)
- matplotlib
- scipy
- alphashape
- opencv-python
- gmsh - only needed if planning to create test meshes with the *maketestdomain* module, rather than providing a mesh


## Basic Usage
### Command Line Control
`python swirlgenerator <config file>` performs the basic functionality of the tool. Generates a boundary condition according to the options specified in the config file. A template config file has been included in the repository which shows all available functionality/configurations.
 
`python /swirlgenerator -help` shows the command line options available for performing extra functionality. 


### Calling from user scripts
The generic workflow and method for generating boundary conditions using swirlgenerator functions is illustrated within **Main.py**. This serves as a showcase for the capabilities of the tool.

**However an outline of possible workflows is shown here (if the user prefers not to use a config file or for easier integration with existing python scripts):**

A list of node coordinates needs to be provided. Swirlgenerator provides capability for extracting the nodes making up the inlet plane of a mesh from the **pre.py** module.
```
nodes = pre.Input.extractMesh(meshfilename)
```
This can be bypassed if the user already has the node coordinates - in the form of a list of 3D vectors, shape (n,3).

These node coordinates are used to initialise a *FlowField* class instance.
```
flowfield = core.FlowField(nodes)
```

From here, there are two methods which can be used to generate a swirling inlet velocity field:
- creating a velocity profile by combining the effect of multiple discrete vortices (from core.py module).
```
# Initialise the domain configuration with information about each vortex to be defined within the domain.
vortexDefs = core.Vortices(model, centres, strengths, core_radii)

# Calculate the velocity field with
flowfield.computeDomain(vortexDefs, axialVel)
```

- reconstructing a velocity profile by translating contour plot images of tangenetial and radial flow angles (from contour_translation.py module). Swirlgenerator has the capability of extracting the colour map from a colour bar within the image, but specifying a colour map name gives significant accuracy increase.
```
import contour_translation as ct

# Estimate the values used to create the plots within the images using inverse colour mapping
tangential = ct.Contour(image_of_tangential_flow_angle_plot, colourbar_range, flowfield.coords, cmap=)
radial     = ct.Contour(image_of_radial_flow_angle_plot,     colourbar_range, flowfield.coords, cmap=)

# Reconstruct the velocity field with [the (uniform) axial velocity of the bulk flow needs to be provided also]
flowfield.reconstructDomain(tangential.values, radial.values, axial_velocity)
```

The actual boundary condition data file can then be created, in this case for an SU2 simulation.
```
writeBC.writeSU2(flowfield, filename)
```


## Method of generation
1. Calculated by placing multiple discrete vortices into the domain and modelling how they would interact with simple velocity vector summations.

2. Image recognition capability for receiving contour plots of tangential and radial flow angles and generating an approximation of the underlying velocity field.


## Limitations
- Only su2 format for the boundary condition is currently supported by the *writeBC* modules
- Only su2 mesh format can be read directly by the *pre* module
- Only supports incompressible boundary conditions
