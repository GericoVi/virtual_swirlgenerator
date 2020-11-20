# --------------------------------------------------------------------------------------------------------
#
# Generates simple geometry and meshes which are compatible with the generated boundary condition
# for testing the development of swirling profiles created by swirlGenerator
# using the python API of GMSH
# https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python
#
# --------------------------------------------------------------------------------------------------------

import gmsh
import core as sg
import pre

def testDomain(InputData: pre.Input, meshfilename, showmesh=False, verbose=False):
    '''
    Wrapper function which calls the correct function depending on the inlet shape specified
    - InputData - Input object from core.py, containing geometry and mesh properties
    - meshfilename - the filename's extension will determine what format to write the mesh
    - showmesh - flag for showing the mesh using the gmsh gui, not recommended for large meshes
    '''

    # Dictionary to map inlet shape to correct function
    domains = {'circle': simpleCylinder, 'rect': simpleBox}

    # Initialise gmsh api
    gmsh.initialize()

    # Set verbosity level
    if verbose:
        verbosity = 5
    else:
        verbosity = 0

    gmsh.option.setNumber("General.Verbosity", verbosity)

    # Call appropriate function using handle to create the geometry and mesh settings
    print('Creating geometry...')
    func = domains.get(InputData.shape)
    func(InputData)

    # Before meshing, the CAD entities created need to be synchronised with the GMsh model
    gmsh.model.geo.synchronize()

    # Generate 3D mesh
    print('Generating mesh...')
    gmsh.model.mesh.generate(3)

    # Write mesh to file
    gmsh.write(meshfilename)
    print(f"Mesh file written to {meshfilename}")

    # Visualise model in gui
    if showmesh:
        gmsh.fltk.run()

    gmsh.finalize()


def simpleBox(InputData: pre.Input):
    '''
    Writes a simple 3D rectangular geometry and mesh settings into the gmsh instance
    - InputData - Input object from core.py, containing geometry and mesh properties
    - Uses gmsh python api
    '''

    # Get vertices of face
    vertices = [[InputData.xSide/2, InputData.ySide/2], [InputData.xSide/2, -InputData.ySide/2], [-InputData.xSide/2, -InputData.ySide/2], [-InputData.xSide/2, InputData.ySide/2]] 

    # Add a new geometry model
    gmsh.model.add("box")

    # Define the points of the box
    points = [gmsh.model.geo.addPoint(point[0],point[1],0) for point in vertices]

    # Define the lines between the points - Needs to be done in a certain order for later
    lines = []
    # Define the lines for the rectangular face
    lines.append(gmsh.model.geo.addLine(points[0],points[1]))
    lines.append(gmsh.model.geo.addLine(points[1],points[2]))
    lines.append(gmsh.model.geo.addLine(points[2],points[3]))
    lines.append(gmsh.model.geo.addLine(points[3],points[0]))

    # Define the loop around the rectangle
    loop = gmsh.model.geo.addCurveLoop(lines)

    # Define rectangular surface 
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Recombine triangular mesh into quadrilaterals
    gmsh.model.geo.mesh.setRecombine(2, surface)

    # Define a transfinite mesh on the lines
    [gmsh.model.geo.mesh.setTransfiniteCurve(line, InputData.xNumCells+1) for line in [lines[1],lines[3]]]
    [gmsh.model.geo.mesh.setTransfiniteCurve(line, InputData.yNumCells+1) for line in [lines[0],lines[2]]]

    # Define the surface as transfinite - so the 2D mesh is also structured
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)

    # Extrude this mesh in the z direction to get the duct
    gmsh.model.geo.extrude([[2, surface]], 0, 0, InputData.zSide, numElements=[InputData.zNumCells], recombine=True)

    # Set physical groups for the surfaces so that boundary conditions can be defined on them - surface tags found from looking at the generated geometry
    pSurfaces = [gmsh.model.addPhysicalGroup(2, [1])]
    pSurfaces.append(gmsh.model.addPhysicalGroup(2, [26]))
    pSurfaces.append(gmsh.model.addPhysicalGroup(2, [13,17,21,25]))

    # Name the physical surfaces for applying boundary conditions
    gmsh.model.setPhysicalName(2, pSurfaces[0], "inlet")
    gmsh.model.setPhysicalName(2, pSurfaces[1], "outlet")
    gmsh.model.setPhysicalName(2, pSurfaces[2], "walls")

    # Set 3D Physical group for domain volume
    pVol = gmsh.model.addPhysicalGroup(3, [1])
    gmsh.model.setPhysicalName(3, pVol, "domain")


def simpleCylinder(InputData: pre.Input):
    '''
    Writes a simple 3D cylindrical duct geometry and mesh settings into the gmsh instance
    - InputData - Input object from core.py, containing geometry and mesh properties
    - Uses gmsh python api
    '''

    # For convenience (used to place the corner points to anchor the circle)
    cos = 0.70710678118            # cos45
    r = InputData.radius

    # Points for creating the circular face
    pts = [[0,0],                                                                                       # Mid point for creating the circular arcs
            [r/2*cos, r/2*cos], [r/2*cos, -r/2*cos], [-r/2*cos, -r/2*cos], [-r/2*cos, r/2*cos],         # Corners of inner square region
            [r*cos, r*cos], [r*cos, -r*cos], [-r*cos, -r*cos], [-r*cos, r*cos]]                         # Points on the circle in line with inner square corners

    # Add a new geometry model
    gmsh.model.add("cylinder")

    # Define the points in the gmsh geometry
    points = [gmsh.model.geo.addPoint(point[0],point[1],0) for point in pts]

    lines = []
    # Define the lines for the inner square region
    lines.append(gmsh.model.geo.addLine(points[1],points[2]))
    lines.append(gmsh.model.geo.addLine(points[2],points[3]))
    lines.append(gmsh.model.geo.addLine(points[3],points[4]))
    lines.append(gmsh.model.geo.addLine(points[4],points[1]))

    # Define the circle arcs
    lines.append(gmsh.model.geo.addCircleArc(points[5],points[0],points[6]))
    lines.append(gmsh.model.geo.addCircleArc(points[6],points[0],points[7]))
    lines.append(gmsh.model.geo.addCircleArc(points[7],points[0],points[8]))
    lines.append(gmsh.model.geo.addCircleArc(points[8],points[0],points[5]))

    # Define the lines connecting the square corners to the circle
    lines.append(gmsh.model.geo.addLine(points[1],points[5]))
    lines.append(gmsh.model.geo.addLine(points[2],points[6]))
    lines.append(gmsh.model.geo.addLine(points[3],points[7]))
    lines.append(gmsh.model.geo.addLine(points[4],points[8]))

    loops = []
    # Define the loop around the square section
    loops.append(gmsh.model.geo.addCurveLoop(lines[0:4]))

    # Define the loops around the circle quadrants - direction the lines were defined in matter
    loops.append(gmsh.model.geo.addCurveLoop([-lines[0], lines[8], lines[4], -lines[9]]))
    loops.append(gmsh.model.geo.addCurveLoop([-lines[1], lines[9], lines[5], -lines[10]]))
    loops.append(gmsh.model.geo.addCurveLoop([-lines[2], lines[10], lines[6], -lines[11]]))
    loops.append(gmsh.model.geo.addCurveLoop([-lines[3], lines[11], lines[7], -lines[8]]))

    # Define the plane surfaces using these loops
    surfaces = [gmsh.model.geo.addPlaneSurface([loop]) for loop in loops]

    # Recombine surfaces so that the triangular cells get converted to quadrilaterals where possible
    [gmsh.model.geo.mesh.setRecombine(2, surface) for surface in surfaces]

    # Define a transfinite mesh on the tangential and radial lines
    [gmsh.model.geo.mesh.setTransfiniteCurve(line, InputData.curveNumCells+1) for line in lines[0:8]]
    [gmsh.model.geo.mesh.setTransfiniteCurve(line, InputData.radialNumCells+1) for line in lines[8:12]]

    # Define the surface as transfinite - so the 2D mesh is also structured
    [gmsh.model.geo.mesh.setTransfiniteSurface(surface) for surface in surfaces]

    # Make dim and tag pairs for extrusion
    pairs = [[2,surface] for surface in surfaces[0:5]]
    # Extrude this mesh in the z direction to get the duct
    gmsh.model.geo.extrude(pairs, 0, 0, InputData.zSide, numElements=[InputData.zNumCells], recombine=True)

    # Set physical groups for the surfaces so that boundary conditions can be defined on them - surface tags found from looking at the generated geometry
    pSurfaces = [gmsh.model.addPhysicalGroup(2, [1,2,3,4,5])]
    pSurfaces.append(gmsh.model.addPhysicalGroup(2, [34,56,78,100,122]))
    pSurfaces.append(gmsh.model.addPhysicalGroup(2, [51,73,95,117]))

    # Name the physical surfaces for applying boundary conditions
    gmsh.model.setPhysicalName(2, pSurfaces[0], 'inlet')
    gmsh.model.setPhysicalName(2, pSurfaces[1], 'outlet')
    gmsh.model.setPhysicalName(2, pSurfaces[2], 'walls')

    # Do the same for the volume
    pVol = gmsh.model.addPhysicalGroup(3, [1,2,3,4,5])
    gmsh.model.setPhysicalName(3, pVol, 'domain')

