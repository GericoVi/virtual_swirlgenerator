# -------------------------------------------------------------------------------------------------------------------
#
# Main module for creating the requested swirl profile
# Also has functionality for comparing profiles
#
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
from typing import Union
import pre

'''
Heavy use of numpy for fast matrix and vector operations
Matplotlib for doing visualisations
'''

# Fluid parameters
GAMMA = 1.4
kin_visc = 1.81e-5
density = 1.225         # ISA sea level condition - for incompressible

class Vortices: 
    '''
    For storing and convenient querying of information about the vortices which have been defined for the domain
    - Positive vortex strength is defined as anti-clockwise rotation
    '''

    # Object constructor accepts lists also for convenience, then later converts to numpy arrays
    def __init__(self, model: int, centres: Union[list, np.ndarray], strengths: Union[list, np.ndarray],
                       radius: Union[list, np.ndarray], axialVel: float):

        self.numVortices    = len(strengths)

        self.model          = model         # Vortex type - which mathematical model to use for all the vortices in the domain

        # Make sure these are all numpy arrays not just lists
        self.centres        = (centres      if isinstance(centres,np.ndarray)   else np.array(centres))       # Vortex centre
        self.strengths      = (strengths    if isinstance(strengths,np.ndarray) else np.array(strengths))     # Vortex strength
        self.radius         = (radius       if isinstance(radius,np.ndarray)    else np.array(radius))        # Vortex core radius - where the majority of vorticity is concentrated
        self.axialVel       = axialVel                                                                        # Uniform axial velcoity - only needed for forced swirl type

    # Return data for requested vortex as tuple
    def getVortex(self,vortexIndex):
        # Check if at end of vortex list
        if (vortexIndex >= self.numVortices):
            raise IndexError(f"Index {self.vortNum} is out of bounds of vortex list with size {self.numVortices}")
        else:
            # Output tuple format 
            data = (self.centres[vortexIndex], self.strengths[vortexIndex], self.radius[vortexIndex], self.axialVel)

        return data 


class FlowField:
    '''
    Class containing data and functions relevant to the flow field
    - Initialised with an Input object
    '''

    def __init__(self, InputData: pre.Input):
        # Get flow field descretisation descriptions from input object
        self.shape = InputData.shape
        self.radius = InputData.radius

        if InputData.xSide is not None:
            self.sideLengths = np.array([InputData.xSide, InputData.ySide])
        else:
            # If circular domain, still need side lengths for defining the grid
            self.sideLengths = np.array([InputData.radius*2, InputData.radius*2])
        
        if self.shape == 'rect':
            self.numCells = np.array([InputData.xNumCells, InputData.yNumCells])
        else:
            self.numCells = np.array([InputData.radialNumCells, InputData.curveNumCells])

        # Initialise the actual flow field variables
        self.velocity   = None
        self.rho        = None
        self.pressure   = None

        # Initialise grid and domain variables
        self.coords = None
        self.axis = None
        self.domainMask = None
        self.boundaryMask = None
        self.boundaryCurve = None
        self._sortIdx_ = None

        # Some comparison and metrics
        self.swirlAngle = None

        # Side lengths of each cell - using np.divide here also serves to automatically convert python lists into numpy arrays
        if self.shape == 'rect':
            # For rectangular domain, simple calculation
            self.cellSides = np.divide(self.sideLengths,self.numCells)
        elif self.shape == 'circle':
            # For circular domain, diameter (equal to grid side length) is used
            self.cellSides = np.divide(self.radius*2,self.numCells)
        else:
            raise NotImplementedError('Invalid domain shape \'{self.shape}\'')

        # If nodes taken from mesh file, use these. If not, generate some based on user input
        if InputData.coords is not None:
            self.coords = InputData.coords
        else:
            # Set coords and axis attributes
            self.makeGrid()

            # Set domainMask and boundaryMask attributes
            self.setDomain()

            # Set cells outside domain to nan so that the flow field there is not unnecessarily calculated
            self.coords[np.invert(self.domainMask)] = np.nan

    
    def makeGrid(self):
        '''
        Make meshgrids to store coordinate system
        - As a result, all variable fields will be meshgrids also, good performance since using numpy matrix operations
        - Also stores x and y axis ticks
        - Stores coords of mesh nodes rather than cell centres
        '''

        # Create coordinate system from mesh info - domain is centered at 0, I think makes for more intuitive definition of vortex positions
        x = np.linspace(-self.sideLengths[0]/2, self.sideLengths[0]/2, self.numCells[0]+1)
        y = np.linspace(-self.sideLengths[1]/2, self.sideLengths[1]/2, self.numCells[1]+1)     

        # Protection for division by zero later - better solution than this?
        x[x == 0] = 1e-32
        y[y == 0] = 1e-32

        # Use meshgrid function to get the coordinates of all points
        X, Y = np.meshgrid(x,y)

        # Then flatten and store as a list of complex numbers
        self.coords = X.flatten() + 1j * Y.flatten()

        # Stack axis ticks
        self.axis = np.vstack([x,y])


    def setDomain(self):
        ''' 
        Create a mask to specify the domain shape and borders within the meshgrid.
        Sets object attributes: domainMask, boundaryMask, boundaryCurve, _sortIdx_
        - domainMask is true when cell is within the boundary
        - boundaryMask is true when cell is at the boundary
        - boundaryCurve stores the points (in order) which make up the boundary, as complex numbers
        - _sortIdx_ is an internal attribute, index order to sort boundary cells
        '''

        if self.shape == 'circle':
            # Radius of each cell from origin
            radius = np.abs(self.coords)

            # Get domainMask using inequality - add buffer so that circular domain edges touch grid edges, since working with nodes rather than cell centres
            self.domainMask = radius < self.radius + self.cellSides[0]/2

            # Get boundary using equality with a tolerance since discrete space
            self.boundaryMask = abs(radius - self.radius) < self.cellSides[0]/2

        elif self.shape == 'rect':
            # All cells are within boundary when rectangular domain shape
            self.domainMask = np.ones(self.coords.shape, dtype=bool)

            # Boundary cells are simply those at the edges
            self.boundaryMask = np.zeros(self.domainMask.shape, dtype=bool)
            self.boundaryMask[abs(self.coords.real) == self.sideLengths[0]/2] = True    # Right and left wall
            self.boundaryMask[abs(self.coords.imag) == self.sideLengths[1]/2] = True    # Top and bottom wall

        else:
            raise NotImplementedError(f'Domain shape \'{self.shape}\' not valid')

        boundaryPoints = self.coords[self.boundaryMask]
        self._sortIdx_ = np.argsort(np.angle(boundaryPoints))
        self.boundaryCurve = boundaryPoints[self._sortIdx_]

        # Show boundary for debugging
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.domainMask.reshape(self.numCells+1))
        # plt.figure()
        # plt.imshow(self.boundaryMask.reshape(self.numCells+1))
        # plt.show()


    def computeDomain(self, vortDefs: Vortices, axialVel, density = None):
        '''
        Generic multiple vortices function.
        Calculates the velocity field by superimposing the effect of the individual vortices.
        - vortDefs - Vortices object
        - axialVel - uniform axial velocity to be applied
        - density - if defined, assume that flow is incompressible
        - Solid boundaries are modelled using the method of images
        '''

        # Intialise 3D array to store the velocity effect of each vortex - ie a stack of multiple lists of 2D vectors
        velComps = np.zeros([self.coords.size, 2, vortDefs.strengths.shape[0]])

        # Dictionary mapping for functions - will be faster than multiple if/else statements - also more readable code
        vortexType = {'iso':self.__isoVortex__, 'lo':self.__loVortex__, 'solid':self.__solidVortex__}

        # Loop through given vortices and calculate their effect on each cell of the grid
        for i in range(vortDefs.strengths.shape[0]):
            # Get function for this vortex type
            func = vortexType.get(vortDefs.model)
            # Call vortex function to fill component arrays - with data for a single vortex
            velComps[:,:,i] = func(vortDefs.getVortex(i))

            # Calculate the effect of solid walls on this vortex using mirror image vortices
            velComps[:,:,i] = self.__boundary__(vortDefs.getVortex(i), velComps[:,:,i], func)


        # Collate effects of each vortex
        vel_UV = np.sum(velComps,axis=2)

        # Add uniform axial velocity field? Or have some other equation for it
        W = np.ones(vel_UV.shape[0])*axialVel

        # Stack velocity grids into multidimensional array
        self.velocity = np.column_stack([vel_UV,W])

        # Get swirl angle
        self.getSwirl()

    
    def __boundary__(self, vortData, velComp, vortexFunc):
        '''
        Models the effect of a solid wall on a vortex using the Method of Images.
        Effect of these image vortices are superimposed onto the input arrays.
        - Internal function, should not be used outside core.py
        - WIP ---- RECTANGULAR BOUNDARY DOES NOT CURRENTLY WORK CORRECTLY
        - vortData - tuple produced by getVortex() function of Vortices class
        - velComp - velocity field outputted by a vortex function
        - vortexFunc - pointer to the correct vortex function depending on chosen model
        '''

        if self.shape == 'rect':
            # Get distance of vortex from walls - defined starting with bottom wall, going clockwise
            vortXc, vortYc = vortData[0]
            boundaryDist = [-self.sideLengths[1]/2-vortYc, -self.sideLengths[0]/2-vortXc, self.sideLengths[1]/2-vortYc, self.sideLengths[0]/2-vortXc]
            boundaryDist = list(map(abs,boundaryDist))  # Magnitudes
       
            # Place image vortices outside the domain - such that the bounday conditions are met while keeping the total circulation of the unbounded domain equalt to 0
            imageVortO = [[vortXc, vortYc-(2*boundaryDist[0])], 
                          [vortXc-(2*boundaryDist[1]), vortYc], 
                          [vortXc-(2*boundaryDist[1]), vortYc-(2*boundaryDist[0])],
                          [vortXc, vortYc+(2*boundaryDist[1])],
                          [vortXc, vortYc+(3*boundaryDist[1])],
                          [vortXc+(2*boundaryDist[3]), vortYc],
                          [vortXc+(3*boundaryDist[3]), vortYc]]
            imageVortS = [-vortData[1],-vortData[1],vortData[1],-vortData[1],vortData[1],-vortData[1],vortData[1]]

            for i in range(len(imageVortS)):
                # Create new array for this image vortex to be passed on to the appropriate vortex model function
                imageVortData = list(vortData)
                imageVortData[0] = imageVortO[i]
                imageVortData[1] = imageVortS[i]

                #print(f'image vortex @ {imageVortData[0]}, with strength {imageVortData[1]}')

                # Get effect of image vortex on velocity field
                velBoundary = vortexFunc(tuple(imageVortData))

                # Superimpose effect
                velComp += velBoundary


        elif self.shape == 'circle':
            # Protection for division by zero
            vortData[0][vortData[0] == 0] = 1e-32   

            # Vortex of opposite strength
            imageVortS = -vortData[1]
            # At the inverse point - according to circle theorem
            imageVortO = (self.radius**2/(np.linalg.norm(vortData[0]))**2)*vortData[0]

            # Creating new vortex data list
            imageVortData = list(vortData)
            imageVortData[0] = imageVortO
            imageVortData[1] = imageVortS

            #print(f'image vortex @ {imageVortData[0]}, with strength {imageVortData[1]}')

            # Get effect of image vortex on velocity field
            velBoundary = vortexFunc(tuple(imageVortData))

            # Superimpose effect
            velComp += velBoundary
            
        else:
            raise NotImplementedError('Inlet shape not valid')

        return velComp

    
    def __isoVortex__(self, vortData):
        '''
        Function for outputting the effect of a simple isentropic vortex on the domain
        - vortData - tuple produced by getVortex() function of Vortices class
        - Internal function, should not be used outside core.py
        '''

        # Initialise velocity array - list of 2D vectors
        velComp = np.zeros([self.coords.size,2])

        # Extract vortex centre coordinates into a complex number
        vortO = vortData[0][0] + 1j * vortData[0][1]

        # Displacement of each node from vortex centres
        disp = self.coords - vortO
        # Get radius of each cell from centre of this vortex
        r = np.abs(disp)

        # Velocity components due to this vortex
        velComp[:,0] = (vortData[1]/(2*np.pi)) * np.exp(0.5*(1-r**2)) * disp.imag
        velComp[:,1] = (vortData[1]/(2*np.pi)) * np.exp(0.5*(1-r**2)) * disp.real

        return velComp

    
    def __loVortex__(self, vortData):
        '''
        Function for outputting the effect of a Lamb-Oseen vortex
        - Internal function, should not be used outside core.py
        - vortData - tuple produced by getVortex() function of Vortices class
        - using equations given by Brandt (2009)
        '''

        # Initialise velocity array - list of 2D vectors
        velComp = np.zeros([self.coords.size,2])

        # Extract vortex centre coordinates into a complex number
        vortO = vortData[0][0] + 1j * vortData[0][1]

        # Extract other individual variables from the vortData tuple
        strength = vortData[1]
        a0 = vortData[2]

        # Displacement of each node from vortex centres
        disp = self.coords - vortO
        # Get radius squared of each cell from centre of this vortex
        rr = np.abs(disp)**2

        # Get omega, the peak magnitude of vorticity (positive counterclockwise)
        omega = -strength/(np.pi * a0**2)

        # Velocity components due to this vortex
        velComp[:,0] = 0.5  * (a0**2 * omega * disp.imag / rr) * (1 - np.exp(-rr/a0**2))
        velComp[:,1] = -0.5 * (a0**2 * omega * disp.real / rr) * (1 - np.exp(-rr/a0**2))

        return velComp

    
    def __solidVortex__(self, vortData):
        '''
        Function for outputting the effect of a forced vortex
        - Internal function, should not be used outside core.py
        - vortData - tuple produced by getNextVortex() function of Vortices class
        - linear increase in swirl angle from center to outer edge
        - solid/forced vortex - not realistic; ie instantaneously created vortex, no effect on cells outside it's radius
        '''

        # Initialise velocity array - list of 2D vectors
        velComp = np.zeros([self.coords.size,2])

        # Extract vortex centre coordinates into a complex number
        vortO = vortData[0][0] + 1j * vortData[0][1]

        # Get swirl angle and convert it to radians
        maxSwirlAngle = np.deg2rad(np.abs(vortData[1]))

        # Get vortex rotation information from sign of maximum angle specified
        anitclockwise = (True if vortData[1] > 0 else False)

        # Displacement of each node from vortex centres
        disp = self.coords - vortO
        # Get axial coordinates
        r = np.abs(disp)
        theta = np.arctan(self.coords.imag/self.coords.real)

        # Normalise radius for straightforward angle calculation and set cells outside vortex size to 0
        rNorm = r/vortData[2]
        # Add some tolerance to the equality to smooth out circle because discretised as nodes
        rNorm[np.nan_to_num(rNorm) > 1] = 0

        # Get swirl angle distribution
        swirlAngles = maxSwirlAngle*rNorm

        # Transform so swirl is coherent (either clockwise or anticlockwise) - without this, the swirl profile produced is mirrored about the y axis
        swirlAngles[(np.nan_to_num(self.coords.real * anitclockwise) < 0)] = swirlAngles[(np.nan_to_num(self.coords.real * anitclockwise) < 0)] * -1

        # Get tangential velocity at each cell
        tangentVel = vortData[3]*np.tan(swirlAngles)

        # Get theta_dot at each cell
        theta_dot = tangentVel/r

        # Get velocity vector components, in-plane cartesian (assume no radial velocity)
        velComp[:,0] = -r*theta_dot*np.sin(theta)
        velComp[:,1] =  r*theta_dot*np.cos(theta)

        return velComp


    def checkBoundaries(self, tolerance=1e-6):
        '''
        For verifying physically correct boundary conditions.
        ie checking if there is any flow across the solid boundaries and no slip condition
        - tolerance - maximum flux out of boundary considered negligible
        - Currently only checks for no-flux condition
        '''

        boundary_ok = True

        # Get planar velocity vectors as complex numbers
        vels   = self.velocity[:,0] + 1j * self.velocity[:,1]
        # Get only the velocities of the nodes at the boundary
        vels = vels[self.boundaryMask]

        # Sort data based on increasing phi polar coordinate
        sortedVels = vels[self._sortIdx_]

        # Calculate vectors which are parallel to the boundary curve
        parallelVect = np.empty(self.boundaryCurve.size, dtype=complex)
        for i in range(self.boundaryCurve.size):
            if (i != 0 and i != self.boundaryCurve.size-1):
                parallelVect[i] = self.boundaryCurve[i+1]-self.boundaryCurve[i-1]
            elif (i == 0):
                parallelVect[0] = self.boundaryCurve[1]-self.boundaryCurve[-1]
            else:
                parallelVect[i] = self.boundaryCurve[0]-self.boundaryCurve[i-1]

        # Calculate vectors which are perpendicular to the boundary curve
        perpendicularVect = np.empty(self.boundaryCurve.size, dtype=complex)
        for i, vect in enumerate(parallelVect):
            perpendicularVect[i] = vect.imag - 1j*vect.real

        # Now calculate the component of the velocity at each point, perpendicular to the boundary curve
        velOut = np.abs(sortedVels)*np.dot(sortedVels,perpendicularVect)

        # Integrate to get total flux through boundary
        fluxOut = np.sum(np.abs(velOut)*np.abs(parallelVect)/2)

        # Check if this flux is considered negligible or not
        if fluxOut > tolerance:
            boundary_ok = False
            print(f'Flux out of boundary: {fluxOut} units/sec')

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.quiver(self.boundaryCurve.real, self.boundaryCurve.imag, parallelVect.real, parallelVect.imag,units='dots', width=2,headwidth=5,headlength=5,headaxislength=2.5,color='blue')
        # plt.quiver(self.boundaryCurve.real, self.boundaryCurve.imag, perpendicularVect.real, perpendicularVect.imag,units='dots', width=2,headwidth=5,headlength=5,headaxislength=2.5,color='red')
        # plt.scatter(self.boundaryCurve.real, self.boundaryCurve.imag)
        # plt.show()
            
        return boundary_ok

    
    def getSwirl(self):
        '''
        Calculate swirl angles of velocity field
        '''

        # Get radius
        r = np.abs(self.coords)
        # Get theta_dot - rate of chane of theta angle (rad/s)
        theta_dot = (self.coords.real*self.velocity[:,1] - self.velocity[:,0]*self.coords.imag) / r

        # Get tangential velocity
        velTheta = r*theta_dot

        # Get swirl angle - as defined in literature
        swirlAngle = np.arctan(velTheta/self.velocity[:,2])
        # Convert to degrees
        self.swirlAngle = np.rad2deg(swirlAngle)

    
    def getError(self, desiredSwirl):
        '''
        Calculate Root Mean Square error between this flow field's swirl angle profile and a given one
        - desiredSwirl - swirl angle data of each point in the flow field to be compared
        '''

        RMSE = np.sqrt((1/np.size(self.swirlAngle))*np.sum((self.swirlAngle-desiredSwirl)**2))

        return RMSE

    
    def save(self, outputFile):
        '''
        Wrapper function for saving the flow field in a format which can be loaded by core.load() later
        - so calling script does not need to import numpy just for this
        '''

        np.savez(outputFile, velocity=self.velocity, rho=self.rho, pressure=self.pressure, swirl=self.swirlAngle)


    def load(self, file):
        '''
        Unpacks zipped archive file created by save() and returns the numpy arrays in the familiar format
        '''

        # Extract file into an npz file
        npzfile = np.load(file)

        # Check if correct format
        if ('velocity' in npzfile and 'rho' in npzfile and 'pressure' in npzfile and 'swirl' in npzfile):
            self.velocity    = npzfile['velocity']
            self.rho         = npzfile['rho']
            self.pressure    = npzfile['pressure']
            self.swirlAngle  = npzfile['swirl']

        else:
            raise RuntimeError('File format/contents invalid - make sure this file was created by swirlGenerator.saveFigsToPdf')

    
    def copy(self):
        '''
        Utility function for copying this flow field into another separate object
        '''

        # Create new object
        newField = FlowField(self.sideLengths,self.numCells)

        # Copy all data so far
        newField.velocity   = self.velocity
        newField.rho        = self.rho
        newField.pressure   = self.pressure
        newField.swirlAngle = self.swirlAngle

        return newField
