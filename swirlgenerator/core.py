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
        # Initialise the actual flow field variables
        self.velocity   = None
        self.rho        = None
        self.pressure   = None

        # Initialise domain variables
        self.boundaryCurve = None
        self._sortIdx_ = None
        self.isCircle = False

        # Some comparison and metrics
        self.swirlAngle = None

        # Extract node coordinates from the inlet of the mesh file
        self.coords = InputData.getNodes()

        # Set boundaryCurve attribute - ie the nodes which make up the domain boundary
        self.__getBoundary__()

        self.__checkIfCircle__()


    def __getBoundary__(self):
        ''' 
        Sets object attributes: boundaryCurve, _sortIdx_
        - Internal function, should not be used in isolation outside core.py
        - Uses the alpha shape algorithm to get the boundary nodes - from alphashape library
        - boundaryCurve stores the points (in order) which make up the boundary, as complex numbers
        - _sortIdx_ is an internal attribute, index order to sort boundary cells, for connecting them correctly
        '''

        # Do alphashape import here, not needed anywhere else
        import alphashape

        # Get points into array format
        points = np.column_stack([self.coords.real, self.coords.imag])

        # Use alpha shape to get the bounding curve 
        # an alpha value of 0.1 performs well with nodes arranged in a circle or rectangle
        # Basically just a small value over 0 seems to generalise well to simple shapes without ignoring the points on the sides of rectangle - since alpha=0 returns the convex hull
        hull = alphashape.alphashape(points,alpha=0.1)

        # Extract the actual coordinates into a numpy array
        hull_pts = hull.boundary.coords.xy
        hull_pts = np.array([hull_pts[0],hull_pts[1]])

        # Put into complex format and assign into class attribute
        self.boundaryCurve = hull_pts[0] + 1j * hull_pts[1]

        # Get indices of these points within the full coords array
        self._sortIdx_ = [np.where(self.coords == point)[0][0] for point in self.boundaryCurve]


    def __checkIfCircle__(self):
        '''
        Method to check if the inlet face extracted from the mesh is circular
        - Sets attribute isCircle
        - Needed since solid boundary effect is only available for circular inlets using the method of images & circle theorem
        '''

        # Get radius of all points in the boundary curve
        radius = np.abs(self.boundaryCurve)
        
        # Check if all points in the boundary are equally spaced from the centre - with a tolerance for numerical inaccuracy
        self.isCircle = np.all(np.abs(radius - radius[0]) < 1e-15)


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
        - WIP ---- Only implemented for circular inlets
        - vortData - tuple produced by getVortex() function of Vortices class
        - velComp - velocity field outputted by a vortex function
        - vortexFunc - pointer to the correct vortex function depending on chosen model
        '''

        if self.isCircle:
            # Protection for division by zero
            vortData[0][vortData[0] == 0] = 1e-32   

            # Get the radius of the inlet - maximum radius of nodes at boundary (should all be equal but just in case)
            radius = np.amax(np.abs(self.boundaryCurve))

            # Vortex of opposite strength
            imageVortS = -vortData[1]
            # At the inverse point - according to circle theorem
            imageVortO = (radius**2/(np.linalg.norm(vortData[0]))**2)*vortData[0]

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
            # Warn that vortices in non-circular inlets will not interact with the boundary
            import warnings

            warnings.warn("Effect of solid boundaries for non-circular inlets have not been implemented")
            
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
        # Get the velocities of the nodes at the boundary
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
        velOut = np.array([vel*np.vdot(vel,perp) for vel,perp in zip(sortedVels,perpendicularVect)])

        # Integrate to get total flux through boundary
        fluxOut = np.sum(np.abs(velOut)*np.abs(parallelVect)/2)

        # Check if this flux is considered negligible or not
        if fluxOut > tolerance:
            boundary_ok = False
            print(f'Flux out of boundary: {fluxOut} units/sec')


        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.quiver(self.boundaryCurve.real, self.boundaryCurve.imag, velOut.real,velOut.imag,color='red')
        # plt.quiver(self.boundaryCurve.real, self.boundaryCurve.imag, sortedVels.real,sortedVels.imag,color='blue')
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

