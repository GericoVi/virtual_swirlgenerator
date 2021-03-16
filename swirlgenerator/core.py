# -------------------------------------------------------------------------------------------------------------------
#
# Main module for creating the requested swirl profile
# Also has functionality for comparing profiles
#
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
from typing import Union
import alphashape

'''
Heavy use of numpy for fast matrix and vector operations
Matplotlib for doing visualisations
'''

# Fluid parameters
GAMMA = 1.4
KIN_VISC = 1.716e-5
DENSITY = 1.225         # ISA sea level condition - for incompressible

class Vortices: 
    '''
    For storing and convenient querying of information about the vortices which have been defined for the domain
    - model - mathematical model to use when describing the vortices
    - centres - location of each vortex in the inlet
    - strengths - strength of each vortex
    - radius - core radius of each vortex
    - axialVel - uniform axial velocity of the bulk flow
    - inputObject - instance of Input object from swirlgenerator/pre.py which contains the above information,
    is this is defined, other inputs are ignored
    '''

    # Object constructor accepts lists also for convenience, then later converts to numpy arrays
    def __init__(self, model: str = None, centres: Union[list, np.ndarray] = None, strengths: Union[list, np.ndarray] = None, radius: Union[list, np.ndarray] = None, 
                        axialVel: float = None, inputObject = None):

        # Extract attributes from Input object if supplied
        if inputObject is not None:
            # Check if this is an Input object from the pre module - do import and check here so that this import isn't always needed
            import pre
            if not isinstance(inputObject, pre.Input):
                raise RuntimeError('\'inputObject\' parameter given for Vortices class initialisation is not instance of Input class from pre module')

            self.numVortices    = len(inputObject.vortStrengths)
            self.model          = inputObject.vortModel           # Vortex type - which mathematical model to use for all the vortices in the domain
            self.centres        = inputObject.vortCoords          # Vortex centre
            self.strengths      = inputObject.vortStrengths       # Vortex strength
            self.radius         = inputObject.vortRadius          # Vortex core radius - where the majority of vorticity is concentrated
            self.axialVel       = inputObject.axialVel            # Uniform axial velocity - only needed for forced swirl type
        
        # Or take attributes from individual inputs
        else:
            self.numVortices    = len(strengths)
            self.model          = model                                                                           # Vortex type - which mathematical model to use for all the vortices in the domain
            self.axialVel       = axialVel                                                                        # Uniform axial velcoity - only needed for forced swirl type
            # Make sure these are all numpy arrays not just lists
            self.centres        = (centres      if isinstance(centres,np.ndarray)   else np.array(centres))       # Vortex centre
            self.strengths      = (strengths    if isinstance(strengths,np.ndarray) else np.array(strengths))     # Vortex strength
            self.radius         = (radius       if isinstance(radius,np.ndarray)    else np.array(radius))        # Vortex core radius - where the majority of vorticity is concentrated


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
    - nodes - numpy array of the 2D coordinates of the inlet nodes (dtype=complex)
    '''

    def __init__(self, nodes: np.ndarray, compressible=False):
        # Initialise the actual flow field variables
        self.velocity   = None
        self.rho        = None
        self.pressure   = None

        # Constant density
        if not compressible:
            self.rho = DENSITY

        # Initialise domain variables
        self.boundaryCurve = None
        self._sortIdx_ = None
        self.isCircle = False

        # Swirl descriptors
        self.swirlAngle = None      # Tangential flow angle
        self.radialAngle = None     # Radial flow angle

        # Misc parameters
        self.bl_delta = None        # boundary layer thickness

        # Extract x and y coordinates of nodes as complex numbers - we're assuming that the inlet plane is parallel with the x,y plane
        self.coords = nodes[:,0] + 1j * nodes[:,1]

        # Store the z coordinate of the nodes - should be the same for all nodes because of above assumption
        self.zCoord = nodes[0,2]

        # Protection for division by zero so that warnings don't come up
        self.coords.real[self.coords.real == 0] = 1e-32
        self.coords.imag[self.coords.imag == 0] = 1e-32

        # Also store coordinates in polar axis for convenience
        radius = np.abs(self.coords)
        theta = np.arctan2(self.coords.imag, self.coords.real)
        self.coords_polar = radius + 1j * theta

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
        self.isCircle = np.all(np.abs(radius - radius[0]) < 1e-9)


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
            velComps[:,:,i] = self.makeSolidWall(vortDefs.getVortex(i), velComps[:,:,i], func)


        # Collate effects of each vortex
        vel_UV = np.sum(velComps,axis=2)

        # Add uniform axial velocity field? Or have some other equation for it
        W = np.ones(vel_UV.shape[0])*axialVel

        # Stack velocity grids into multidimensional array
        self.velocity = np.column_stack([vel_UV,W])

        # Get swirl angle
        self.getSwirl()


    def reconstructDomain(self, tangential_angles, radial_angles, axialVel, degrees=True):
        '''
        Reconstructs the velocity field at the inlet using the tangential and radial flow angles
        - tangential_angles - tangential flow angle, atan(vel_theta/vel_z), flat array with order corresponding to the nodes list
        - radial_angles - radial flow angle, atan(vel_r/vel_z), flat array with order corresponding to the nodes list
        - axialVel - mean axial velocity of the bulk flow (streamwise velocity)
        - degrees - flag to choose what angle units the inputs use, degrees by default
        '''
        
        # Convert to radians if necessary
        if degrees:
            tangential_angles = np.deg2rad(tangential_angles)
            radial_angles = np.deg2rad(radial_angles)

        # Get velocities in polar coordinate system
        vel_theta = np.tan(tangential_angles)*axialVel
        r_dot = np.tan(radial_angles)*axialVel

        # Get theta dot
        theta_dot = vel_theta/self.coords_polar.real

        # Get velocity field in Cartesian system
        u = r_dot*np.cos(self.coords_polar.imag) - self.coords_polar.real*theta_dot*np.sin(self.coords_polar.imag)
        v = r_dot*np.sin(self.coords_polar.imag) + self.coords_polar.real*theta_dot*np.cos(self.coords_polar.imag)

        # Assume a uniform streamwise velocity
        w = np.ones(u.shape[0])*axialVel

        # Stack into the velocity attribute
        self.velocity = np.column_stack([u,v,w])

        # Get swirl angle
        self.getSwirl()

    
    def makeSolidWall(self, vortData, velComp, vortexFunc):
        '''
        Models the effect of a solid wall on a vortex using the Method of Images.
        Effect of these image vortices are superimposed onto the input arrays.
        - Only implemented for circular inlets
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
            # Assuming that the origin is at the centre of the circular domain
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


    def makeBoundaryLayer(self, ref_len, kappa=0.384, B=4.17):
        '''
        Rough function for imposing a turbulent boundary layer on the flow near the walls
        Only works for circular ducts with origin at the centre
        '''

        if self.isCircle:
            # Mean velocity magnitude?
            velMag = np.linalg.norm(self.velocity, axis=1)
            u_inf = np.mean(velMag)

            # Get Reynold's number based on reference length - distance the boundary layer has been developing for
            Rex = u_inf * ref_len / KIN_VISC
            #print(f'Re = {Rex}')

            # Crudely estimate the BL thickness and skin friction coefficient - from BL theory
            self.bl_delta = ref_len * 0.38 * Rex**(-0.2)
            cf = 0.059 * Rex**(-0.2)

            # Get friction velocity from skin friction coefficient
            u_tau = np.sqrt(0.5*cf*u_inf*u_inf)

            # Get distance of each node from the wall
            y = np.max(self.coords_polar.real) - self.coords_polar.real

            # Get yplus
            yplus = u_tau * y / KIN_VISC

            # Calc non-dimensional parameters
            R = u_tau * self.bl_delta / KIN_VISC
            u_inf_plus = u_inf / u_tau

            # Eta in terms of y plus
            eta = yplus / R

            # Wake parameter
            Pi = 0.5 * kappa * (u_inf_plus - (1/kappa) * np.log(R) - B)
            
            # Set inner region profile with Reichardt formulation of law of the wall
            temp = yplus
            temp[temp > R] = R
            uplus1 = (1/kappa) * np.log(1 + kappa * temp) + 7.1 * (1 - np.exp(-temp/12))
            
            # Set the outer region's velocity defect law profile with Finlay formulation
            temp = eta
            temp[eta > 1] = 1
            uplus2 = (1/kappa) * Pi * (((1/Pi) + 6) * temp**2 - ((1/Pi) + 4) * temp**3)

            # Get full non-dimensional output profile
            uplus = uplus1 + uplus2
            # Redimensionalise - this is the adjusted velocity magnitude
            u = uplus * u_tau

            # Distribute the velocity magnitude into the components at the same proportions as before the BL correction
            flowDirection = self.velocity / np.column_stack((velMag,velMag,velMag))

            self.velocity = np.reshape(u,[u.size,1]) * flowDirection

            return uplus

        
        else:
            # Warn that boundary layer from no-slip in non-circular inlets will not be modelled
            import warnings

            warnings.warn("No-slip condition boundary layer for non-circular inlets have not been implemented")

    
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

        # Displacement of each node from vortex centres
        disp = self.coords - vortO
        # Get radial coordinate from vortex centre - theta coordinate is the same (no rotations happening)
        r = np.abs(disp)

        # Normalise radius for straightforward angle calculation and set cells outside vortex size to 0
        rNorm = r/vortData[2]
        # Add some tolerance to the equality to smooth out circle because discretised as nodes
        rNorm[np.nan_to_num(rNorm) > 1] = 0

        # Get swirl angle distribution
        swirlAngles = maxSwirlAngle*rNorm

        # Get tangential velocity at each cell
        tangentVel = vortData[3]*np.tan(swirlAngles)

        # Get theta_dot at each cell
        theta_dot = tangentVel/r

        # Get velocity vector components, in-plane cartesian (assume no radial velocity)
        velComp[:,0] = -r*theta_dot*np.sin(self.coords_polar.imag)
        velComp[:,1] =  r*theta_dot*np.cos(self.coords_polar.imag)

        return velComp


    def checkBoundaries(self, plot=False):
        '''
        For verifying physically correct boundary conditions.
        ie checking if there is any flow across the solid boundaries and no slip condition
        - Outputs the average specific flux out per unit perimiter of the boundary (units/s)
        - Currently only checks for no-flux condition
        '''

        # Get planar velocity vectors
        vels   = self.velocity[:,0:2]
        # Get the velocities of the nodes at the boundary - without duplicate starting point
        sortedVels = vels[self._sortIdx_[:-1],:]

        # Get boundary nodes without duplicate starting point
        boundaryNodes = self.boundaryCurve[:-1]

        # Calculate vectors which are parallel to the boundary curve
        parallelVect = np.empty(sortedVels.shape)
        for i in range(boundaryNodes.size):    
            if (i == 0):
                parallel = boundaryNodes[1]-boundaryNodes[-1]
            elif (i == boundaryNodes.size-1):
                parallel = boundaryNodes[0]-boundaryNodes[i-1]
            else:
                parallel = boundaryNodes[i+1]-boundaryNodes[i-1]

            parallelVect[i,:] = [parallel.real, parallel.imag]

        # Calculate vectors which are perpendicular to the boundary curve
        perpendicularVect = np.column_stack([-parallelVect[:,1], parallelVect[:,0]])

        # Now calculate the component of the velocity at each point, perpendicular to the boundary curve
        velOut = np.array([perp * np.dot(vel, perp)/np.dot(perp, perp) for vel,perp in zip(sortedVels,perpendicularVect)])

        # Integrate to get total flux through boundary
        fluxOut = np.sum([np.dot(vel,vel)*np.dot(parallel,parallel)/2 for vel, parallel in zip(velOut,parallelVect)])

        # Get boundary length - need duplicate start point here to use diff function
        perimeter = np.sum(np.abs(np.diff(self.boundaryCurve, axis=0)))

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.quiver(boundaryNodes.real, boundaryNodes.imag, velOut[:,0],velOut[:,1],color='red')
            plt.quiver(boundaryNodes.real, boundaryNodes.imag, parallelVect[:,0],parallelVect[:,1],color='blue')
            #plt.show()
            
        # Return avg flux out per unit boundary length
        return fluxOut/perimeter

    
    def getSwirl(self):
        '''
        Calculate swirl (tangential flow) angles of velocity field and radial flow angles
        '''

        # Get theta_dot - rate of change of theta angle (rad/s)
        theta_dot = (self.coords.real*self.velocity[:,1] - self.velocity[:,0]*self.coords.imag) / self.coords_polar.real**2

        # Get tangential velocity
        velTheta = self.coords_polar.real*theta_dot

        # Get swirl angle - as defined in literature
        swirlAngle = np.arctan(velTheta/self.velocity[:,2])
        # Convert to degrees
        self.swirlAngle = np.rad2deg(swirlAngle)

        # Get radial velocity
        velRad = (self.coords.real*self.velocity[:,0] + self.coords.imag*self.velocity[:,1]) / self.coords_polar.real

        # Get radial flow angle
        radialAngle = np.arctan(velRad/self.velocity[:,2])
        # Convert to degrees
        self.radialAngle = np.rad2deg(radialAngle)

    
    def getError(self, desiredSwirl):
        '''
        Calculate Root Mean Square error between this flow field's swirl angle profile and a given one
        - desiredSwirl - swirl angle data of each point in the flow field to be compared
        '''

        RMSE = np.sqrt((1/np.size(self.swirlAngle))*np.sum((self.swirlAngle-desiredSwirl)**2))

        return RMSE

    
    def save(self, outputFile, saveCoords=False):
        '''
        Wrapper function for saving the flow field in a format which can be loaded by core.load() later
        - outputFile - without file extension
        - so calling script does not need to import numpy just for this
        '''

        if not saveCoords:
            np.savez(outputFile, velocity=self.velocity, rho=self.rho, pressure=self.pressure, swirl=self.swirlAngle, radialangle=self.radialAngle)
        else:
            coords = np.column_stack([self.coords.real, self.coords.imag, self.zCoord])
            np.savez(outputFile, velocity=self.velocity, rho=self.rho, pressure=self.pressure, swirl=self.swirlAngle, radialangle=self.radialAngle, nodes=coords)


    def load(self, file, loadCoords=False):
        '''
        Unpacks zipped archive file created by save() and returns the numpy arrays in the familiar format
        '''

        # Extract file into an npz file
        npzfile = np.load(file, allow_pickle=True)

        # Check if correct format
        if ('velocity' in npzfile and 'rho' in npzfile and 'pressure' in npzfile and 'swirl' in npzfile):
            self.velocity    = npzfile['velocity']
            self.rho         = npzfile['rho']
            self.pressure    = npzfile['pressure']
            self.swirlAngle  = npzfile['swirl']
        else:
            raise RuntimeError('File format/contents invalid - make sure this file was created by swirlGenerator.saveFigsToPdf')

        # Check if the nodes were saved and needs to be loaded
        if loadCoords:
            if 'nodes' in npzfile:
                self.coords = npzfile['nodes'][:,0] + 1j * npzfile['nodes'][:,1]
                self.zCoord = npzfile['nodes'][:,2]
            else:
                raise RuntimeError('loadCoords requested but specified file does not contain node coordinates')

        # Older saved data does not have this
        if ('radialangle' in npzfile):
            self.radialAngle = npzfile['radialangle']

