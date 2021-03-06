# -------------------------------------------------------------------------------------------------------------------
#
# Module for plotting the variables contained in the flow field object
#
# -------------------------------------------------------------------------------------------------------------------

import core as sg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapz

class Plots:
    '''
    For handling the creation of figures for the flowfield data
    - flowfield - FlowField object which contains the data
    - plotDensity - the number of points to plot along the x and y axis. If any is None, the functions will attempt to mimic the original's data density in that direction
    - Interpolates the irregularly spaced data from flowfield to a regular grid which can then be plotted
    '''

    def __init__(self, flowfield: sg.FlowField, plotDensity=[None,None]):
        '''
        Plots initialisation needs to create the variables for using griddata within the other functions
        '''
        # Extract coordinates
        self.xy = np.vstack([flowfield.coords.real, flowfield.coords.imag])

        # Store variables for plotting
        self.vel            = flowfield.velocity
        self.thermos        = [flowfield.rho, flowfield.pressure]
        self.swirlAngle     = flowfield.swirlAngle
        self.radialAngle    = flowfield.radialAngle
        self.boundary       = np.vstack([flowfield.boundaryCurve.real, flowfield.boundaryCurve.imag])
        self.bl_delta       = flowfield.bl_delta
        self.radius         = np.max(flowfield.coords_polar.real)

        # Get regularly spaced axis for plotting
        self.xy_i = Plots.makeRegularAxis(self.xy, plotDensity)


    def plotAll(self, pdfName=None, swirlAxisRange=[None,None], swirlAxisNTicks=None):
        '''
        Utility for showing and saving all plots
        - pdfName - filename to save the plots to, should include extension
        - swirlAxisRange - specify the max and min values for the colormap of the swirl angle contour plot
        - swirlAxisNTicks - specify the number of ticks to show on the colorbar for the swirl angle contour plot
        '''

        self.plotVelocity()

        #self.plotThermos()

        self.plotSwirl(axisRange=swirlAxisRange, numTicks=swirlAxisNTicks)

        # If saving, don't show plots
        if (pdfName != None):
            Plots.saveFigsToPdf(pdfName)
        else:
            plt.show()


    def plotVelocity(self, arrowDensity=50, quiver=True, streamlines=True, axial=True, border=True, bl_line=False):
        '''
        Create plots for the swirling velocity profile as a quiver plot and a streamlines plot
        - arrowDensity - Parameter which controls the sub sampling of the velocity field for the quiver plot
        '''

        # Sub sample the velocity profile so that the quiver plot is readable
        dim_max = np.max(self.xy,axis=1)
        dim_min = np.min(self.xy,axis=1)
        reduced_x = np.linspace(dim_min[0],dim_max[0],arrowDensity)
        reduced_y = np.linspace(dim_min[1],dim_max[1],arrowDensity)

        if quiver:
            u = griddata((self.xy[0],self.xy[1]), self.vel[:,0], (reduced_x[None,:],reduced_y[:,None]), method='linear')
            v = griddata((self.xy[0],self.xy[1]), self.vel[:,1], (reduced_x[None,:],reduced_y[:,None]), method='linear')

            # Make quiver plot
            plt.figure()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Quiver")
            plt.quiver(reduced_x, reduced_y, u, v, units='dots', width=2,headwidth=5,headlength=5,headaxislength=2.5)
            plt.axis('off')
            if border:
                # Draw boundary
                plt.plot(self.boundary[0], self.boundary[1],'k-')

        if streamlines:
            # Interpolate data to a regularly spaced grid
            u = griddata((self.xy[0],self.xy[1]), self.vel[:,0], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='linear')
            v = griddata((self.xy[0],self.xy[1]), self.vel[:,1], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='linear')

            # Make streamlines plot
            plt.figure()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Streamlines")
            plt.streamplot(self.xy_i[0], self.xy_i[1], u, v, density=2)            # streamplot uses vector axis for xy instead of meshgrid for some reason?
            plt.axis('off')
            if border:
                # Draw boundary
                plt.plot(self.boundary[0], self.boundary[1],'k-')

        if axial:
            if border:
                boundary = self.boundary
            else:
                boundary = None

            # Make contour plot of streamwise velocity
            axial_fig = Plots.makeContourPlot(self.vel[:,2], self.xy, 11, title='Axial velocity', regularPoints=self.xy_i, boundaryPoints=boundary, tightRange=True)

            # Add line for boundary layer
            if bl_line:
                plt.figure(axial_fig)
                
                R, Theta = np.meshgrid([self.radius-self.bl_delta], np.deg2rad(np.linspace(-180,180,360,endpoint=False)), indexing='ij')
                R, Theta = R.flatten(), Theta.flatten()
                x, y     = R*np.cos(Theta) , R*np.sin(Theta)

                plt.plot(x, y, 'k-.')


    def plotThermos(self):
        '''
        Create contour plots for density and pressure field
        '''

        rho = griddata((self.xy[0],self.xy[1]), self.thermos[0], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='linear')
        pressure = griddata((self.xy[0],self.xy[1]), self.thermos[1], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='linear')

        plt.figure()
        plt.title('Density')
        plt.contourf(self.xy_i[0],self.xy_i[1],rho,100,cmap='jet')
        plt.colorbar()

        plt.figure()
        plt.title('Pressure')
        plt.contourf(self.xy_i[0],self.xy_i[1],pressure,100,cmap='jet')
        plt.colorbar()


    def plotSwirl(self, axisRange=[None,None], numTicks = None, border=True):
        '''
        Create contour plot for swirl (tangential flow) angle and radial flow angle
        - axisRange - specify the max and min values for the colormap
        - numTicks - specify the number of ticks to show on the colorbar
        '''

        numTicks = (numTicks if numTicks is not None else 11)

        if len(axisRange) > 2:
            axisRange1 = axisRange[0:2]
            axisRange2 = axisRange[2:]
        else:
            axisRange1 = axisRange
            axisRange2 = [None,None]

        if border:
            boundary = self.boundary
        else:
            boundary = None

        # Make tangential flow angle contour plot
        Plots.makeContourPlot(self.swirlAngle, self.xy, numTicks, title='Tangential flow (Swirl) angle', regularPoints=self.xy_i, axisRange=axisRange1, boundaryPoints=boundary)

        # Make radial flow angle contour plot
        Plots.makeContourPlot(self.radialAngle, self.xy, numTicks, title='Radial flow angle', regularPoints=self.xy_i, axisRange=axisRange2, boundaryPoints=boundary)


    def plotInletNodes(self, show=True):
        '''
        Utility function for showing the position of the nodes at the inlet mesh
        '''

        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Inlet mesh')
        plt.plot(self.boundary[0], self.boundary[1],'k-')
        plt.scatter(self.xy[0],self.xy[1],marker='.')

        if show:
            plt.show()


    @staticmethod
    def makeRegularAxis(points, plotDensity=[None,None]):
        '''
        Makes regular axis from arbitrarily spaced points, for use with contour and streamline plotting.
        - points - list of 2D coordinates of original irregular points
        - plotDensity - the number of points to plot along the x and y axis. If any is None, the functions will attempt to mimic the original's data density in that direction
        '''
        # Get plot density from input or try to mimic original data density
        for i,axis in enumerate(plotDensity):
            if axis is None:
                # Get all points of this dimension
                dim_discrete = np.array(points[i])

                # Get unique 'ticks' of this axis with some tolerance for numeric error
                ticks = np.unique(dim_discrete.round(decimals=6))

                plotDensity[i] = ticks.size

        # Create new regularly spaced axis
        dim_max = np.max(points,axis=1)
        dim_min = np.min(points,axis=1)
        xy_i = np.vstack([np.linspace(dim_min[0], dim_max[0], num=plotDensity[0]), 
                                np.linspace(dim_min[1], dim_max[1], num=plotDensity[1])])

        return xy_i

    @staticmethod
    def makeContourPlot(values, points, numTicks, title=None, regularPoints=None, axisRange=[None,None], boundaryPoints=None, cmap=None, tightRange=False):
        '''
        Creates contour plot figure from 
        - values - values to plot at each point in points
        - points - list of 2D coords corresponding to the values, if regularPoints is not specified, this is assumed to be arranged in a regular grid
        - numTicks - number of ticks for the colour bar
        - title - plot title
        - regularPoints - if this is specified, it is assumed that the input values need to be interpolated from the arbitrarily arranged points to the regularPoints
        - axisRange - the max and min values for the colormap and colourbar
        - boundaryPoints - for drawing a black border for the contour plot
        - cmap - colourmap to use when plotting
        '''
        # Interpolate to a regularly spaced grid if needed
        if regularPoints is not None:
            values = griddata((points[0],points[1]), values, (regularPoints[0][None,:],regularPoints[1][:,None]), method='linear')
        else:
            regularPoints = points

        # Make our own reasonable max and min range if not specified, or have axis tight against actual max and min values
        if not tightRange:
            if None in axisRange:
                (minVal, maxVal) = Plots.__getContourRange__(values)
            else:
                # Make sure the range is increasing
                axisRange = np.sort(axisRange)
                minVal, maxVal = axisRange[0:2]
        else:
            (minVal, maxVal) = np.nanmin(values), np.nanmax(values)

        # Protection for uniform contour plots
        if maxVal-minVal < 1:
                minVal = 0

        # Make ticks for colormap
        ticks = np.linspace(minVal,maxVal,numTicks)
        # Make colormap levels
        levels = np.linspace(minVal,maxVal,101)

        if cmap is None:
            cmap = 'jet'

        # Make contour plot
        fig = plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(title)
        plt.contourf(regularPoints[0],regularPoints[1],values,levels=levels,cmap=cmap,vmin=minVal,vmax=maxVal)
        plt.colorbar(ticks=ticks)
        plt.axis('off')
        # Draw boundary
        if boundaryPoints is not None:
            plt.plot(boundaryPoints[0], boundaryPoints[1], 'k-')

        return fig.number


    @staticmethod
    def saveFigsToPdf(outputFile):
        '''
        Save all current figures into a multi-page pdf
        -outputFile - name of file to save to, including extension
        '''

        with PdfPages(outputFile) as pdf:
            # Go through all active figures and save to a separate pdf page
            for fig in range(1, plt.gcf().number+1):
                pdf.savefig(fig)

        plt.close('all')

    @staticmethod
    def __getContourRange__(values):
        # Convert nans to zeros for statistical calcs
        values = np.nan_to_num(values)
        # Get maximum magnitude of swirl angle
        maxMag = np.max(np.abs(values))
        # Choose appropriate 'round to the nearest'
        if maxMag < 5:
            rounding = 1
        elif maxMag < 30:
            rounding = 5
        else:
            rounding = 10
        # Round max/min values to create range of swirl angles
        minVal = np.floor(np.min(values) / rounding) * rounding
        maxVal = np.ceil(np.max(values)  / rounding) * rounding

        return (minVal,maxVal)


class SwirlDescriptors:
    '''
    For calculating the swirl distortion descriptors developed in "A Methodology for Assessing Inlet Swirl Distortion"
    - flowfield - FlowField object which contains the swirl angle data and the coordinates of the nodes which they are defined at.
    - numRings - number of equal area radial rings to defined the descriptors for
    - numCircPoints - number of equally spaced circumferential points on each ring to sample the swirl angle
    '''

    def __init__(self, flowfield: sg.FlowField, numRings=5, numCircPoints=72):
        # Attribute initialisations
        self.ring_radii = None              # Radial positions of the equal area rings
        self.circAngles = None              # Circumferential position of the equally spaced sample points
        self.swirl = None                   # Swirl angles sampled at the discretisation points
        # Descriptors as defined in literature
        self.circExtent = [None] * numRings         # Circumferential extente
        self.sectSwirl = [None] * numRings          # Sector swirl
        self.SI_list = np.zeros(numRings)           # Swirl intensity for each ring
        self.SD_list = np.zeros(numRings)           # Swirl directivity for each ring
        self.SP_list = np.zeros(numRings)           # Swirl pairs for each ring
        self.SI_stats = np.zeros(3)                 # [Mean, standard deviation, max] of the swirl intensity
        self.SD_stats = np.zeros(3)                 # [Mean, standard deviation, max] of the swirl directivity
        self.SP_stats = np.zeros(3)                 # [Mean, standard deviation, max] of the swirl pairs parameter

        # Get radius of the plane
        R = max(flowfield.coords_polar.real)

        # Initialise array to store radial position of the equal-area rings - plus the actual radius of the plane
        self.ring_radii = np.zeros([numRings+1])
        self.ring_radii[numRings] = R

        # Get radius of equal area rings
        for i in range(numRings-1,-1,-1):
            self.ring_radii[i] = np.sqrt(((i+1)/(i+2))*self.ring_radii[i+1]**2)

        # Now we can remove the last element - we don't need statistics at the wall
        self.ring_radii = self.ring_radii[0:-1]

        # Create circumferential points
        self.circAngles = np.linspace(-180,180,numCircPoints,endpoint=False)

        # Create the polar coordinates of the discretisation points
        Radii, Theta = np.meshgrid(self.ring_radii, self.circAngles, indexing='ij')
        Radii = Radii.flatten()
        Theta = Theta.flatten()

        # Get cartesian coordinates - for interpolating
        Theta_rad = np.deg2rad(Theta)
        cart_points = np.column_stack([Radii * np.cos(Theta_rad), Radii * np.sin(Theta_rad)])

        # Extract the swirl and node location data from the flowfield object
        swirl_orig = flowfield.swirlAngle
        nodes_orig = np.column_stack([flowfield.coords.real, flowfield.coords.imag])

        # Interpolate onto desired discretisation
        self.swirl = griddata((nodes_orig[:,0],nodes_orig[:,1]), swirl_orig, (cart_points[:,0], cart_points[:,1]), method='linear')

        # Reshape the swirl samples so that it is a list of lists of samples at each ring
        self.swirl = self.swirl.reshape([-1,72])


    def getSwirlExtentPairs(self):
        '''
        Calculates the 'swirl pairs' (circumferential extent and sector swirl) as defined in the literature
        - Sets the self.circExtent and self.sectSwirl attributes, as list of ndarrays
        '''

        for i in range(len(self.ring_radii)):
            # Get the zero crossings - https://stackoverflow.com/a/46911822/2946404
            y = self.swirl[i,:]
            x = self.circAngles
            s = np.abs(np.diff(np.sign(y))).astype(bool)
            zero_crossings = x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

            # Append to the ends
            zero_crossings = np.concatenate([[-180],zero_crossings,[180]])
            
            # Get circumferential extents
            self.circExtent[i] = np.diff(zero_crossings)

            # Get correpsonding sector swirls - nested for not ideal but shouldn't have large performance impact, sample size is smalle
            self.sectSwirl[i] = []
            for j in range(len(self.circExtent[i])):
                # Get indices of samples withing this 'sector'
                sectorIdxs = np.logical_and(x >= zero_crossings[j], x <= zero_crossings[j+1])
                self.sectSwirl[i].append(trapz(y[sectorIdxs],x[sectorIdxs]) / self.circExtent[i][j])
            
            self.sectSwirl[i] = np.array(self.sectSwirl[i])


    def getSwirlDescriptors(self):
        '''
        Calculates the swirl descriptors as defined in the literature and their associated statistics
        - Sets the atttributes: self.SI_list, self.SD_list, self.SP_list, self.SI_stats, self.SD_stats, self.SP_stats
        '''

        # Get swirl descriptors for each ring
        for i in range(len(self.ring_radii)):
            absoluteSum = np.sum(np.abs(self.sectSwirl[i])*self.circExtent[i])
            directionalSum = np.sum(self.sectSwirl[i]*self.circExtent[i])

            self.SI_list[i] = absoluteSum / 360
            self.SD_list[i] = directionalSum / absoluteSum
            self.SP_list[i] = absoluteSum / (2 * np.max(np.abs(self.sectSwirl[i])*self.circExtent))

        # Get swirl descriptor statistics across all rings
        self.SI_stats[0] = np.mean(self.SI_list)
        self.SI_stats[1] = np.std(self.SI_list)
        self.SI_stats[2] = np.max(self.SI_list)
        self.SD_stats[0] = np.mean(self.SD_list)
        self.SD_stats[1] = np.std(self.SD_list)
        self.SD_stats[2] = np.max(self.SD_list)
        self.SP_stats[0] = np.mean(self.SP_list)
        self.SP_stats[1] = np.std(self.SP_list)
        self.SP_stats[2] = np.max(self.SP_list)


    def getStatistics(self):
        '''
        Wrapper function to output the swirl statistics as a 2D array
        - Output: (Mean, standard deviation, max) columns; (SI, SD, SP) rows
        '''

        # Call calculation functions
        self.getSwirlExtentPairs()
        self.getSwirlDescriptors()

        # Form output
        out = np.array([list(self.SI_stats), list(self.SD_stats), list(self.SP_stats)])

        return out
        

    @staticmethod
    def getError(field1, field2):
        '''
        Method for calculating the Root Mean Square error between field 1 and field 2
        - field1, field2 - array list of vectors or scalars to be compared, must be the same size
        - if list of vectors, values will be flattened and rmse calculated across all values
        '''

        # Make sure these are ndarrays
        field1 = (field1 if isinstance(field1,np.ndarray) else np.array(field1))
        field2 = (field2 if isinstance(field1,np.ndarray) else np.array(field2))

        field1 = field1.flatten()
        field2 = field2.flatten()

        # Check is same number of elements
        if not np.size(field1) == np.size(field2):
            raise RuntimeError('Two input arrays must have the same number of elements')

        return np.sqrt(np.sum((field1-field2)**2)/np.size(field1))

