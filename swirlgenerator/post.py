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
        self.vel = flowfield.velocity[:,0:2]
        self.thermos = [flowfield.rho, flowfield.pressure]
        self.swirlAngle = flowfield.swirlAngle
        self.boundary = np.vstack([flowfield.boundaryCurve.real, flowfield.boundaryCurve.imag])

        # Get plot density from input or try to mimic original data density
        for i,axis in enumerate(plotDensity):
            if axis is None:
                # Get all points of this dimension
                dim_discrete = np.array(self.xy[i])

                # Get unique 'ticks' of this axis with some tolerance for numeric error
                ticks = np.unique(dim_discrete.round(decimals=6))

                plotDensity[i] = ticks.size

        self.plotDensity = plotDensity

        # Create new regularly spaced axis
        self.dim_max = np.max(self.xy,axis=1)
        self.dim_min = np.min(self.xy,axis=1)
        self.xy_i = np.vstack([np.linspace(self.dim_min[0], self.dim_max[0], num=self.plotDensity[0]), 
                                np.linspace(self.dim_min[1], self.dim_max[1], num=self.plotDensity[1])])


    def plotAll(self, pdfName=None):
        '''
        Utility for showing and saving all plots
        - pdfName - filename to save the plots to, should include extension
        '''

        self.plotVelocity()

        #self.plotThermos()

        self.plotSwirl()

        # If saving, don't show plots
        if (pdfName != None):
            self.__saveFigsToPdf__(pdfName)
        else:
            plt.show()


    def plotVelocity(self, arrowDensity=50):
        '''
        Create plots for the swirling velocity profile as a quiver plot and a streamlines plot
        - arrowDensity - Parameter which controls the sub sampling of the velocity field for the quiver plot
        '''

        # Sub sample the velocity profile so that the quiver plot is readable
        reduced_x = np.linspace(self.dim_min[0],self.dim_max[0],arrowDensity)
        reduced_y = np.linspace(self.dim_min[1],self.dim_max[1],arrowDensity)
        u = griddata((self.xy[0],self.xy[1]), self.vel[:,0], (reduced_x[None,:],reduced_y[:,None]), method='linear')
        v = griddata((self.xy[0],self.xy[1]), self.vel[:,1], (reduced_x[None,:],reduced_y[:,None]), method='linear')

        # Make quiver plot
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Quiver")
        plt.quiver(reduced_x, reduced_y, u, v, units='dots', width=2,headwidth=5,headlength=5,headaxislength=2.5)
        plt.axis('off')
        # Draw boundary
        plt.plot(self.boundary[0], self.boundary[1],'k-')

        # Interpolate data to a regularly spaced grid
        u = griddata((self.xy[0],self.xy[1]), self.vel[:,0], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='cubic')
        v = griddata((self.xy[0],self.xy[1]), self.vel[:,1], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='cubic')

        # Make streamlines plot
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Streamlines")
        plt.streamplot(self.xy_i[0], self.xy_i[1], u, v, density=2)            # streamplot uses vector axis for xy instead of meshgrid for some reason?
        plt.axis('off')
        # Draw boundary
        plt.plot(self.boundary[0], self.boundary[1],'k-')


    def plotThermos(self):
        '''
        Create contour plots for density and pressure field
        '''

        rho = griddata((self.xy[0],self.xy[1]), self.thermos[0], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='cubic')
        pressure = griddata((self.xy[0],self.xy[1]), self.thermos[1], (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='cubic')

        plt.figure()
        plt.title('Density')
        plt.contourf(self.xy_i[0],self.xy_i[1],rho,100,cmap='jet')
        plt.colorbar()

        plt.figure()
        plt.title('Pressure')
        plt.contourf(self.xy_i[0],self.xy_i[1],pressure,100,cmap='jet')
        plt.colorbar()


    def plotSwirl(self):
        '''
        Create contour plot for swirl angle
        '''

        # Interpolate data to a regularly spaced grid
        swirlAngle = griddata((self.xy[0],self.xy[1]), self.swirlAngle, (self.xy_i[0][None,:],self.xy_i[1][:,None]), method='cubic')

        # Convert nans to zeros for statistical calcs
        swirlAngle_noNans = np.nan_to_num(swirlAngle)
        # Get maximum magnitude of swirl angle
        maxMag = np.max(np.abs(swirlAngle_noNans))
        # Choose appropriate 'round to the nearest'
        if maxMag < 5:
            rounding = 1
        elif maxMag < 10:
            rounding = 5
        else:
            rounding = 10
        # Round max/min values to create range of swirl angles
        minVal = np.floor(np.min(swirlAngle_noNans) / rounding) * rounding
        maxVal = np.ceil(np.max(swirlAngle_noNans)  / rounding) * rounding

        # Make ticks for colormap
        ticks = np.linspace(minVal,maxVal,11)

        # Make contour plot
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Swirl angle')
        # For some reason contourf doesn't like when the coordinate grids have nans in them, so using zero instead of nan versions of array
        plt.contourf(self.xy_i[0], self.xy_i[1],swirlAngle,100,cmap='jet',vmin=minVal,vmax=maxVal)
        plt.colorbar(ticks=ticks)
        plt.axis('off')
        # Draw boundary
        plt.plot(self.boundary[0], self.boundary[1],'k-')


    def __saveFigsToPdf__(self, outputFile):
        '''
        Save all current figures into a multi-page pdf
        - Internal function, should not be used outside plots.py
        -outputFile - name of file to save to, including extension
        '''

        with PdfPages(outputFile) as pdf:
            # Go through all active figures and save to a separate pdf page
            for fig in range(1, plt.gcf().number+1):
                pdf.savefig(fig)