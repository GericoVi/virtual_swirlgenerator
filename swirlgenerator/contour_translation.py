# -------------------------------------------------------------------------------------------------------------------
#
# Module for translating images containing contour plots and colour bars into values
# which can then be used to reconstruct the original flow field
#
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
import os
from scipy.interpolate import griddata

class Contour:
    '''
    Class for processing images conatining the flow angles as contour plots
    - imgFile - PNG image file containing the contour plot to translate
    - range - [min, max] values which correspond to the colorbar range of the contour plot
    - cmap - a colourmap can be given instead of extracting it from the image, an array list of RGB values of shape (n,3)
    - sampleDist - parameters for controlling the resolution at which to sample the contour plot, [number of radial rings, angular resolution]
    - showSegmentation - for debugging - if true, shows the bounding circle/box found overlayed onto a greyscale version of the input image
    '''
    
    def __init__(self, imgFile, range, cmap=None, sampleDist=[10,15], showSegmentation=False):
        # Check file existance
        if not os.path.exists(imgFile):
            raise FileNotFoundError(f'{imgFile} not found')
        
        # Extract data into attributes
        self.imgArray = cv2.imread(imgFile)
        self.range = range
        self.cmap = cmap

        # Image geometry
        self.imgGeom = self.imgArray.shape[0:2]
        self.imgArea = self.imgGeom[0] * self.imgGeom[1]

        # Initialise some attributes
        self.contours = None                        # Edges within image as contours (binary image array)
        self.boundaries = None                      # ( (centre_xy,radius), (top_left_x, top_left_y, width, height) ), of bounding circles/boxes
        self.plotPixels = None                      # [[complex coords], [B,G,R]] - stores the BGR 'vectors' of each pixel and their corresponding coordinates (mapped to actual inlet dimensions)

        # Flag for showing 
        self.showSegmentation = showSegmentation

        '''
        Do contour translation workflow on initialisation
        '''
        # If a colourmap has been provided already, we don't need to look for the colour bar in the image
        getColourbar = (True if cmap is None else False)

        # Get the bounding circle/box of the contour plot and colour bar
        self.segmentImage(getColourbar)

        # Get the pixels within the plot and their coords in terms of the inlet dimensions
        self.getPlotPixels()

        if cmap is None:
            # Extract the colourmap associated to this plot from the image if needed
            self.__getCmap__()

        # Get points for sampling the contour plot
        self.samples = Contour.samplePoints(sampleDist[0],sampleDist[1],self.boundaries[0][1])

        # Get values at sample points
        self.__getValues__()


    @staticmethod
    def samplePoints(numRings, angleRes, plotRadius):
        '''
        Generates a distribution of sample points to query the contour plot
        - numRings - number of radial positions to sample
        - angleRes - angular resolution we want to sample at, number of circumferential points will be calculated from this
        - plotRadius - radius of contour plot within the image, in pixels
        - Outputs list of sample point coordinates [complex cartesian]
        - Static method since this could be useful outside the class
        '''

        # Get number of circumferential points from input angular resolution
        numAngles = int(np.ceil(360/angleRes))

        # Make polar axis list
        radii = np.linspace(0,plotRadius,numRings+1,endpoint=False)        # We don't need sample points at the boundary - since contour plots normally have a black border
        angles = np.linspace(-180,180,numAngles,endpoint=False)    # We don't need another point at 360, already have on there (0 degrees)

        # Convert to radians
        angles = np.deg2rad(angles)

        # Remove the 0 radial position - added again later, so we don't have multiple points at the centre
        radii = radii[1:]

        # Make grid of coords
        R, Theta = np.meshgrid(radii,angles,indexing='ij')

        # Flatten into list and add the point at the centre
        coords_polar = np.column_stack([R.flatten(), Theta.flatten()])
        coords_polar = np.concatenate([[[0,0]], coords_polar])

        # Transform to complex cartesian coords - easier Euclidean distance calculations later
        coords = (coords_polar[:,0] * np.cos(coords_polar[:,1])) + 1j * (coords_polar[:,0] * np.sin(coords_polar[:,1]))

        return coords


    def segmentImage(self, getColourbar=True):
        '''
        Segments the input image and returns data on the bounding circle of the plot and the bounding rectangle of the colour bar 
        - getColourbar - flag to control if we looking for the colour bar as well
        - Sets the self.contours and self.boundaries attributes
        '''

        # Greyscale for edge detection
        img_grey = cv2.cvtColor(self.imgArray,cv2.COLOR_BGR2GRAY)

        # Greyscale image but with 3 channels - so the bounding boxes/circles can be drawn in colour to contrast with original image
        img = np.dstack([img_grey,img_grey,img_grey])

        # Get edges as a binary image with the canny algorithm - with these threshold values, the plot and colour bar edges are reliably found while the contour lines within the plot are mostly ignored
        canny_edges = cv2.Canny(img_grey,100,200)

        # Get edges as contours (lists of points) - as just a list with no hierarchical relationships (we don't need the hierarchy) and with no approximation so all points within the contour are returned
        self.contours, _ = cv2.findContours(canny_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        # Find the bounding rectangle for the colour bar if needed
        if getColourbar:
            colourbar_box = self.__findColourbar__(drawing=img)
        else:
            colourbar_box = None

        # Find the bounding circle for the plot
        plot_circle = self.__findPlot__(drawing=img)

        # Show segmented image
        if self.showSegmentation:
            # Show with matplotlib so we can zoom
            from matplotlib import pyplot as plt
            plt.figure(), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            plt.show()

        # Form tuple and assign to attribute
        self.boundaries = (plot_circle, colourbar_box)


    def __findPlot__(self, drawing=None):
        '''
        Finds the bounding circle of the contour plot within the image and returns its centre and radius
        - Internal function, should not be used outside Contours class
        - drawing - 3 channel array to draw the bounding circle to, won't draw if none
        - Outputs tuple - ([centre_x, centre_y], radius)
        - Assumes that the contour plot takes up more than half of the image area and is roughly circular
        '''

        # Get the contours which are larger than half the whole image area - these will probably be the edges around the contour plot
        possible_contours = [contour for contour in self.contours if cv2.contourArea(contour) >= 0.3*self.imgArea]

        # The smallest contour out of these will probably be the one at the edge of the plot
        plot_edge_idx = np.argmin([cv2.contourArea(contour) for contour in possible_contours])

        # Get the bounding cirlce of the plot 
        centre, radius = cv2.minEnclosingCircle(possible_contours[plot_edge_idx])

        # Draw on image if it was given - does not need to be outputted since image is passed by reference
        if drawing is not None:
            cv2.circle(drawing, (int(centre[0]), int(centre[1])), int(radius), (0,0,255),2)

        return (centre, radius)


    def __findColourbar__(self, drawing=None):
        '''
        Finds the bounding rectangle of the colour bar within the image and returns its ____
        - Internal function, should not be used outside Contours class
        - drawing - 3 channel array to draw the bounding circle to, won't draw if none
        - Outputs tuple - (top_left_x, top_left_y, width, height)
        - Assumes that the colour bar is below the contour plot, in the lower third of the image
        '''

        possible_contours = []
        rectangles = []
        for c in self.contours:
            # Get bounding rectangle
            rectangle = cv2.boundingRect(c)

            # Get centre of bounding box
            cx = int(rectangle[0]+rectangle[2]/2)
            cy = int(rectangle[1]+rectangle[3]/2)

            # Add to list if the contour is in the outer quarters of the image (opencv has y axis positive downwards)
            if (cy > 3*self.imgGeom[0]/4 or cy < self.imgGeom[0]/4 or cx > 3*self.imgGeom[1]/4 or cx < self.imgGeom[1]/4):
                possible_contours.append(c)
                rectangles.append(rectangle)

            #cv2.circle(img, (cx,cy),1,(0,255,0),3)
            #cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[0]+rectangle[2]), int(rectangle[1]+rectangle[3])), (255,0,0),3)

        # Get the index of the bounding box in the lower half of the image with the largest area - probably the colourbar
        bounding_idx = np.argmax([rectangle[2]*rectangle[3] for rectangle in rectangles])

        # Get actual bounding box from list
        boundingbox = rectangles[bounding_idx]

        # Draw on image if it was given - does not need to be outputted since image is passed by reference
        if drawing is not None:
            cv2.rectangle(drawing, (int(boundingbox[0]), int(boundingbox[1])), (int(boundingbox[0]+boundingbox[2]), int(boundingbox[1]+boundingbox[3])), (255,0,0),2)

        return boundingbox


    def __getCmap__(self):
        '''
        Extracts the RGB values of the levels within the colourbar
        - Internal function, should not be used outside Contours class
        - Sets the self.cmap attribute
        '''

        boundingbox = self.boundaries[1]

        # Check if colour bar is horizontal or vertical and extract list of rgb levels based on this
        if boundingbox[2] > boundingbox[3]:
            # Coordinates of start and end of colour bar - sampling at a point on the lower third of the box, less sensitivity to inaccuracies in bounding box
            colourbar_start = [boundingbox[0], boundingbox[1]+2*int(boundingbox[3]/3)]
            colourbar_end   = [boundingbox[0]+boundingbox[2], boundingbox[1]+2*int(boundingbox[3])]

            # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
            # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
            colourbar = self.imgArray[colourbar_start[1],colourbar_start[0]+2:colourbar_end[0]-2, :]
        else:
            # Coordinates of start and end of colour bar - sampling at a point on the left third of the box, less sensitivity to inaccuracies in bounding box
            # Also, flipped compared to above since opencv y axis is positive downwards
            colourbar_start   = [boundingbox[0]+int(boundingbox[2]/3), boundingbox[1]+boundingbox[3]]
            colourbar_end = [boundingbox[0]+int(boundingbox[2]/3), boundingbox[1]]

            # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
            # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
            colourbar = self.imgArray[colourbar_start[1]+2:colourbar_end[1]-2:-1,colourbar_start[0], :]

        # Normalise RGB values
        cmap = colourbar / 255

        # Transform to RGB
        self.cmap = np.column_stack([cmap[:,2],cmap[:,1],cmap[:,0]])


    def getPlotPixels(self):
        '''
        Extracts the BGR 'vectors' with their corresponding coordinates mapped to the actual inlet dimensions
        - Sets the self.plotPixels attribute
        '''

        boundingcircle = self.boundaries[0]

        # Make coordinate grid for the image in terms of pixels - indexing in [row,column] format to match how cv2 stores images
        Y,X = np.meshgrid(np.arange(0,self.imgGeom[0]), np.arange(self.imgGeom[1]), indexing='ij')

        # Flatten both the coordinates and the imgArray (preserving the shape of the RGB 'vectors')
        x = X.reshape(-1,1)
        y = Y.reshape(-1,1)
        pixels = self.imgArray.reshape(-1,3)

        # Transform coordinates to have origin coincident to plot centre
        x = x-boundingcircle[0][0]
        y = y-boundingcircle[0][1]

        # Form into complex coords - easier euclidean distance calculations
        coords = x + 1j * y
        
        # Get indices of pixels in plot
        plotIdxs = np.abs(coords) <= boundingcircle[1]
        plotIdxs = plotIdxs[:,0]

        # Get only the pixels in the plot
        plotPixels = pixels[plotIdxs, :]

        # Normalise the BGR values
        plotPixels = plotPixels/255

        # Get corresponding coords
        plotCoords = coords[plotIdxs]

        # Transform y coordinates - since opencv y axis is positive downwards
        plotCoords = plotCoords.real + 1j * -plotCoords.imag

        # Form tuple and assign to attribute
        self.plotPixels = ((plotCoords),(plotPixels))
        

    def getValuesAtNodes(self, nodes):
        '''
        Interpolates the values at the extracted sample points onto a separate input discretisation
        - nodes - the new discretisation points to interpolate the values to, list of 2D cartesian coordinates or complex coordinates
        - Returns the a list of values corresponding
        '''

        # Make sure that nodes are in complex coordinate form
        if len(nodes.shape) == 2:
            nodes = nodes[:,0] * 1j * nodes[:,1]
        elif len(nodes.shape) != 1:
            raise RuntimeError(f'Invalid shape {nodes.shape} for input nodes list. Should be either list of 2D cartesian coordinates or complex coordinates')


        # Get pixel to inlet unit mapping - assuming that the nodes capture the boundary
        ductRadius = max(abs(nodes))
        pixToUnit = ductRadius/self.boundaries[0][1]

        # Convert sample point coordinates from pixel units to inlet units, also from complex coords to 2D array
        samples = np.column_stack([self.samples.real*pixToUnit, self.samples.imag*pixToUnit])

        # Interpolate onto desired discretisation
        values = griddata((samples[:,0], samples[:,1]), self.values, (nodes.real, nodes.imag), method='linear')

        # Since we have not sampled at the edges, the boundary nodes will have nan values.
        # Need to use griddata again for boundary nodes using the 'nearest' interpolation method
        boundaryIdxs = np.isnan(values)
        values[boundaryIdxs] = griddata((samples[:,0], samples[:,1]), self.values, (nodes[boundaryIdxs].real, nodes[boundaryIdxs].imag), method='nearest')

        return values


    def __getValues__(self):
        '''
        Gets the numerical values associated with the contour plot at the sample points
        - Sets the self.values attribute
        '''

        # Extract the arrays from the input tuple
        pixelCoords = self.plotPixels[0][:,0]
        pixelBGRs = self.plotPixels[1]

        # Transform colour map to BGR, to match the pixel values read by opencv
        cmap = np.column_stack([self.cmap[:,2],self.cmap[:,1],self.cmap[:,0]])

        # For storing the corresponding level of the pixels' colour at that node within the colourmap
        levels = np.array([None]*len(self.samples))

        # Loop through the sample points
        for i,sample in enumerate(self.samples):
            # Get index of closest pixel to this sample point
            closest_idx = np.argmin(np.abs(pixelCoords - sample))

            # Get BGR for this pixel
            BGR = pixelBGRs[closest_idx]
            #print(BGR)
            # Get index of closest closer to this value from the colour map levels, by getting the Euclidean distance in a 3D space where B,G,R are the dimensions
            level_idx = np.argmin(np.linalg.norm((cmap - BGR), axis=1))

            # Normalise
            levels[i] = level_idx/cmap.shape[0]
            
        # Map these levels to the actual value range and output
        self.values = np.array((levels*(self.range[1]-self.range[0]) + self.range[0]), dtype=float)

