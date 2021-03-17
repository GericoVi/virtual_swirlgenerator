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
from scipy.spatial import ConvexHull

class Contour:
    '''
    Class for processing images conatining the flow angles as contour plots
    - imgFile           - PNG image file containing the contour plot to translate
    - range             - [min, max] values which correspond to the colorbar range of the contour plot
    - doTranslation     - by default, the contour plot image will be translated on initialisation with default parameters, this can stop it for greater user control
    - cmap              - a colourmap can be given instead of extracting it from the image, an array list of RGB values of shape (n,3)
    - showSegmentation  - for debugging - if true, shows the bounding circle/box found overlayed onto a greyscale version of the input image
    '''
    
    def __init__(self, imgFile, range, doTranslation=True, cmap=None, showSegmentation=False):
        # Check file existance
        if not os.path.exists(imgFile):
            raise FileNotFoundError(f'{imgFile} not found')
        
        # Extract data into attributes
        self.imgArray = cv2.imread(imgFile)
        self.debug_fig = self.imgArray.copy()   # Copy of image for drawing over segmentation boundaries
        self.range = np.sort(range)             # Make sure the colourmap range is increasing
        self.cmap = cmap

        # Image geometry
        self.imgGeom = self.imgArray.shape[0:2]
        self.imgArea = self.imgGeom[0] * self.imgGeom[1]

        # Initialise some attributes
        self.values = None                          # Values extracted from contour plot
        self.contours = None                        # Edges within image as contours (binary image array)
        self.boundaries = None                      # ( (centre_xy,radius), (top_left_x, top_left_y, width, height) ), of bounding circles/boxes
        self.pixels = None                          # [[complex coords], [B,G,R]] - stores the BGR 'vectors' of each pixel and their corresponding coordinates relative to an origin coincident with the centre of the contour plot
        self.error = 0                              # Error flag
        self.error_message = None                   # Corresponding error message

        # Flag for showing segmented image
        self.showSegmentation = showSegmentation

        # Image processing parameters
        self.circleParams = None
        self.minLevels = None

        # If a colourmap has been provided already, we don't need to look for the colour bar in the image
        getColourbar = (True if cmap is None else False)

        '''
        Do contour translation workflow on initialisation with default values
        '''
        if doTranslation:
            self.translateContourPlot(getColourbar=getColourbar)


    def translateContourPlot(self, getColourbar=True, samplingMode=1, samplingParams=(10), circleParams=(200,30), minLevels=100, shrinkPlotMax=10):
        '''
        Wrapper function for translating a contour plot image after initialisation
        - getColourbar      - flag for extracting the colourbar or not from the image
        - samplingMode      - sampling distribution mode to be used - details in samplePoints() function
        - samplingParams    - parameters for controlling the distribution of sampling points - details in samplePoints() function
        - circleParams      - two parameters which control the Hough circle algorithm function from openCV2 - https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        - minLevels         - Minimum number of unique RGB values in the list to stop searching for the colour map within the image
        - shrinkPlotMax     - Number of times we can try to reduce the plot size - relevant to the __shrinkPlot__() function
        '''

        # Set the necessary attributes using the inputs
        self.circleParams = circleParams
        self.minLevels = minLevels

        # Get the bounding circle/box of the contour plot and colour bar
        self.segmentImage(getColourbar)

        # Continue if image segmentation was successful
        if self.error == 0:

            # Get normalised pixel colours and their coords in terms of an origin coincident with the plot's centre
            self.getPixels()

            # Adjust the plot radius by checking if the edge pixels have colours within the stored colourmap
            if shrinkPlotMax > 0:
                self.shrinkPlot(shrinkPlotMax)

            # Show final segmented image if requested
            if self.showSegmentation:
                # Plot segmentation
                cv2.circle(self.debug_fig, (self.boundaries[0][0][0],self.boundaries[0][0][1]), self.boundaries[0][1], (255,0,255), 2)
                
                # Colourbar segmentation
                if getColourbar:
                    cv2.rectangle(self.debug_fig, (int(self.boundaries[1][0]), int(self.boundaries[1][1])), (int(self.boundaries[1][0]+self.boundaries[1][2]), int(self.boundaries[1][1]+self.boundaries[1][3])), (255,0,255),2)

                # Show with matplotlib so we can zoom
                from matplotlib import pyplot as plt
                plt.figure(), plt.imshow(cv2.cvtColor(self.debug_fig,cv2.COLOR_BGR2RGB))

            # Get points for sampling the contour plot
            self.samples = Contour.samplePoints(samplingMode, samplingParams,self.boundaries[0][1])

            # Get values at sample points
            self.getValues()
        
        else:
            if self.error == 1:
                self.error_message = f'Suitable plot area not found'
            elif self.error == 2:
                self.error_message = f'Suitable colour bar not found'


    @staticmethod
    def samplePoints(mode, params, plotRadius):
        '''
        Generates a distribution of sample points to query the contour plot
        - mode          - sampling distribution mode to be used
                        - 1 - equidistant points
                        - 2 - constant angular resolution
                        - 3 - using nodes from an outside mesh
        - params        - tuple containing the parameters needed to define the distribution of the sample points, depends on mode
                        - mode 1 - (number of radial rings)
                        - mode 2 - (number of radial rings, angular resolution in degrees)
                        - mode 3 - ([list of complex coordinates of nodes])
        - numRings      - number of radial positions to sample
        - angleRes      - angular resolution we want to sample at, number of circumferential points will be calculated from this
        - plotRadius    - radius of contour plot within the image, in pixels
        - Outputs list of sample point coordinates [complex cartesian]
        - Static method since this could be useful outside the class
        '''

        # Equidistant points, controlled by number of rings - algorithm adapted from http://www.holoborodko.com/pavel/2015/07/23/generating-equidistant-points-on-unit-disk/
        if mode == 1:
            # Extract parameters from input tuple
            numRings = params

            # Distance between rings
            dR = plotRadius/numRings

            # Initialise list of list of coordinates of points on each ring
            coords_list = [None]*numRings

            # Place points on concentric circles with (almost) equal arc length between them
            for k,r in enumerate(np.arange(dR,plotRadius+dR,dR), 1):
                # Number of points needed
                n = int(np.round( np.pi / np.arcsin(1/(2*k)) ))

                # Angular position of points
                angles = np.linspace(0, 2*np.pi, n+1)

                # Make list of polar coordinate pairs
                coords_polar = np.column_stack([[r]*len(angles) , angles])

                # Add to list of cartesian coords
                coords_list[k-1] = (coords_polar[:,0] * np.cos(coords_polar[:,1])) + 1j * (coords_polar[:,0] * np.sin(coords_polar[:,1]))

            # Merge into one list, and add point in the centre
            coords = np.concatenate(coords_list)
            coords = np.concatenate([coords, [0+1j*0]])

        # Constant angluar displacement between points across the rings
        elif mode == 2:
            # Extract parameters from input tuple
            numRings, angleRes = params

            # Get number of circumferential points from input angular resolution
            numAngles = int(np.ceil(360/angleRes))

            # Make polar axis list
            radii = np.linspace(0,plotRadius,numRings+1,endpoint=True)        
            angles = np.linspace(-180,180,numAngles,endpoint=False)     # We don't need another point at 360, already have one there (0 degrees)

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

        # Sample point distribution received as input
        elif mode == 3:
            # Extract node locations from input tuple
            coords = params

            # Assuming that there are nodes at the boundary of the circle, get radius
            radius = np.max(np.abs(coords))

            # Normalise and map to pixel coordinates
            coords = (coords/radius)*plotRadius

        return coords


    def segmentImage(self, getColourbar=True):
        '''
        Segments the input image and returns data on the bounding circle of the plot and the bounding rectangle of the colour bar 
        - getColourbar - flag to control if we looking for the colour bar as well
        - Sets the self.contours and self.boundaries attributes
        '''

        # Greyscale for edge detection
        img_grey = cv2.cvtColor(self.imgArray,cv2.COLOR_BGR2GRAY)

        # Get edges as a binary image with the canny algorithm - with these threshold values, the plot and colour bar edges are reliably found while the contour lines within the plot are mostly ignored
        canny_edges = cv2.Canny(img_grey,100,200)

        # Dilate the edges so we can close small gaps (with a 3x3 rectangular kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilated = cv2.dilate(canny_edges, kernel)
        # Get edges as contours (lists of points) - as just a list with no hierarchical relationships (we don't need the hierarchy) and with no approximation so all points within the contour are returned
        self.contours, _ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Find the bounding rectangle for the colour bar if needed
        if getColourbar:
            colourbar_box = self.__findColourbar__()
        else:
            colourbar_box = None

        # Find the bounding circle for the plot
        plot_circle = self.__findPlot__(img_grey)

        # Form tuple and assign to attribute
        self.boundaries = (plot_circle, colourbar_box)

        if self.error != 0:
            # If error flag, there was no qualifying contours found during one of the checks (either to find the plot or the colourbar)
            # So we will draw the contours, so we can debug
            cv2.drawContours(self.debug_fig, self.contours, -1, (0,255,0), 1)
            from matplotlib import pyplot as plt
            plt.figure(), plt.imshow(cv2.cvtColor(self.debug_fig,cv2.COLOR_BGR2RGB))


    def __findPlot__(self, greyscale):
        '''
        Finds the bounding circle of the contour plot within the image and returns its centre and radius
        - Internal function, should not be used outside Contours class
        - greyscale - single channel image
        - Outputs tuple - ([centre_x, centre_y], radius)
        '''

        rows = greyscale.shape[1]

        # Get circle - tends to return the largest circle around plot with these parameters
        circles = cv2.HoughCircles(greyscale, cv2.HOUGH_GRADIENT, 1, rows/8, param1=self.circleParams[0], param2=self.circleParams[1], minRadius=int(rows/4), maxRadius=int(rows/2))
        # for x,y,r in circles[0]:
        #     x,y,r = int(x), int(y), int(r)
        #     cv2.circle(self.debug_fig, (x,y), r, (0,255,0), 1)

        # cv2.imshow('circles',self.debug_fig)
        # cv2.waitKey(0)

        # print(circles)

        if circles is not None:
            # Use the biggest circle
            circle = circles[0, np.argmax(circles[0,:,2]), :]

            x,y,r = circle
            x,y,r = int(x), int(y), int(r)

            return ([x,y],r)

        else:
            self.error = 1
            return None


    def __findColourbar__(self):
        '''
        Finds the bounding rectangle of the colour bar within the image and returns its ____
        - Internal function, should not be used outside Contours class
        - Outputs tuple - (top_left_x, top_left_y, width, height)
        - Assumes that the colour bar is in the outer quarters of the image
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

        try:
            # Get the index of the bounding box in the lower half of the image with the largest area - probably the colourbar
            bounding_idx = np.argmax([rectangle[2]*rectangle[3] for rectangle in rectangles])

            # Get actual bounding box from list
            boundingbox = rectangles[bounding_idx]

            # Goes through all the bounding boxes within the processed image. If we run out of bounding boxes, error is passed to top level.
            while True:
                # Try and find the colourmap in this bounding box
                found = self.__getCmap__(boundingbox)

                if found:
                    break
                else:
                    # If a suitable colourmap was not found within this bounding box, check the next largest one
                    rectangles = np.delete(rectangles, bounding_idx)
                    bounding_idx = np.argmax([rectangle[2]*rectangle[3] for rectangle in rectangles])
                    boundingbox = rectangles[bounding_idx]

            return boundingbox
        
        # If colour bar was not found, set error
        except:
            self.error = 2
            return None


    def __getCmap__(self, boundingbox):
        '''
        Extracts the RGB values of the levels within the colourbar. Checks the middle and the outer halfs of the bounding box to increase likelihood of finding the colourmap.
        - Internal function, should not be used outside Contours class

        - Sets the self.cmap attribute
        '''

        found = False

        # Check the outer thirds of the boundaing box where a clear sample of the colourmap will likely be
        positions = [1/3, 2/3]

        for position in positions:
            # Check if colour bar is horizontal or vertical and extract list of rgb levels based on this
            if boundingbox[2] > boundingbox[3]:
                # Coordinates of start and end of colour bar
                colourbar_start = [boundingbox[0], boundingbox[1]+int(boundingbox[3]*position)]
                colourbar_end   = [boundingbox[0]+boundingbox[2], boundingbox[1]+int(boundingbox[3]*position)]

                # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
                # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
                colourbar = self.imgArray[colourbar_start[1],colourbar_start[0]+2:colourbar_end[0]-2, :]
            else:
                # Coordinates of start and end of colour bar
                # Also, flipped compared to above since opencv y axis is positive downwards
                colourbar_start   = [boundingbox[0]+int(boundingbox[2]*position), boundingbox[1]+boundingbox[3]]
                colourbar_end = [boundingbox[0]+int(boundingbox[2]*position), boundingbox[1]]

                # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
                # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
                colourbar = self.imgArray[colourbar_start[1]+2:colourbar_end[1]-2:-1,colourbar_start[0], :]

            # Check if we have actually sampled the colour map by checking if there is a minimum number of unique RGB values
            if np.shape(np.unique(colourbar, axis=0))[0] > self.minLevels:
                found = True
                break

        # Normalise RGB values
        cmap = colourbar / 255

        # Transform to RGB
        self.cmap = np.column_stack([cmap[:,2],cmap[:,1],cmap[:,0]])

        return found


    def getPixels(self):
        '''
        Extracts the BGR 'vectors' with their corresponding coordinates mapped to the actual inlet dimensions
        - Sets the self.pixels attribute
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

        # Normalise the BGR values
        pixels = pixels/255

        # Transform y coordinates - since opencv y axis is positive downwards
        coords = coords.real + 1j * -coords.imag

        # Form tuple and assign to attribute
        self.pixels = ((coords),(pixels))
        

    def getValuesAtNodes(self, nodes, interpolation='linear'):
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

        # Get convex hull of sample points - Need to add an additional outer ring of points so that all boundary nodes get values
        # Because the input nodes will likely be of a higher resolution than the sample points, some will lay outside the convex hull of the sample points because of circle discretisation
        hull = ConvexHull(samples)
        samples = np.append(samples, samples[hull.vertices,:]*1.1, 0)
        values = np.append(self.values, self.values[hull.vertices])

        # Interpolate onto desired discretisation
        values = griddata((samples[:,0], samples[:,1]), values, (nodes.real, nodes.imag), method=interpolation)

        return values


    def getValues(self):
        '''
        Gets the numerical values associated with the contour plot at the sample points
        - Sets the self.values attribute
        '''

        # Extract the arrays from the tuple
        pixelCoords = self.pixels[0][:,0]
        pixelBGRs = self.pixels[1]

        # Reduce to only the pixels within the plot for less computation
        radius = self.boundaries[0][1]
        plotIdxs = np.abs(pixelCoords) <= radius
       
        pixelCoords = pixelCoords[plotIdxs]
        pixelBGRs   = pixelBGRs[plotIdxs, :]

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

            # Get index of colour closest to this value from the colour map levels, by getting the Euclidean distance in a 3D space where B,G,R are the dimensions
            level_idx = np.argmin(np.linalg.norm((cmap - BGR), axis=1))

            # Normalise
            levels[i] = level_idx/cmap.shape[0]

        # Map these levels to the actual value range and output
        self.values = np.array((levels*(self.range[1]-self.range[0]) + self.range[0]), dtype=float)


    def shrinkPlot(self, numShrinks, ):
        '''
        Reduces the stored plot radius to try and 'crop' out the border or other artifacts which are not relevant to the contour plot.
        By checking the edge pixels to see if over half of them have RGB values which are within the stored colourmap
        - numShrinks - maximum number of times to try and shrink the plot area to crop out borders etc before giving up
        '''

        # Extract the arrays from the tuple
        pixelCoords = self.pixels[0][:,0]
        pixelBGRs = self.pixels[1]

        # Transform colour map to BGR, to match the pixel values read by opencv
        cmap = np.column_stack([self.cmap[:,2],self.cmap[:,1],self.cmap[:,0]])

        # Get current plot radius
        radius = self.boundaries[0][1]

        for _ in range(numShrinks):

            # Make edge sample points with 10 degrees between them
            angles = np.deg2rad(np.linspace(-180,180,36, endpoint=False))

            # Make coords
            coords_polar = np.column_stack([[radius]*36 , angles])
            coords = (coords_polar[:,0] * np.cos(coords_polar[:,1])) + 1j * (coords_polar[:,0] * np.sin(coords_polar[:,1]))

            tol = np.empty(np.shape(coords))

            for i,sample in enumerate(coords):
                # Get index of pixel at this sample point
                closest_idx = np.argmin(np.abs(pixelCoords - sample))

                # Get BGR for this pixel
                BGR = pixelBGRs[closest_idx]

                # Get closest distance from a colour map level and append to list
                tol[i] = np.min(np.linalg.norm((cmap - BGR), axis=1))

            # If the pixel colour is within the colourmap, its minimum distance tends to be less than 0.1
            # If more than 90% of the samples have a value higher than this, assume we are sampling outside the actual contour plot and reduce the radius
            tol = tol >= 0.05

            if np.sum(tol) > np.size(tol)*0.25:
                radius = radius - 1
            else:
                break

        #print(f'Shrinked {n} times')

        # Replace tuple - since can't do item assignment
        centre = self.boundaries[0][0]
        box = self.boundaries[1]
        self.boundaries = ((centre,radius),box)

