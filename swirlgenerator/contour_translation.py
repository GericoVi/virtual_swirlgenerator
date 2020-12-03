# -------------------------------------------------------------------------------------------------------------------
#
# Module for translating images containing contour plots and colour bars into values
# which can then be used to reconstruct the original flow field
#
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
import os

class Contour:
    '''
    Class for processing images conatining the flow angles as contour plots
    - imgFile1 - PNG image file containing the contour plot to translate
    - range - [min, max] values which correspond to the colorbar range of the contour plot
    - nodes - complex coordinates of nodes on the inlet plane (extracted from mesh)
    - cmap - a colourmap can be given instead of extracting it from the image, an array list of RGB values of shape (n,3)
    '''
    
    def __init__(self,imgFile,range,nodes,cmap=None,showSegmentation=False):
        # Check file existance
        if not os.path.exists(imgFile):
            raise FileNotFoundError(f'{imgFile} not found')
        
        # Extract variables into object
        self.imgArray = cv2.imread(imgFile)
        self.range = range
        self.nodes = nodes

        # Flag for showing 
        self.showSegmentation = showSegmentation

        # Get the actual radius of the plot in terms of the inlet units - using the nodes, assuming that the boundary is made up of nodes
        self.duct_radius = max(abs(self.nodes))

        # Get values of contour plot at nodes during initialisation and store into an attribute 
        # - makes outside workflows more streamlined, while keeping flexibility for calling individual methods after
        self.values = self.translateContour(cmap)


    def translateContour(self,cmap=None):
        '''
        Main function which will be called outside this module, contains the workflow for reading in the contour plot image and extracting the values used to plot
        - cmap - a colourmap can be given instead of extracting it from the image, an array list of RGB values of shape (n,3)
        - Outputs a flat numpy array with the order of values corresponding to the input nodes
        '''

        # If a colourmap has been provided already, we don't need to look for the colour bar in the image
        getColourbar = (True if cmap is None else False)

        # Get the bounding circle/box of the contour plot and colour bar
        boundaries = self.segmentImage(self.imgArray, getColourbar)

        # Get the pixels within the plot and their coords in terms of the inlet dimensions
        plotPixels = self.getPlotPixels(self.imgArray, boundaries[0])

        if cmap is None:
            # Extract the colourmap associated to this plot from the image if needed
            cmap = self.__getCmap__(self.imgArray, boundaries[1])

        # Get actual values at nodes
        return self.getValuesAtNodes(plotPixels, cmap, self.nodes, self.range)


    def segmentImage(self, imgArray, getColourbar=True):
        '''
        Segments the input image and returns data on the bounding circle of the plot and the bounding rectangle of the colour bar 
        - imgArray - numpy array of the 3 channel BGR image
        - getColourbar - flag to control if we looking for the colour bar as well
        - Output: tuple, ( (centre_xy,radius), (top_left_x, top_left_y, width, height) ), of 
        '''

        # Get image dimensions
        imgGeom = imgArray.shape[0:2]

        # Greyscale for edge detection
        img_grey = cv2.cvtColor(imgArray,cv2.COLOR_BGR2GRAY)

        # Greyscale image but with 3 channels - so the bounding boxes/circles can be drawn in colour to contrast with original image
        img = np.dstack([img_grey,img_grey,img_grey])

        # Get edges as a binary image with the canny algorithm - with these threshold values, the plot and colour bar edges are reliably found while the contour lines within the plot are mostly ignored
        canny_edges = cv2.Canny(img_grey,100,200)

        # Get edges as contours (lists of points) - as just a list with no hierarchical relationships (we don't need the hierarchy) and with no approximation so all points within the contour are returned
        contours, _ = cv2.findContours(canny_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        # Find the bounding rectangle for the colour bar if needed
        if getColourbar:
            colourbar_box = self.__findColourbar__(imgGeom, contours, drawing=img)
        else:
            colourbar_box = None

        # Find the bounding circle for the plot
        plot_circle = self.__findPlot__(imgGeom, contours, drawing=img)

        # Show segmented image
        if self.showSegmentation:
            # Show with matplotlib so we can zoom
            from matplotlib import pyplot as plt
            plt.figure(), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            plt.show()

        # Form output tuple
        out = (plot_circle, colourbar_box)

        return out


    def __findPlot__(self, imgGeom, contours, drawing=None):
        '''
        Finds the bounding circle of the contour plot within the image and returns its centre and radius
        - Internal function, should not be used outside Contours class
        - imgGeom - dimensions of the image, [height, width]
        - contours - list of contours within the image obtained from call to cv2.findContours
        - drawing - 3 channel array to draw the bounding circle to, won't draw if none
        - Outputs tuple - ([centre_x, centre_y], radius)
        - Assumes that the contour plot takes up more than half of the image area and is roughly circular
        '''

        # Get image area
        image_area = imgGeom[0]*imgGeom[1]

        # Get the contours which are larger than half the whole image area - these will probably be the edges around the contour plot
        possible_contours = [contour for contour in contours if cv2.contourArea(contour) >= 0.3*image_area]

        # The smallest contour out of these will probably be the one at the edge of the plot
        plot_edge_idx = np.argmin([cv2.contourArea(contour) for contour in possible_contours])

        # Get the bounding cirlce of the plot 
        centre, radius = cv2.minEnclosingCircle(possible_contours[plot_edge_idx])

        # Draw on image if it was given - does not need to be outputted since image is passed by reference
        if drawing is not None:
            cv2.circle(drawing, (int(centre[0]), int(centre[1])), int(radius), (0,0,255),2)

        return (centre, radius)


    def __findColourbar__(self, imgGeom, contours, drawing=None):
        '''
        Finds the bounding rectangle of the colour bar within the image and returns its ____
        - Internal function, should not be used outside Contours class
        - imgGeom - image dimensions (height,width)
        - contours - list of contours within the image obtained from call to cv2.findContours
        - drawing - 3 channel array to draw the bounding circle to, won't draw if none
        - Outputs tuple - (top_left_x, top_left_y, width, height)
        - Assumes that the colour bar is below the contour plot, in the lower third of the image
        '''

        possible_contours = []
        rectangles = []
        for c in contours:
            # Get bounding rectangle
            rectangle = cv2.boundingRect(c)

            # Get centre of bounding box
            cx = int(rectangle[0]+rectangle[2]/2)
            cy = int(rectangle[1]+rectangle[3]/2)

            # Add to list if the contour is in the outer quarters of the image (opencv has y axis positive downwards)
            if (cy > 3*imgGeom[0]/4 or cy < imgGeom[0]/4 or cx > 3*imgGeom[1]/4 or cx < imgGeom[1]/4):
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


    def __getCmap__(self, imgArray, boundingbox):
        '''
        Extracts the RGB values of the levels within the colourbar
        - Internal function, should not be used outside Contours class
        - imgArray - 3 channel array of the BGR image (from opencv imread)
        - boundingbox - bounding box containing the colour bar, obtained from __findColourbar__
        '''

        # Check if colour bar is horizontal or vertical and extract list of rgb levels based on this
        if boundingbox[2] > boundingbox[3]:
            # Coordinates of start and end of colour bar - sampling at a point on the lower third of the box, less sensitivity to inaccuracies in bounding box
            colourbar_start = [boundingbox[0], boundingbox[1]+2*int(boundingbox[3]/3)]
            colourbar_end   = [boundingbox[0]+boundingbox[2], boundingbox[1]+2*int(boundingbox[3])]

            # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
            # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
            colourbar = imgArray[colourbar_start[1],colourbar_start[0]+2:colourbar_end[0]-2, :]
        else:
            # Coordinates of start and end of colour bar - sampling at a point on the left third of the box, less sensitivity to inaccuracies in bounding box
            # Also, flipped compared to above since opencv y axis is positive downwards
            colourbar_start   = [boundingbox[0]+int(boundingbox[2]/3), boundingbox[1]+boundingbox[3]]
            colourbar_end = [boundingbox[0]+int(boundingbox[2]/3), boundingbox[1]]

            # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
            # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
            colourbar = imgArray[colourbar_start[1]+2:-1:colourbar_end[1]-2,colourbar_start[0], :]

        # Normalise RGB values
        cmap = colourbar / 255

        # Transform to RGB
        cmap = np.column_stack([cmap[:,2],cmap[:,1],cmap[:,0]])

        return cmap


    def getPlotPixels(self, imgArray,boundingcircle):
        '''
        Extracts the BGR 'vectors' with their corresponding coordinates mapped to the actual inlet dimensions
        - imgArray - 3 channel RGB image containing the contour plot
        - boundingcircle - centre and radius of the bounding circle for the plot, ((centre_x,centre_y),radius), 
        can be obtained from call to __findPlot__
        - Output: [[complex coords], [R,G,B]]
        '''

        # Get image dimensions
        imgGeom = imgArray.shape[0:2]

        # Get pixel to inlet unit mapping
        pixToUnit = self.duct_radius/boundingcircle[1]

        # Make coordinate grid for the image in terms of pixels - indexing in [row,column] format to match how cv2 stores images
        Y,X = np.meshgrid(np.arange(0,imgGeom[0]), np.arange(imgGeom[1]), indexing='ij')

        # Flatten both the coordinates and the imgArray (preserving the shape of the RGB 'vectors')
        x = X.reshape(-1,1)
        y = Y.reshape(-1,1)
        pixels = imgArray.reshape(-1,3)

        # Transform coordinates to have origin coincident to plot centre and map to inlet dimension units
        x = (x-boundingcircle[0][0]) * pixToUnit
        y = (y-boundingcircle[0][1]) * pixToUnit

        # Form into complex coordinates
        coords = x + 1j * y

        # Get indices of pixels in plot
        plotIdxs = np.abs(coords) <= self.duct_radius
        plotIdxs = plotIdxs[:,0]

        # Get only the pixels in the plot
        plotPixels = pixels[plotIdxs, :]

        # Normalise the BGR values
        plotPixels = plotPixels/255

        # Get corresponding coords
        plotCoords = coords[plotIdxs]

        # Transform y coordinates - since opencv y axis is positive downwards
        plotCoords = plotCoords.real + 1j * -plotCoords.imag

        return ((plotCoords),(plotPixels))
        

    def getValuesAtNodes(self, plotPixels, cmap, nodes, range):
        '''
        Gets the numerical values associated with the contour plot at the node points
        - plotPixels - tuple containing the BGR values of the pixels and their coordinates, [[complex coords], [R,G,B]], output of getPlotPixels()
        - cmap - colour map associated with the contour plot being translated, list of RGB values of the levels
        - nodes - coordinates of points to sample
        - range - [min,max] values of the colour bar corresponding with the contour plot
        '''

        # Extract the arrays from the input tuple
        pixelCoords = plotPixels[0]
        pixelBGRs = plotPixels[1]

        # Transform colour map to BGR, to match the pixel values read by opencv
        cmap = np.column_stack([cmap[:,2],cmap[:,1],cmap[:,0]])

        # For storing the corresponding level of the pixels' colour at that node within the colourmap
        levels = np.array([None]*len(nodes))

        # Loop through the nodes
        for i,node in enumerate(nodes):
            # Get index of closest pixel to this node
            closest_idx = np.argmin(np.abs(pixelCoords - node))

            # Get BGR for this pixel
            BGR = pixelBGRs[closest_idx]
            
            # Get index of closest closer to this value from the colour map levels, by getting the Euclidean distance in a 3D space where B,G,R are the dimensions
            level_idx = np.argmin(np.linalg.norm((cmap - BGR), axis=1))

            # Normalise
            levels[i] = level_idx/cmap.shape[0]

        # Map these levels to the actual value range and output
        return np.array((levels*(range[1]-range[0]) + range[0]), dtype=float)

