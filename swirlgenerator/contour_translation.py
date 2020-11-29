# -------------------------------------------------------------------------------------------------------------------
#
# Module for translating images containing contour plots and colour bars into values
# which can then be used to reconstruct the original flow field
#
# -------------------------------------------------------------------------------------------------------------------

from matplotlib.pyplot import plot
import numpy as np
import cv2

class Contours:
    '''
    Class for processing images conatining the flow angles as contour plots
    - imgFile1 - RGB image file containing the contour plot of the tangential flow angle field
    - imgFile2 - RGB image file containing the contour plot of the radial flow angle field
    - duct_radius - actual radius of the circular inlet
    - ranges - set of [min, max] values which corresponds to the colorbar ranges of the two contour plots; size (2,2)
    - nodes - complex coordinates of nodes on the inlet plane (extracted from mesh)
    '''
    
    def __init__(self,imgFile1,imgFile2,duct_radius,ranges,nodes):
        
        # Extract variables into object
        self.imgArrays = (cv2.imread(imgFile1), cv2.imread(imgFile2))
        self.ranges = ranges
        self.duct_radius = duct_radius
        self.nodes = nodes


    def getFlowField(self, axialVel):
        '''
        Reconstructs the velocity field at the inlet using the tangential and radial flow angle contour plots
        - axialVel - mean axial velocity of the bulk flow
        '''

        values = [None, None]
        # Extract values from both contour plots
        for i, imgArray in enumerate(self.imgArrays):
            # Get the bounding circle/box of the contour plot and colour bar
            boundaries = self.segmentImage(imgArray)

            # Get the pixels within the plot and their coords in terms of the inlet dimensions
            plotPixels = self.getPlotPixels(imgArray, boundaries[0])

            # Extract the colourmap associated to this plot from the image
            cmap = self.__getCmap__(imgArray, boundaries[1])

            # Get actual values at nodes
            values[i] = self.getValuesAtNodes(plotPixels, cmap, self.nodes, self.ranges[i])

        # Extract from list
        tangential_flow_angle = np.deg2rad(values[0])
        radial_flow_angle = np.deg2rad(values[1])

        # Get velocities in polar coordinate system
        vel_theta = np.tan(tangential_flow_angle)*axialVel
        r_dot = np.tan(radial_flow_angle)*axialVel

        # Get polar coordinates of nodes
        radius = np.abs(self.nodes)
        theta = np.arctan2(self.nodes.imag,self.nodes.real)

        # Get theta dot
        theta_dot = vel_theta/radius

        # Get cartesian velocities
        u = r_dot*np.cos(theta) - radius*theta_dot*np.sin(theta)
        v = r_dot*np.sin(theta) + radius*theta_dot*np.cos(theta)

        return np.column_stack([u,v])


    def segmentImage(self, imgArray, showresult=False):
        '''
        Segments the input image and returns data on the bounding circle of the plot and the bounding rectangle of the colour bar 
        - imgArray - numpy array of the 3 channel BGR image
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

        # Find the bounding rectangle for the colour bar and extract the colour map RGB
        colourbar_box = self.__findColourbar__(imgGeom, contours, drawing=img)

        # Find the bounding circle for the plot
        plot_circle = self.__findPlot__(imgGeom, contours, drawing=img)

        # Show segmented image
        if showresult:
            # Show with matplotlib so we can zoom
            from matplotlib import pyplot as plt
            plt.figure(), plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
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
        possible_contours = [contour for contour in contours if cv2.contourArea(contour) >= 0.5*image_area]

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
        - Assumes that the colour bar is below the contour plot, in the lower half of the image
        '''

        possible_contours = []
        rectangles = []
        for c in contours:
            # Get bounding rectangle
            rectangle = cv2.boundingRect(c)

            # Get centre of bounding box
            cx = int(rectangle[0]+rectangle[2]/2)
            cy = int(rectangle[1]+rectangle[3]/2)

            # Add to list if the contour is in the lower half of the image
            if cy > imgGeom[0]/2:
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
        - imgArray - 3 channel array of the BGR image
        - boundingbox - bounding box containing the colour bar, obtained from __findColourbar__
        '''

        # Coordinates of start and end of colour bar - sampling at a point on the lower third of the box, less sensitivity to inaccuracies in bounding box
        colourbar_start = [boundingbox[0], boundingbox[1]+2*int(boundingbox[3]/3)]
        colourbar_end   = [boundingbox[0]+boundingbox[2], boundingbox[1]+2*int(boundingbox[3])]

        # Get list of RGB values for colourmap - by getting the pixels in the colourbar, truncate start an end to leave out the black boundary
        # Depending on the accuracy of the opencv bounding box, this colourmap could be only an approximation of the actual range
        colourbar = imgArray[colourbar_start[1],colourbar_start[0]+2:colourbar_end[0]-2, :]

        # Normalise RGB values
        cmap = colourbar / 255

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
        

    def getValuesAtNodes(self, plotPixels, cmap_BGR, nodes, range):
        '''
        Gets the numerical values associated with the contour plot at the node points
        - plotPixels - tuple containing the BGR values of the pixels and their coordinates, [[complex coords], [R,G,B]], output of getPlotPixels()
        - cmap_BGR - colour map associated with the contour plot being translated, list of BGR values of the levels
        - nodes - coordinates of points to sample
        - range - [min,max] values of the colour bar corresponding with the contour plot
        '''

        # Extract the arrays from the input tuple
        pixelCoords = plotPixels[0]
        pixelBGRs = plotPixels[1]

        # For storing the corresponding level of the pixels' colour at that node within the colourmap
        levels = np.array([None]*len(nodes))

        # Loop through the nodes
        for i,node in enumerate(nodes):
            # Get index of closest pixel to this node
            closest_idx = np.argmin(np.abs(pixelCoords - node))

            # Get BGR for this pixel
            BGR = pixelBGRs[closest_idx]
            
            # Get index of closest closer to this value from the colour map levels, by getting the Euclidean distance in a 3D space where B,G,R are the dimensions
            level_idx = np.argmin(np.linalg.norm((cmap_BGR - BGR), axis=1))

            # Normalise
            levels[i] = level_idx/cmap_BGR.shape[0]

        # Map these levels to the actual value range and output
        return np.array((levels*(range[1]-range[0]) + range[0]), dtype=float)

