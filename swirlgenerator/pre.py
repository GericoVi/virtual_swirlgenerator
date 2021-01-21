# -------------------------------------------------------------------------------------------------------------------
#
# Module for reading inputs and pre-processing into formats useable by the core module
#
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
from configparser import ConfigParser

class Input:
    '''
    Class for reading and storing the information in the config file
    - configfile - name of config file to read from when the object is initialised
    - Full object is passed in and used in other core classes
    '''

    def __init__(self, configfile):
        # Section flags
        self.metadata_flag = False
        self.vortDefs_flag = False
        self.contours_flag = False
        self.mesh_flag     = False
        self.extra_flag    = False

        # Intiailise all possible variables
        # Metadata
        self.filename = None
        self.format = None
        self.meshfilename = None

        # Vortex definitions
        self.numVortices = None
        self.vortModel = None
        self.vortCoords = []
        self.vortStrengths = []
        self.vortRadius = []

        # Contour translation inputs
        self.tanImg = None
        self.tanRng = []
        self.tancmap = None
        self.radImg = None
        self.radRng = []
        self.tancmap = None
        self.numRings = None
        self.angRes = None

        # Extra parameters
        self.axialVel = None
        self.swirlPlotRange = [None,None]
        self.swirlPlotNTicks = None

        # Read in the config file on initialisation of the object since it has no other functionality anyway
        self.read(configfile)


    def read(self, configFile):
        # Initialise config parser and read config file
        config = ConfigParser()

        config.read(configFile)

        # Check which sections are present
        self.metadata_flag = 'METADATA' in config
        self.vortDefs_flag = 'VORTEX DEFINITIONS' in config
        self.contours_flag = 'CONTOUR TRANSLATION' in config
        self.mesh_flag     = 'MESH DEFINITION' in config 
        self.extra_flag    = 'EXTRA' in config

        if self.metadata_flag:
            # Get section
            metadata = config['METADATA']

            try:
                # Get name for output boundary condition file
                self.filename = metadata.get('filename')
                
                # Get format to write the boundary condition in
                self.format = metadata.get('format')
                
                # Get the mesh filename to read the inlet node coordinates from (also the filename to write a generated mesh to)
                self.meshfilename = metadata.get('mesh')
            except:
                pass


        if self.vortDefs_flag:
            # Get section
            vortexDefs = config['VORTEX DEFINITIONS']

            # Get number of vortices defined
            self.numVortices = sum(1 for _ in vortexDefs) - 1

            # Check present inputs
            try:
                self.vortModel = vortexDefs.get('vortex_model').lower()
            except:
                pass

            if (self.numVortices > 0):
                try:
                    # Extract the numeric data from the string for each vortex into an array
                    for i in range(1,self.numVortices+1):
                        data = list(float(numString) for numString in vortexDefs.get(f"vortex{i}")[1:-1].split(','))

                        if (len(data) < 4):
                            raise SyntaxError(f"Invalid number of parameters when defining vortex {i}")

                        self.vortCoords.append(data[0:2])
                        self.vortStrengths.append(data[2])
                        self.vortRadius.append(data[3])

                    # Convert to numpy arrays
                    self.vortCoords = np.array(self.vortCoords)
                    self.vortStrengths = np.array(self.vortStrengths)
                    self.vortRadius = np.array(self.vortRadius)

                except ValueError:
                    raise ValueError(f"Invalid values defined for vortex parameters")


        if self.contours_flag:
            # Get section
            contours = config['CONTOUR TRANSLATION']

            # Get present inputs
            try:
                self.tanImg = contours.get('tan_img')
                self.tanRng = list(float(numString) for numString in contours.get('tan_range')[1:-1].split(','))
                self.tancmap = contours.get('tan_cmap')
            except:
                pass
            
            # Get information on the radial flow plot image if they are there
            try:
                self.radImg = contours.get('rad_img')
                self.radRng = list(float(numString) for numString in contours.get('rad_range')[1:-1].split(','))
                self.radcmap = contours.get('rad_cmap')
            except:
                pass

            # Get information on the sampling distribution if it is there
            try:
                self.numRings = int(contours.get('num_rings'))
                self.angRes = float(contours.get('ang_res'))
            except:
                pass

            # If colour map name has been defined, get values
            if self.tancmap is not None:
                self.tancmap = Input.getCmapValues(self.tancmap)
            if self.radcmap is not None:
                self.radcmap = Input.getCmapValues(self.radcmap)


        # Optional section
        if self.extra_flag:
            # Get section
            extraParams = config['EXTRA']

            # May need better solution than this in future since will need a try/except pair for each optional config
            try:
                self.axialVel = float(extraParams.get('axial_vel'))
            except:
                pass
            try:
                self.swirlPlotRange = list(float(numString) for numString in extraParams.get('swirl_contour_range')[1:-1].split(','))
            except:
                pass
            try:
                self.swirlPlotNTicks = int(extraParams.get('swirl_colorbar_num_ticks'))
            except:
                pass


    def getNodes(self):
        '''
        Wrapper utility function. Just calls the static method, but from the object instance
        '''
        return self.extractMesh(self.meshfilename)


    @classmethod
    def extractMesh(cls, meshfilename):
        '''
        Extracts the coordinates of the nodes which make up the inlet from an input mesh file. Wrapper to call correct method based on file format
        - meshfilename - Name of the mesh file to extract the inlet node coordinates from
        '''

        # Supported formats
        formatMap = {'su2': cls.readSU2mesh}

        # Get file extension to determine format
        format = meshfilename.split('.')[-1]

        # Read in all lines from the file\
        try:
            with open(meshfilename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Specificed mesh file {meshfilename} was not found")

        # Call correct method based on file format
        try:
            func = formatMap.get(format)
            return func(lines)
        except KeyError:
            raise RuntimeError(f'.{format} file extension not supported for input meshes')


    @staticmethod
    def readSU2mesh(lines):
        '''
        Finds the inlet nodes and extracts their coordinates from text extracted from a .su2 mesh file
        - lines - all lines from the file
        '''

        inletIdx = None
        pointIdx = None
        fileValid = False
        # Loop through lines in file to get the indices of where the inlet boundary and the points definitions start
        for i,line in enumerate(lines):
            if 'inlet' in line.lower():
                inletIdx = i
            if 'NPOIN' in line:
                pointIdx = i
            
            if inletIdx is not None and pointIdx is not None:
                fileValid = True
                break

        # Check validity of mesh file by seeing if the above lines were found
        if not fileValid:
            raise RuntimeError('Invalid mesh file')

        # Get number of elements/lines to read in next line
        numElem = int(lines[inletIdx+1].split()[1])

        # Get all elements listed within the inlet tag and store as a numpy array so we can do vectorised operations
        inletElements = np.array(lines[inletIdx+2:inletIdx+2+numElem])

        # Split each string to isolate the numbers, store this list of digit strings as a numpy array and convert to integers at the same time
        nodeIdxs = [np.array(line).astype(int) for line in np.char.split(inletElements)]    # This is an list of numpy arrays
        # Stack the numpy arrays into one big array and cut off the first elements (since this is the id of the element type which we don't need)
        nodeIdxs = np.vstack(nodeIdxs)[:,1:]
        # Now we flatten this array and remove the duplicate values (each node will show up between 2 to 4 times since they are vertices of quadrilaterals)
        nodeIdxs = np.unique(nodeIdxs.flatten())

        # Get all lines corresponding to the nodes in the inlet
        nodes = np.array([lines[pointIdx + i+1] for i in nodeIdxs])

        # Split each string to isolate the numbers, and convert to floats
        nodes = [np.array(line[:3]).astype(float) for line in np.char.split(nodes)]             # Returns list of numpy arrays
        
        # Stack into a single array
        nodes = np.vstack(nodes)

        # Return array of vectors - 3D coordinate of each node
        return nodes


    @staticmethod
    def getCmapValues(cmapName):
        # Do import here, not needed anywhere else
        from matplotlib.cm import get_cmap

        try:
            cmap = get_cmap(cmapName)
            cmapValues = cmap(range(cmap.N))[:,0:3]
        except ValueError:
            raise ValueError(f'Invalid colourmap name \'{cmapName}\'. See https://matplotlib.org/gallery/color/colormap_reference.html?highlight=colormap%20reference for list of available')

        return cmapValues

