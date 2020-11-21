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
        # Intiailise all possible variables first
        self.filename = None
        self.format = None
        self.shape = None
        self.radius = None
        self.curveNumCells = None
        self.radialNumCells = None
        self.xSide = None
        self.ySide = None
        self.zSide = None
        self.xNumCells = None
        self.yNumCells = None
        self.zNumCells = None
        self.vortModel = None
        self.vortCoords = []
        self.vortStrengths = []
        self.vortRadius = []
        self.axialVel = None
        self.meshfilename = None

        # Read in the config file on initialisation of the object since it has no other functionality anyway
        self.read(configfile)


    def read(self, configFile):
        # Initialise config parser and read config file
        config = ConfigParser()
        config.read(configFile)

        # Check which sections are present

        if ('METADATA' in config):
            # Get section
            metadata = config['METADATA']

            # Supported formats 
            formats = ['su2']

            try:
                # Get name for output boundary condition file
                self.filename = metadata.get('filename')
                
                # Get format to write the boundary condition in
                format = metadata.get('format')
                # Check if supported format
                if (format in formats):
                    self.format = format
                else:
                    raise NotImplementedError(f"{format} not supported")
                
                # Get the mesh filename to read the inlet node coordinates from (also the filename to write a generated mesh to)
                self.meshfilename = metadata.get('mesh')

            except KeyError:
                raise KeyError(f"Non-optional metadata missing in file {configFile}")

        else:
            raise RuntimeError(f"Metadata section missing in file {configFile}")



        if ('MESH DEFINITION' in config):
            # Get section
            meshDefinitions = config['MESH DEFINITION']

            # Get specified inlet shape
            try:
                self.shape = meshDefinitions.get('shape')
            except:
                raise KeyError("Shape of inlet face must be specified")

            try:
                # Get mesh length parameters
                self.zSide = float(meshDefinitions.get('z_side'))
                self.zNumCells = int(meshDefinitions.get('z_num_cells'))

                # Get necessary inputs for inlet shape
                if self.shape == 'circle':
                    # Get circle radius
                    self.radius = float(meshDefinitions.get('radius'))
                    # Get mesh density
                    self.curveNumCells = int(meshDefinitions.get('quadrant_num_cells'))
                    self.radialNumCells = int(meshDefinitions.get('radial_num_cells'))
                    
                elif self.shape == 'rect':
                    # Get side lengths
                    self.xSide = float(meshDefinitions.get('x_side'))
                    self.ySide = float(meshDefinitions.get('y_side'))
                    # Get mesh density
                    self.xNumCells = int(meshDefinitions.get('x_num_cells'))
                    self.yNumCells = int(meshDefinitions.get('y_num_cells'))
                
                else:
                    raise NotImplementedError("Specified inlet shape not valid")

            # Catch errors and print appropriate helpful error messages
            except KeyError:
                raise KeyError(f"Non-optional geometry/mesh parameters are missing in file {configFile} for {self.shape} inlet shape")
            except ValueError:
                raise ValueError("Invalid values defined for mesh/geometry")


        if ('VORTEX DEFINITIONS' in config):
            # Get section
            vortexDefs = config['VORTEX DEFINITIONS']

            # Get number of vortices defined
            numVortices = sum(1 for key in vortexDefs) - 1

            # Check present inputs
            try:
                self.vortModel = vortexDefs.get('vortex_model').lower()
            except KeyError:
                raise KeyError(f"Non-optional vortex parameters are missing in file {configFile}")

            if (numVortices > 0):
                try:
                    # Extract the numeric data from the string for each vortex into an array
                    for i in range(1,numVortices+1):
                        data = list(float(numString) for numString in vortexDefs.get(f"vortex{i}")[1:-1].split(','))

                        if (len(data) < 4):
                            raise SyntaxError(f"Invalid number of parameters when defining vortex {i}")

                        self.vortCoords.append(data[0:2])
                        self.vortStrengths.append(data[2])
                        self.vortRadius.append(data[3])

                except ValueError:
                    raise ValueError(f"Invalid values defined for vortex parameters")
            else:
                raise KeyError(f"At least one vortex needs to be defined in {configFile}")
        else:
            raise ValueError(f"Non-optional vortex definitions section not present in file {configFile}")

        # Optional section
        if ('EXTRA' in config):
            # Get section
            extraParams = config['EXTRA']

            # May need better solution than this in future since will need a try/except pair for each optional config
            try:
                self.axialVel = float(extraParams.get('axial_vel'))
            except:
                pass

        # Set defaults if values weren't set
        self.axialVel   = (1.0 if self.axialVel is None else self.axialVel)


    def getNodes(self):
        '''
        Extracts the coordinates of the nodes which make up the inlet from an input mesh file
        '''

        # Read in all lines from the file so we can index them in any order
        try:
            with open(self.meshfilename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Specificed mesh file {self.meshfilename} was not found")

        # Get index of where the inlet boundary definition starts
        inletIdx = [i for [i,line] in enumerate(lines) if 'inlet' in line.lower()]

        # Get index of where the point definitions start
        pointIdx = [i for [i,line] in enumerate(lines) if 'NPOIN' in line]

        # Check validity of mesh file by seeing if the above lines were found
        if not inletIdx or not pointIdx:
            raise RuntimeError('Invalid mesh file')
        # Then we can extract the integers from the arrays
        inletIdx = inletIdx[0]
        pointIdx = pointIdx[0]

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
        nodes = [np.array(line).astype(float) for line in np.char.split(nodes)]             # Returns list of numpy arrays
        # Stack into a single array and extract only first two coords - assume that this is x,y and that z is 0
        nodes = np.vstack(nodes)[:,0:2]

        # Return as an array of complex numbers
        return nodes[:,0] + 1j * nodes[:,1]