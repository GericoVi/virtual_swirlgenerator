# -------------------------------------------------------------------------------------------------------------------
#
# Module for reading inputs and pre-processing into formats useable by the core module
#
# -------------------------------------------------------------------------------------------------------------------

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
                self.filename = metadata.get('filename')
                
                format = metadata.get('format')
                if (format in formats):
                    self.format = format
                else:
                    raise NotImplementedError(f"{format} not supported")
            except KeyError:
                raise KeyError(f"Non-optional matadata missing in file {configFile}")

        if ('MESH DEFINITION' in config):
            # Get section
            meshDefinitions = config['MESH DEFINITION']

            # Get specified inlet shape
            try:
                self.shape = meshDefinitions.get('shape')
            except:
                raise KeyError("Shape of inlet face must be specified")

            try:
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
                raise KeyError(f"Non-optional geometry/mesh parameters are missing in file {configFile}")
            except ValueError:
                raise ValueError("Invalid values defined for mesh/geometry")

            # Optional parameters
            if ('z_side' in meshDefinitions):
                self.zSide = float(meshDefinitions.get('z_side'))

            if ('z_num_cells' in meshDefinitions):
                self.zNumCells = int(meshDefinitions.get('z_num_cells'))

        else:
            raise ValueError(f"Non-optional mesh definitions section not present in file {configFile}")

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