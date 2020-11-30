import core as sg
import writeBC as bc
import maketestdomain as domain
import post
import pre
import contour_translation as ct
import sys


def main():
    '''
    Call all necessary functions to create a boundary condition from input data
    '''

    # Get command line options
    options = Options(sys.argv)

    # Only try to generate boundary condition if the config file has been specified
    if options.configfile is not None:
        # Initialise some things for later
        tangential = None

        # Initialise Input object and read config file
        inputData = pre.Input(options.configfile)

        # Check the inputs and set correct mode flags
        inputData = options.checkInputs(inputData)

        # Create a test meshed geometry based on user inputs if requested - node coordinates of flowfield object taken from the inlet of this mesh
        if options.makemesh:
            # Throw error if mesh generation requested but no filename specified
            if inputData.meshfilename is None:
                raise RuntimeError("Mesh generation requested but no filename specified in config")
            else:
                domain.testDomain(inputData, inputData.meshfilename, options.showmesh)

        # Intialise flow field object with coordinate system
        flowfield = sg.FlowField(inputData.getNodes())

        # Create flow field with vortex method
        if not options.reconstruct:
            # Initialise domain configuration object with vortex definitions
            vortexDefs = sg.Vortices(inputObject=inputData)

            # Calculate velocity field from effect of vortices
            flowfield.computeDomain(vortexDefs, axialVel=inputData.axialVel)

        # Reconstruct flow field from contour plots
        else:
            # Translate contour plot images to array of values
            tangential = ct.Contour(inputData.tanImg, inputData.tanRng, flowfield.coords)
            radial = ct.Contour(inputData.radImg, inputData.radRng, flowfield.coords)

            # Calculate velocity field from flow angles
            flowfield.reconstructDomain(tangential.values, radial.values, inputData.axialVel)


        # Get RMSE between calculated swirl angle field and that estimated from contour plot image if requested
        if options.validate:
            # Get the swirl angles from a contour plot if haven't already done so
            if tangential is None:
                tangential = ct.Contour(inputData.tanImg, inputData.tanRng, flowfield.coords)

            print(f'RMSE compared to estimated plot values: {flowfield.getError(tangential.values)}')

        # Verify boundary conditions if requested
        if options.checkboundaries:
            flowfield.checkBoundaries()

        # Write inlet boundary condition file
        bc.writeInlet(InputObject=inputData, flowField=flowfield)

        # Initialise plotting object
        plots = post.Plots(flowfield)

        # Save flow fields in pdf if requested - name the pdf the same as the boundary condition .dat file
        if options.saveplots:
            pdfname = options.configfile.split('.')[0]
            plots.plotAll(pdfName=f'{pdfname}.pdf', swirlAxisRange=inputData.swirlPlotRange, swirlAxisNTicks=inputData.swirlPlotNTicks)

        # Show flow fields if requested
        if options.showFields:
            plots.plotAll(swirlAxisRange=inputData.swirlPlotRange, swirlAxisNTicks=inputData.swirlPlotNTicks)



class Options:
    '''
    Class for handling the command line options of this script
    - arguments - list of command line arguments, obtained from sys.argv
    '''

    def __init__(self, arguments):
        self.arguments = arguments
        self.configfile         = None

        # Command line options
        self.checkboundaries    = False
        self.showplots          = False
        self.saveplots          = False
        self.makemesh           = False
        self.showmesh           = False

        # Options based on the inputs in config file
        self.reconstruct = False            # Are we reconstructing the flow field from contour plots or creating from discrete vortices
        self.validate = False               # Are we getting the error between the created flow field and a swirl angle contour plot

        # For getting help with the command line arguements
        if ('-help' in self.arguments[1]):
            print('Usage: swirlgenerator [config file] [options]')
            print('Options:')
            print('-reconstruct             Reconstructs flow field from input contour plot images rather than vortex method - overrides priority')
            print('-validate                Gets the RMSE between the calculated swirl angle and the estimated values from a contour plot image')
            print('-checkboundaries         Runs the function which checks if the boundary conditions have been satisfied')
            print('-show                    Shows the plots of the flow fields in separate windows')
            print('-saveplots               Saves the plots into a pdf file with the same name as the config file')
            print('-makemesh                Creates a meshed empty domain with the parameters defined in the config file')
            print('-showmesh                Renders the mesh using GMSH GUI - beware this can be very slow with large meshes')

        else:
            self.__checkargs__()


    def __checkargs__(self):
        '''
        Checks command line arguments and sets flags appropriately
        - Internal function, should not be used outside Main.py
        - arguments - list of command line arguments extracted from sys.argv
        '''

        # Configuration file name
        if (len(self.arguments) < 2) or (self.arguments[1].find('-') == 0):
            raise RuntimeError('Configuration file missing')
        else:
            self.configfile = self.arguments[1]

        # Override priority and use reconstruction method even if domain vortices have been defined
        self.reconstruct = (True if '-reconstruct' in self.arguments else False)

        # Override and validate the calculated swirl angle with the estimated values translated from a contour plot
        self.validate = (True if '-validate' in self.arguments else False)

        # Check validity of boundary conditions
        self.checkboundaries = (True if '-checkboundaries' in self.arguments else False)

        # Make a simple meshed geometry to test the boundary condition
        self.makemesh = (True if '-makemesh' in self.arguments else False)

        # Show created test mesh
        self.showmesh = (True if '-showmesh' in self.arguments else False)

        # Show plots
        self.showFields = (True if ('-show' in self.arguments or '-showplots' in self.arguments) else False)

        # Save plots
        self.saveplots = (True if '-saveplots' in self.arguments else False)


    def checkInputs(self, inputdata: pre.Input):
        '''
        Checks which inputs are present from the config file and activates the correct flags for controlling 
        - inputdata - Input object containing the fields read from the config file
        - Output - returns the inputdata object in case we modified it in here
        '''

        # Formats supported for writing boundary conditions
        formats = ['su2']

        # Valid input shapes
        inlet_shapes = ['circle', 'rect', 'square']

        # These metadata fields are necessary for using any functionality
        if not inputdata.metadata_flag:
            raise RuntimeError(f'\nMetadata missing in file {self.configfile}')
        if (inputdata.filename is None or inputdata.format is None or inputdata.meshfilename is None):
            raise RuntimeError(f'\nNon-optional metadata missing in file {self.configfile}')

        # Check format specified
        if (inputdata.format not in formats):
            raise NotImplementedError(f"\n{format} not supported")

        # Check if vortices have been defined
        if inputdata.vortDefs_flag:
            # Check vortex definitions
            if inputdata.vortModel is None:
                raise RuntimeError(f'\nVortex model to be used was not defined in {self.configfile}')

            if inputdata.numVortices < 1:
                raise RuntimeError(f'\nAt least one vortex needs to be defined in {self.configfile}')

        # Check if contour plot section is present
        if inputdata.contours_flag:
            # Creation of flow field by vortex method has priority
            if inputdata.vortDefs_flag:
                # If no command line overrides occured, only the tangential flow angle plot will be used to validate the flow field created with vortex method
                self.validate = True
            # If actually reconstructing, we need an extra plot
            else:
                self.reconstruct = True
        
        # Check if reconstruction mode
        if (self.reconstruct and (inputdata.tanImg is None or inputdata.tanRng is None or inputdata.radImg is None or inputdata.radRng is None)):
            raise RuntimeError(f'\nRequired flow angle plot information missing in {self.configfile} for flow field reconstruction')

        # Check if validation mode
        if (self.validate and (inputdata.radImg is None or inputdata.radRng is None)):
            raise RuntimeError(f'\nRequired flow angle plot information missing in {self.configfile} for flow field validation')

        # Check if mesh properties have been defined when needed
        if (self.makemesh and not inputdata.mesh_flag):
            raise RuntimeError(f'\n-makemesh option specified but no mesh properties defined in {self.configfile}')

        if inputdata.mesh_flag:
            if inputdata.shape is None:
                raise RuntimeError(f'\nShape of inlet face must be specified in {self.configfile}')
            if inputdata.shape not in inlet_shapes:
                raise RuntimeError(f'\nSpecified inlet shape \'{inputdata.shape}\' in {self.configfile} is not valid')

            # Check available mesh information for specific geometries
            if (inputdata.shape == 'circle' and (inputdata.radius is None or inputdata.curveNumCells is None or inputdata.radialNumCells is None)):
                raise RuntimeError(f'\nMissing mesh information for a circular inlet in {self.configfile}')

            if ((inputdata.shape == 'rect' or inputdata.shape == 'square') and (inputdata.xSide is None or inputdata.ySide is None or inputdata.xNumCells is None or inputdata.yNumCells is None)):
                raise RuntimeError(f'\nMissing mesh information for a rectangular inlet in {self.configfile}')

        # Set defaults
        inputdata.axialVel = (1.0 if inputdata.axialVel is None else inputdata.axialVel)

        return inputdata


if __name__ == '__main__':
    main()
