import core as sg
import writeBC as bc
import maketestdomain as domain
import post
import pre
import sys


def main():
    '''
    Call all necessary functions to create a boundary condition from input data
    '''

    # Get command line options
    options = Options(sys.argv)

    # Only try to generate boundary condition if the config file has been specified
    if options.configfile is not None:
        # Initialise Input object and read config file
        inputData = pre.Input(options.configfile)

        # Intialise flow field object with coordinate system
        flowField = sg.FlowField(inputData)

        # Initialise domain configuration object with vortex definitions
        vortexDefs = sg.Vortices(inputData.vortModel, inputData.vortCoords, inputData.vortStrengths, inputData.vortRadius, inputData.axialVel)

        # Calculate velocity field
        flowField.computeDomain(vortexDefs, axialVel=inputData.axialVel)

        # Write inlet boundary condition file
        bc.writeInlet(InputObject=inputData, flowField=flowField)

        # Optional functionality
        __extra_functions(options, flowField, inputData)



class Options:
    '''
    Class for handling the command line options of this script
    - arguments - list of command line arguments, obtained from sys.argv
    '''

    def __init__(self, arguments):
        self.arguments = arguments
        self.configfile         = None
        self.checkboundaries    = False
        self.showplots          = False
        self.saveplots          = False
        self.plotsfile          = None
        self.makemesh           = False
        self.meshfile           = None
        self.showmesh           = False

        # For getting help with the command line arguements
        if (self.arguments[1] == '-help'):
            print('Usage: swirlgenerator [config file] [options]')
            print('Options:')
            print('-checkboundaries         Runs the function which checks if the boundary conditions have been satisfied')
            print('-show                    Shows the plots of the flow fields in separate windows')
            print('-saveplots [filename]    Saves the plots into a pdf file')
            print('-makemesh [filename]     Creates a meshed empty domain which is compatible with the generated inlet boundary condition defined by the config')
            print('-showmesh                Renders the mesh using GMSH GUI - beware this can be very slow with large meshes')

        else:
            self.__checkoptions(arguments)

    def __checkoptions(self, arguments):
        '''
        Checks command line arguments and sets flags appropriately
        - Internal function, should not be used outside Main.py
        - arguments - list of command line arguments extracted from sys.argv
        '''

        # Configuration file name
        if (len(arguments) < 2) or (arguments[1].find('-') == 0):
            raise RuntimeError('Configuration file missing')
        else:
            self.configfile = arguments[1]

        # Check validity of boundary conditions
        self.checkboundaries = (True if '-checkboundaries' in arguments else False)

        # Make meshed test domain
        if '-makemesh' in arguments:
            # Get index of this command line arguement
            idx = arguments.index('-makemesh')

            # Check if there is a pair
            try:
                if (arguments[idx+1].find('-') == 0):
                    raise RuntimeError
                else:
                    # Use as filename
                    self.meshfile = arguments[idx+1]
                    self.makemesh = True
            except:
                raise RuntimeError("-makemesh arguement defined but no mesh filename given")

        # Show created test mesh
        self.showmesh = (True if '-showmesh' in arguments else False)

        # Show plots
        self.showFields = (True if '-show' in arguments else False)

        # Save plots
        if '-saveplots' in arguments:
            # Get index of this command line arguement
            idx = arguments.index('-saveplots')

            # Check if there is a pair
            try:
                if (arguments[idx+1].find('-') == 0):
                    raise RuntimeError
                else:
                    # Use as filename
                    self.plotsfile = arguments[idx+1]
                    self.saveplots = True
            except:
                raise RuntimeError("-saveplots arguement defined but no pdf filename given")


def __extra_functions(options: Options, flowfield: sg.FlowField, config: pre.Input):
    '''
    Optional functionalities
    - Internal function, should not be called outside Main.py
    - options - Options object to control which functions are called
    - flowfield - FlowField object containing the flow field data
    - config - Input object with the configuration data
    '''

    # Verify boundary conditions if requested
    if options.checkboundaries:
        flowfield.checkBoundaries()

    # Create a test mesh compatible with the boundary condition that's been generated
    if options.makemesh:
        domain.testDomain(config, options.meshfile, options.showmesh)

    # Save flow fields in pdf if requested
    if options.saveplots:
        post.plotAll(flowfield, options.plotsfile)

    # Show flow fields if requested
    if options.showFields:
        post.plotAll(flowfield)


if __name__ == '__main__':
    main()
