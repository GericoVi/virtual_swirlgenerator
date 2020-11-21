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

        # Create a test meshed geometry based on user inputs if requested - node coordinates of flowfield object taken from the inlet of this mesh
        if options.makemesh:
            # Throw error if mesh generation requested but no filename specified
            if inputData.meshfilename is None:
                raise RuntimeError("Mesh generation requested but no filename specified in config")
            else:
                domain.testDomain(inputData, inputData.meshfilename, options.showmesh)

        # Intialise flow field object with coordinate system
        flowfield = sg.FlowField(inputData)

        # Initialise domain configuration object with vortex definitions
        vortexDefs = sg.Vortices(inputData.vortModel, inputData.vortCoords, inputData.vortStrengths, inputData.vortRadius, inputData.axialVel)

        # Calculate velocity field
        flowfield.computeDomain(vortexDefs, axialVel=inputData.axialVel)

        # Verify boundary conditions if requested
        if options.checkboundaries:
            flowfield.checkBoundaries()

        # Write inlet boundary condition file
        bc.writeInlet(InputObject=inputData, flowField=flowfield)

        # Initialise plotting object
        plots = post.Plots(flowfield)

        # Save flow fields in pdf if requested - name the pdf the same as the boundary condition .dat file
        if options.saveplots:
            pdfname = options.configfile.split()[0]
            plots.plotAll(pdfname)

        # Show flow fields if requested
        if options.showFields:
            plots.plotAll()



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
        self.makemesh           = False
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
            self.__checkoptions__(arguments)

    def __checkoptions__(self, arguments):
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

        # Make a simple meshed geometry to test the boundary condition
        self.makemesh = (True if '-makemesh' in arguments else False)

        # Show created test mesh
        self.showmesh = (True if '-showmesh' in arguments else False)

        # Show plots
        self.showFields = (True if '-show' in arguments else False)

        # Save plots
        self.saveplots = (True if '-saveplots' in arguments else False)


if __name__ == '__main__':
    main()
