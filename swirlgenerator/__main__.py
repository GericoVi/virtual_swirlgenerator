import core as sg
import writeBC as bc
import post
import pre
import contour_translation as ct
import sys
import os


def main():
    '''
    Call all necessary functions to create a boundary condition from input data
    '''

    # Get command line options
    options = Options(sys.argv)

    # Only do this when config file specified - ie not just using '-help'
    if options.configfile is not None:
        # Initialise Input object and read config file
        inputData = pre.Input(options.configfile)

        # Check the inputs and set correct mode flags
        inputData = options.checkInputs(inputData)
        
        if not options.exit:
            tangential = None

            # Create a test meshed geometry based on user inputs if requested - node coordinates of flowfield object taken from the inlet of this mesh
            if options.makemesh:
                print('Generating mesh...')
                # Do import here so user won't need the dependencies if they don't need this functionality
                import maketestdomain as domain

                if inputData.meshfilename is None:
                    raise RuntimeError("Mesh generation requested but no filename specified in config")
                else:
                    domain.testDomain(inputData, inputData.meshfilename, options.showmesh)
                    print(f"Mesh file written to {inputData.meshfilename}")


            # Intialise flow field object with coordinate system
            print(f'Extracting inlet nodes from {inputData.meshfilename}...')
            flowfield = sg.FlowField(inputData.getNodes())


            # Create flow field with vortex method
            if not options.reconstruct:
                print('Calculating flow field...')
                # Initialise domain configuration object with vortex definitions
                vortexDefs = sg.Vortices(inputObject=inputData)

                # Calculate velocity field from effect of vortices
                flowfield.computeDomain(vortexDefs, axialVel=inputData.axialVel)


            # Reconstruct flow field from contour plots
            else:
                print('Reconstructing flow field...')
                # Translate contour plot images to array of values
                tangential = ct.Contour(inputData.tanImg, inputData.tanRng, cmap=inputData.tancmap)
                radial = ct.Contour(inputData.radImg, inputData.radRng, cmap=inputData.radcmap)

                # Calculate velocity field from flow angles mapped to flowfield nodes
                flowfield.reconstructDomain(tangential.getValuesAtNodes(flowfield.coords), radial.getValuesAtNodes(flowfield.coords), inputData.axialVel)


            # Get RMSE between calculated swirl angle field and that estimated from contour plot image if requested
            if options.validate:
                if tangential is None:
                    tangential = ct.Contour(inputData.tanImg, inputData.tanRng, cmap=inputData.tancmap)

                values = tangential.getValuesAtNodes(flowfield.coords)
                print(f'RMSE compared to estimated plot values: {flowfield.getError(values)}')


            # Verify boundary conditions if requested
            if options.checkboundaries:
                flowfield.checkBoundaries()


            # Write inlet boundary condition file
            print('Writing boundary condition...')
            bc.writeInlet(InputObject=inputData, flowField=flowfield)
            print(f'Inlet BC written to {inputData.filename}')


            if options.plot:
                # Initialise plotting object
                print('Creating plots...')
                plots = post.Plots(flowfield)

                # Save flow fields in pdf if requested - name the pdf the same as the boundary condition .dat file
                if options.saveplots:
                    pdfname = options.configfile.split('.')[0]
                    pdfname = f'{pdfname}.pdf'
                    plots.plotAll(pdfName=pdfname, swirlAxisRange=inputData.swirlPlotRange, swirlAxisNTicks=inputData.swirlPlotNTicks)
                    print(f'Figures saved to {pdfname}')

                # Show flow fields if requested
                if options.showFields:
                    plots.plotAll(swirlAxisRange=inputData.swirlPlotRange, swirlAxisNTicks=inputData.swirlPlotNTicks)

                # Show inlet nodes if requested
                if options.showinletnodes:
                    plots.showInletNodes()

        else:
            print('Exiting...')


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
        self.showinletnodes     = False

        # Options based on the inputs in config file
        self.reconstruct = False            # Are we reconstructing the flow field from contour plots or creating from discrete vortices
        self.validate = False               # Are we getting the error between the created flow field and a swirl angle contour plot

        # Misc flags
        self.exit = False
        self.plot = False

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
            print('-showinletnodes          For plotting the inlet nodes - for confirming correct extraction of nodes from mesh')

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

        # Check if config file exists
        if not os.path.exists(self.configfile):
            raise RuntimeError(f'Configuration file {self.configfile} not found')

        # Override priority and use reconstruction method even if domain vortices have been defined
        self.reconstruct = (True if '-reconstruct' in self.arguments else False)

        # Validate the calculated swirl angle with the estimated values translated from a contour plot
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

        # Show inlet nodes
        self.showinletnodes = (True if '-showinletnodes' in self.arguments else False)

        if self.showFields or self.saveplots or self.showinletnodes:
            self.plot = True


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

        # Check if reconstruction mode
        if (self.reconstruct and (inputdata.tanImg is None or inputdata.tanRng is None or inputdata.radImg is None or inputdata.radRng is None)):
            raise RuntimeError(f'\nRequired flow angle plot information missing in {self.configfile} for flow field reconstruction')

        # Check if validation mode
        if (self.validate and (inputdata.radImg is None or inputdata.radRng is None)):
            raise RuntimeError(f'\nRequired flow angle plot information missing in {self.configfile} for flow field validation')

        # Check if contour plot section is present
        if inputdata.contours_flag:
            # If both vortex definition and contour translation section present and no command line overrside given, user needs to pick preferred method
            if inputdata.vortDefs_flag and not self.reconstruct:
                while True:
                    choice = input(f'Vortex definitions section and contour translation section both present in {self.configfile}.\nChoose flow field calculation method (V/R):')

                    if choice.lower() == 'r':
                        self.reconstruct =  True
                        break
                    elif choice.lower() == 'v':
                        self.reconstruct = False
                        break
                    else:
                        print('Invalid choice')
                        continue

            # If only contour translation section is present, then assume we want to do reconstruction
            elif not inputdata.vortDefs_flag:
                self.reconstruct = True

            # See if colourmap names have been defined and turn them into actual RGB arrays
            if ((self.reconstruct or self.validate) and (inputdata.tancmap is not None or inputdata.radcmap is not None)):
                # Do import here, not needed anywhere else
                from matplotlib.cm import get_cmap

                if inputdata.tancmap is not None:
                    try:
                        cmap = get_cmap(inputdata.tancmap)
                        inputdata.tancmap = cmap(range(cmap.N))[:,0:3]
                    except ValueError:
                        raise ValueError(f'Invalid colourmap name {inputdata.tancmap} in {self.configfile}. See https://matplotlib.org/gallery/color/colormap_reference.html?highlight=colormap%20reference for list of available')

                if inputdata.radcmap is not None:
                    try:
                        cmap = get_cmap(inputdata.radcmap)
                        inputdata.radcmap = cmap(range(cmap.N))[:,0:3]
                    except ValueError:
                        raise ValueError(f'Invalid colourmap name {inputdata.radcmap} in {self.configfile}. See https://matplotlib.org/gallery/color/colormap_reference.html?highlight=colormap%20reference for list of available')

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
        else:
            # In case mesh generation has been requested in command line but no mesh properties provided in config
            if self.makemesh:
                raise RuntimeError(f'\n-makemesh option specified but no mesh properties defined in {self.configfile}')

        # If no makemesh option given but mesh file in config can't be found, inform user
        if (not self.makemesh and not os.path.exists(inputdata.meshfilename)):
            while True:
                txt = input(f'Specified mesh file {inputdata.meshfilename} was not found and -makemesh option not given.\nGenerate the mesh using properties in config file? [Y/N]')

                if txt.lower() == 'y':
                    self.makemesh = True
                    break
                elif txt.lower() == 'n':
                    self.exit = True
                    break
                else:
                    print('Invalid choice')
                    continue

        # Set defaults
        inputdata.axialVel = (1.0 if inputdata.axialVel is None else inputdata.axialVel)

        return inputdata


if __name__ == '__main__':
    main()
