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

            # Intialise flow field object with coordinate system
            print(f'Extracting inlet nodes from {inputData.meshfilename}...')
            flowfield = sg.FlowField(inputData.getNodes())


            # Create flow field with vortex method
            if not options.ctm:
                print('Calculating flow field...')
                # Initialise domain configuration object with vortex definitions
                vortexDefs = sg.Vortices(inputObject=inputData)

                # Calculate velocity field from effect of vortices
                flowfield.computeDomain(vortexDefs, axialVel=inputData.axialVel)


            # Reconstruct flow field from contour plots
            else:
                print('Reconstructing flow field...')
                # Translate contour plot images to array of values
                tangential = ct.Contour(inputData.tanImg, inputData.tanRng, cmap=inputData.tancmap, sampleDist=[inputData.numRings, inputData.angRes])
                radial = ct.Contour(inputData.radImg, inputData.radRng, cmap=inputData.radcmap, sampleDist=[inputData.numRings, inputData.angRes])

                # Calculate velocity field from flow angles mapped to flowfield nodes
                flowfield.reconstructDomain(tangential.getValuesAtNodes(flowfield.coords), radial.getValuesAtNodes(flowfield.coords), inputData.axialVel)


            # Verify boundary conditions if requested
            if options.checkboundaries:
                flowfield.checkBoundaries()

            # Save raw numpy arrays if requested
            if options.savenumpy:
                name = options.configfile.split('.')[0]
                flowfield.save(name)

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
        self.showinletnodes     = False
        self.savenumpy          = False

        # Options based on the inputs in config file
        self.ctm    = False            # Are we reconstructing the flow field from contour plots or creating from discrete vortices
        self.vm     = False            # Are we defining the flow field with vortex models

        # Misc flags
        self.exit = False
        self.plot = False

        # For getting help with the command line arguements
        if ('-help' in self.arguments[1]):
            print('Usage: swirlgenerator [config file] [options]')
            print('Options:')
            print('-vm                      Creates flow field from mathematical model and parameters in config file')
            print('-ctm                     Reconstructs flow field from input contour plot images')
            print('-checkboundaries         Runs the function which checks if the boundary conditions have been satisfied')
            print('-show                    Shows the plots of the flow fields in separate windows')
            print('-saveplots               Saves the plots into a pdf file with the same name as the config file')
            print('-showinletnodes          For plotting the inlet nodes - for confirming correct extraction of nodes from mesh')
            print('-savenumpy               For saving the created flowfield as its component numpy arrays')

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

        # Try and use the defined method
        self.vm  = (True if 'vm' in self.arguments else False)
        self.ctm = (True if '-ctm' in self.arguments else False)

        # Check validity of boundary conditions
        self.checkboundaries = (True if '-checkboundaries' in self.arguments else False)

        # Show plots
        self.showFields = (True if ('-show' in self.arguments or '-showplots' in self.arguments) else False)

        # Save plots
        self.saveplots = (True if '-saveplots' in self.arguments else False)

        # Show inlet nodes
        self.showinletnodes = (True if '-showinletnodes' in self.arguments else False)

        if self.showFields or self.saveplots or self.showinletnodes:
            self.plot = True

        # Save raw numpy arrays
        self.savenumpy = (True if '-savenumpy' in self.arguments else False)


    def checkInputs(self, inputdata: pre.Input):
        '''
        Checks which inputs are present from the config file and activates the correct flags for controlling 
        - inputdata - Input object containing the fields read from the config file
        - Output - returns the inputdata object in case we modified it in here
        '''

        # Formats supported for writing boundary conditions
        formats = ['su2']

        # These metadata fields are necessary for using any functionality
        if not inputdata.metadata_flag:
            raise RuntimeError(f'\nMetadata missing in file {self.configfile}')
        if (inputdata.filename is None or inputdata.format is None or inputdata.meshfilename is None):
            raise RuntimeError(f'\nNon-optional metadata missing in file {self.configfile}')

        # Check format specified
        if (inputdata.format not in formats):
            raise NotImplementedError(f"\n{format} not supported")

        # Check flags and update as necessary
        if self.vm and self.ctm:
            self.__askuser__()

        elif not self.vm and not self.ctm:
            if inputdata.vortDefs_flag and inputdata.contours_flag:
                self.__askuser__()

            elif inputdata.vortDefs_flag:
                self.vm  = True

            elif inputdata.contours_flag:
                self.ctm = True

        # Check vortex method inputs if we are going to use it
        if self.vm:
            if inputdata.vortDefs_flag:
                if inputdata.vortModel is None:
                    raise RuntimeError(f'\nVortex model to be used was not defined in {self.configfile}')

                if inputdata.numVortices < 1:
                    raise RuntimeError(f'\nAt least one vortex needs to be defined in {self.configfile}')

            else:
                raise RuntimeError(f'Vortex method specified but no vortex definition section found in {self.configfile}')

        # Check contour translation method inputs if we are going to use it
        if self.ctm:
            if inputdata.contours_flag:
                if inputdata.tanImg is None or inputdata.tanRng is None or inputdata.radImg is None or inputdata.radRng is None:
                    raise RuntimeError(f'\nRequired contour plot information missing in {self.configfile} for flow field reconstruction')

            else:
                raise RuntimeError(f'Contour translation method specified but no contour translation section found in {self.configfile}')
        
        # Set defaults
        inputdata.axialVel = (1.0 if inputdata.axialVel is None else inputdata.axialVel)

        return inputdata

    # Utility function for prompting user
    def __askuser__(self):
        while True:
            choice = input(f'Vortex definitions section and contour translation section both present in {self.configfile}.\nChoose flow field calculation method (V/R):')

            if choice.lower() == 'r':
                self.ctm = True
                self.vm  = False
                break
            elif choice.lower() == 'v':
                self.vm  = True
                self.ctm = False
                break
            else:
                print('Invalid choice')
                continue

    # Utility function for getting the values of a named colour map using matplot lib



if __name__ == '__main__':
    main()
