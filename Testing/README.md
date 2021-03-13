## Test Scripts

 Python scripts utilising the swirlgenerator toolkit to generate flow fields and perform parameteric studies on its methods.

 Used to generate the results discussed within the accompanying technical paper.

 Also showcases how the methods can be integrated into existing workflows.

### Contour Translation Accuracy
Generates random swirling flowfields and outputs their contour plot descriptions. These are used to measure the accuracy of the contour translation method using different parameters.

Utilises parallel pool of cpu cores.


### Flux Across Boundary
Generates random swirling flowfields and calculates the amount of flux across the perimiter to measure the accuracy of the modelled solid boundary.

Utilises parallel pool of cpu cores.


### Idealised Swirl Cases
Collection of config files and images of contour plots to recreate the four idealised swirl test cases described in the literature.

Batch file used to generate the flowfields as boundary condition files and plots within PDF files. Command line interface used.