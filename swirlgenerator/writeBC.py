# -------------------------------------------------------------------------------------------------------------------
#
# Module for writing the raw flow field data into a file format which can be uses as an inlet boundary condition
# Takes a FlowField object as input
#
# Assumes that the bottom corner of the inlet surface is at [0,0,0]
#
# Only has functionality for SU2 format at the moment
#
# -------------------------------------------------------------------------------------------------------------------

import core as sg
import pre
import numpy as np


def writeSU2(flowField: sg.FlowField, filename):
    '''
    For writing the data into an SU2 useable format.
    Column format for SU2 inlet boundary condition (3D domain):
    x, y, z, temperature, velocity magnitude,  x-component of flow direction unit vector, y-component of flow direction unit vector, z-component of flow direction unit vector
    - flowfield - FlowField object containing data to be plotted
    - filename - filename to save boundary condition to, should include the .dat extension
    '''

    # Extract x and y coordinates from list of complex coordinates
    x = flowField.coords.real
    y = flowField.coords.imag
    
    # Z coords at end of mesh
    z = np.zeros(np.shape(x))

    # Extract velocity vector
    velVec = flowField.velocity

    # Get velocity magnitude at each cell
    velMag = np.linalg.norm(velVec, axis=1)

    # Get direction components of unit vector
    flowDirection = velVec/ np.column_stack((velMag,velMag,velMag))

    # Dummy temperature array
    temp = np.zeros(np.shape(x))

    # Collate into one matrix for writing into file
    boundaryConditions = np.column_stack((x,y,z,temp,velMag,flowDirection))

    # Write to file
    with open(filename, 'w') as f:
        # Write metadata
        f.write(f"NMARK= 1\nMARKER_TAG= inlet\nNROW= {np.shape(boundaryConditions)[0]}\nNCOL= 8\n")

        # Write each row of matrix into separate line with 6 decimal places
        np.savetxt(f, boundaryConditions, fmt='%.6f')


def writeInlet(InputObject: pre.Input, flowField: sg.FlowField):
    '''
    Wrapper function for calling the correct function to write the boundary condition in the requested format
    - InputObject - Input object which would contain the format to write the boundary condition in
    - flowfield - FlowField object containing data to be plotted
    - Currently only su2 format is supported
    '''

    # Dictionary mapping for different boundary condition file formats and functions which create them
    formats = {'su2':writeSU2}

    # Write boundary condition in correct format
    func = formats.get(InputObject.format)
    func(flowField, InputObject.filename)