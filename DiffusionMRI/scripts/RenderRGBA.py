#!/usr/bin/env python

#
# Script to volume render a pre-classified RGBA volume
#

import vtk
import numpy as np
from os.path import join


def dvr_rgba(data_matrix, dims):
    """Direct volume render a pre-classified RGBA volume.

    Parameters
    ----------
    vol
        4D numpy array of type uint8 and dimension (nx,ny,nz,4)
    dims
        tuple of voxel spacings (dx,dy,dz)
    """

    assert len(data_matrix.shape) == 4
    (nx, ny, nz, nk) = data_matrix.shape
    assert nk == 4
    assert data_matrix.dtype == np.uint8
    assert len(dims) == 3

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # For VTK to be able to use the data, it must be stored as a VTK-image.
    # This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # Convert numpy array to a string of chars and import it.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # We have four scalar components (RGBA) per voxel
    dataImporter.SetNumberOfScalarComponents(4)
    # Set data extent
    dataImporter.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    dataImporter.SetDataExtentToWholeExtent()
    dataImporter.SetDataSpacing(dims[0], dims[1], dims[2])

    # Establish a trivial opacity transfer function
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(255, 1.0)

    # usually maps data to color and opacity. here, we just use the given values
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetIndependentComponents(False)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    # The volume will be displayed by ray-casting with alpha compositing.
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    volumeMapper.SetBlendModeToComposite()

    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Finally, add the volume to the renderer
    ren.AddVolume(volume)
    ren.SetBackground(1, 1, 1)

    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the left
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetPosition(c[0] + 400, c[1], c[2])
    camera.SetViewUp(0, 0, -1)

    # Increase the size of the render window
    renWin.SetSize(640, 480)

    # Interact with the data.
    iren.Initialize()
    renWin.Render()
    iren.Start()


if __name__ == '__main__':
    # Create a numpy array for demonstration
    SUBJ_ID = "784565"
    DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
    FEATURES_NAME = "PCA"
    RESULTS_PATH = join(DATA_DIR, SUBJ_ID, "outputs", FEATURES_NAME + "_color_volume.npz")
    data_matrix = np.load(RESULTS_PATH)["data"]

    dvr_rgba(data_matrix, (1.0, 1.0, 1.0))
