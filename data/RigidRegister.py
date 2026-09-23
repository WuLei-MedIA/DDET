
import SimpleITK as sitk
import numpy as np

def rigid_register_arterial_to_portal(fixed_path, moving_path, output_path=None):
    """
    fixed  : Portal phase
    moving : Arterial phase
    """

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    registration_method = sitk.ImageRegistrationMethod()

    # -------------------- Mattes Mutual Information --------------------
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)   # resample: 20%

    # -------------------- Linear Interpolator --------------------
    registration_method.SetInterpolator(sitk.sitkLinear)

    # -------------------- Optimizer --------------------
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # -------------------- Multi-resolution strategy --------------------
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # -------------------- Initializer Transformer --------------------
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    print("starting registration process...")
    final_transform = registration_method.Execute(fixed, moving)

    moving_registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    if output_path is not None:
        sitk.WriteImage(moving_registered, output_path)
        print(f"save to: {output_path}")

        sitk.WriteTransform(final_transform, output_path.replace('.nii.gz', '_transform.tfm'))

    return moving_registered, final_transform


if __name__ == "__main__":
    fixed_image_path  = "portal_phase.nii.gz"
    moving_image_path = "arterial_phase.nii.gz"
    output_image_path = "arterial_registered.nii.gz"

    registered_img, transform = rigid_register_arterial_to_portal(
        fixed_image_path,
        moving_image_path,
        output_image_path
    )