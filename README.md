# hinge_and_cold_feature_extraction
This is a Python code implementation of COLD features and Hinge features from any image. These features are particularly used for defining patterns in textual images but can be used for extracting features of other kind of objects too.  

## COLD features
The COLD feature (https://doi.org/10.1016/j.patcog.2016.09.044) is inspired by the shape context descriptor, with respect to the
extracted information into a log-polar histogram, which is more sensitive to regions of
nearby the center than to those farther away. The feature extracts unique shapes of image (here text) components by analyzing the relationship between dominant points;
such as straight, angle-oriented features and curvature over contours of the components.

## Hinge features
The contour-based feature was designed to capture the curvature of the ink trace of the document images, which is considered to be very discriminatory between handwritings. The best contour-based features reported in the literature are the Hinge, QuillHinge, and Delta-n Hinge features. This repo includes an implementation of Hinge features only.

#### Combination of COLD and Hinge features can be used in case like gender identification, writer identification, 2D/3D text classification and many more. In this repo, I have presented an implementation of gender identification from this work (https://doi.org/10.1007/978-3-030-51935-3_25)
