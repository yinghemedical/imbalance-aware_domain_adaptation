import nibabel
import numpy as np
niifile= nibabel.load("KDVBB2_annotation_4.nii")
data=niifile.get_fdata()
data=np.array(data)
print(data.max)