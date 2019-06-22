from dipy.io import read_bvals_bvecs
import pandas as pd
from os.path import join
import vg

exp_dir = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/regina'

df = pd.read_csv("dmri_paths.csv")
subj_bvecs = {}
subj_bvals = {}
for idx, row in df.iterrows():
    subj_id = str(row['subj_id'])
    fbval = join(exp_dir, subj_id, 'Diffusion/bvals')
    fbvec = join(exp_dir, subj_id, 'Diffusion/bvecs')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    subj_bvecs[subj_id] = bvecs
    subj_bvals[subj_id] = bvals

max_angle = 0
max_id_1 = None
max_id_2 = None
max_t = 0
angles = []

for t in range(288):
    for k1, v1 in subj_bvecs.items():
        for k2, v2 in subj_bvecs.items():
            if k1 == k2:
                continue
            angle = vg.angle(v1[t], v2[t])
            angles.append(angle)

            if angle > max_angle:
                max_angle = angle
                max_id_1 = k1
                max_id_2 = k2
                max_t = t

print(max_angle, max_id_1, max_id_2, max_t)

angles = sorted(angles)

all_match = True
for k1, v1 in subj_bvals.items():
    bvals1 = [round(int(x)/100)*100 for x in v1]
    for k2, v2 in subj_bvals.items():
        bvals2 = [round(int(x)/100)*100 for x in v2]
        if bvals1 != bvals2:
            all_match = False
if all_match:
    print("All bvals match")
