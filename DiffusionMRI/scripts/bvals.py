from os.path import join
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'

subj_id = '784565'
pth = join(exp_dir, subj_id, 'bvals')
with open(pth, "r") as f:
    bvals = f.readlines()[0].rstrip(" \n")
    bvals = bvals.split("  ")

bvals_1 = [round(int(x)/100)*100 for x in bvals]

subj_id = '786569'
pth = join(exp_dir, subj_id, 'bvals')
with open(pth, "r") as f:
    bvals = f.readlines()[0].rstrip(" \n")
    bvals = bvals.split("  ")

bvals_2 = [round(int(x)/100)*100 for x in bvals]

subj_id = '789373'
pth = join(exp_dir, subj_id, 'bvals')
with open(pth, "r") as f:
    bvals = f.readlines()[0].rstrip(" \n")
    bvals = bvals.split("  ")

bvals_3 = [round(int(x)/100)*100 for x in bvals]

for a, b, c in zip(bvals_1, bvals_2, bvals_3):
    print(a, " -- ", b, " -- ", c)

print(a == b == c)
