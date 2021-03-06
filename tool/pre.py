import nibabel as nib
import glob
import os

def cropVolume(img, data=False):
    if not data:
       img = img.get_data()
    sum_array = []
    for ch in range(img.shape[2]):
        values, indexes = np.where(img[:, :, ch] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    ch_s = np.nonzero(sum_array)[0][0]
    ch_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[0]):
        values, indexes = np.where(img[width, :, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    wi_s = np.nonzero(sum_array)[0][0]
    wi_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[1]):
        values, indexes = np.where(img[:, width, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    hi_s = np.nonzero(sum_array)[0][0]
    hi_e = np.nonzero(sum_array)[0][-1]
    return ch_s, ch_e, wi_s, wi_e, hi_s, hi_e

def cropVolumes(img1, img2, img3, img4):

    ch_s1, ch_e1, wi_s1, wi_e1, hi_s1, hi_e1 = cropVolume(img1, True)
    ch_s2, ch_e2, wi_s2, wi_e2, hi_s2, hi_e2 = cropVolume(img2, True)
    ch_s3, ch_e3, wi_s3, wi_e3, hi_s3, hi_e3 = cropVolume(img3, True)
    ch_s4, ch_e4, wi_s4, wi_e4, hi_s4, hi_e4 = cropVolume(img4, True)
    ch_st = min(ch_s1, ch_s2, ch_s3, ch_s4)
    ch_en = max(ch_e1, ch_e2, ch_e3, ch_e4)
    wi_st = min(wi_s1, wi_s2, wi_s3, wi_s4)
    wi_en = max(wi_e1, wi_e2, wi_e3, wi_e4)
    hi_st = min(hi_s1, hi_s2, hi_s3, hi_s4)
    hi_en = max(hi_e1, hi_e2, hi_e3, hi_e4)
    return wi_st, wi_en, hi_st, hi_en, ch_st, ch_en

def cropAndSaveVolumes(imgLoc, dataset_name):
    t1_loc = imgLoc.replace('flair', 't1')
    t1ce_loc = imgLoc.replace('flair', 't1ce')
    t2_loc = imgLoc.replace('flair', 't2')
    gth_loc = imgLoc.replace('flair', 'seg')

    img_flair = nib.load(imgLoc)
    affine_flair = img_flair.affine
    header_flair = img_flair.header
    img_t1 = nib.load(t1_loc)
    affine_t1 = img_t1.affine
    header_t1 = img_t1.header
    img_t1ce = nib.load(t1ce_loc)
    affine_t1ce = img_t1ce.affine
    header_t1ce = img_t1ce.header
    img_t2 = nib.load(t2_loc)
    affine_t2 = img_t2.affine
    header_t2 = img_t2.header
    gth = nib.load(gth_loc)
    affine5 = gth.affine
    header5 = gth.header
    img_flair =img_flair.get_data()
    img_t1 = img_t1.get_data()
    img_t1ce = img_t1ce.get_data()
    img_t2 = img_t2.get_data()
    gth = gth.get_data()


    wi_st, wi_en, hi_st, hi_en, ch_st, ch_en = cropVolumes(img_flair, img_t1, img_t1ce, img_t2)
    img_flair = img_flair[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1 = img_t1[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1ce = img_t1ce[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t2 = img_t2[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    gth = gth[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]

    name =imgLoc.replace(imgLoc.split('/')[-1], '').replace(dataset_name, dataset_name + '_pre')
    if not os.path.isdir(name):
        os.makedirs(name)

    img_flair_cropped = nib.Nifti1Image(img_flair, affine_flair, header_flair)
    nib.save(img_flair_cropped, name + os.sep + imgLoc.split('/')[-1])
    img_t1_cropped = nib.Nifti1Image(img_t1, affine_t1, header_t1)
    nib.save(img_t1_cropped, name + os.sep + t1_loc.split('/')[-1])
    img_t1ce_cropped = nib.Nifti1Image(img_t1ce, affine_t1ce, header_t1ce)
    nib.save(img_t1ce_cropped, name + os.sep + t1ce_loc.split('/')[-1])
    img_t2_cropped = nib.Nifti1Image(img_t2, affine_t2, header_t2)
    nib.save(img_t2_cropped, name + os.sep + t2_loc.split('/')[-1])
    gth_cropped = nib.Nifti1Image(gth, affine5, header5)
    nib.save(gth_cropped, name + os.sep + gth_loc.split('/')[-1])


if __name__ == '__main__':
    data = '/root/data/train' + '/' + 'HGG' + '/'
    folderlist = glob.glob(data + '*')
    for folder in folderlist:
        files = glob.glob(folder + '/*file.nii.gz')
        for file in files:
            cropAndSaveVolumes(file, dataset_name)