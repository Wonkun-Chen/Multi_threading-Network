import torch
import numpy as np
import nibabel as nib
from utils import cropVolume
import Model as Net
import os

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


def segmentVoxel(imgLoc, model):
    t1_loc = imgLoc.replace('flair', 't1')
    t1ce_loc = imgLoc.replace('flair', 't1ce')
    t2_loc = imgLoc.replace('flair', 't2')
    label_loc = imgLoc.replace('flair', 'seg')
    img_flair = nib.load(imgLoc).get_data()
    gth_dummy = np.copy(img_flair)
    img_t1 = nib.load(t1_loc).get_data()
    img_t1ce = nib.load(t1ce_loc).get_data()
    img_t2 = nib.load(t2_loc).get_data()

    wi_st, wi_en, hi_st, hi_en, ch_st, ch_en = cropVolumes(img_flair, img_t1, img_t1ce, img_t2)
    img_flair = norm(img_flair[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t1 = norm(img_t1[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t1ce = norm(img_t1ce[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t2 = norm(img_t2[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    resize = (1, img_flair.shape[0], img_flair.shape[1], img_flair.shape[2])
    img_flair = img_flair.reshape(resize)
    img_t1 = img_t1.reshape(resize)
    img_t1ce = img_t1ce.reshape(resize)
    img_t2 = img_t2.reshape(resize)

    tensor_flair = torch.from_numpy(img_flair)
    tensor_t1 = torch.from_numpy(img_t1)
    inputB = torch.from_numpy(img_t1ce)
    tensor_t2 = torch.from_numpy(img_t2)
    del img_flair, img_t1, img_t1ce, img_t2
    tensor_concat = torch.cat([tensor_flair, tensor_t1, inputB, tensor_t2], 0)  # inputB #
    tensor_concat = torch.unsqueeze(tensor_concat, 0)
    tensor_concat = tensor_concat.cuda()

    test_resolutions = [None, (80, 80, 80)]
    output = None
    for res in test_resolutions:
        if output is not None:
            output = output + model(tensor_concat_var, inp_res=res)
        else:
            output = model(tensor_concat_var, inp_res=res)
    output = output / len(test_resolutions)
    del tensor_concat, tensor_concat_var

    # output_ = output[0].max(0)[1].data.byte().cpu().numpy()
    output_numpy = np.zeros_like(gth_dummy)
    output_numpy[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en] = output_
    output_numpy[output_numpy == 3] = 4

    img1 = nib.load(imgLoc)
    img1_new = nib.Nifti1Image(output_numpy, img1.affine, img1.header)
    name = imgLoc.replace('_flair', '').split('/')[-1]
    out_dir = '/root/segment'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    file_name = out_dir + os.sep + name
    nib.save(img1_new, file_name)
    
def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im

if __name__ == '__main__':
    data_dir = '/root/data/val/' 
    test_file = 'val.txt'# data adress
    if not os.path.isfile(data_dir + os.sep + test_file):
        print('None validation file')
        exit(-1)

    best_model_loc = '/pretrained-file-adress'
    if not os.path.isfile(best_model_loc):
        print('None pretrained file')
        exit(-1)

    model = Net.Net(classes=4, channels=4)
    model.load_state_dict(torch.load(best_model_loc))
    model = model.cuda()
    model.eval()

    dice_scores_wt = []
    dice_scores_cm = []
    dice_scores_et = []
    with open(data_dir + test_file) as txtFile:
        for line in txtFile:
            line_arr = line.split(',')
            img_file = ((data_dir).strip() + '/' + line_arr[0].strip()).strip()
            segmentVoxel(img_file, model)