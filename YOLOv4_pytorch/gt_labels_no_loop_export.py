import torch
import numpy as np
import random

def doOverlap(l1, r1, l2, r2):

    # If one rectangle is on left side of other
    no_int_left_cond = (torch.ge(l1[0], r2[0]) | torch.ge(l2[0],r1[0]))
    #print(no_int_left_cond)

    # If one rectangle is above other
    no_int_top_cond = (torch.ge(r1[1], l2[1]) | torch.ge(r2[1],l1[1]))
    #print(no_int_top_cond)

    no_intersec_cond = no_int_left_cond | no_int_top_cond
    #print(no_intersec_cond)

    return ~no_intersec_cond

def labels_ablation(labels, img_h, img_w, seed, abl_size, abl_size2, device):

    col_abl = 0
    row_abl = 1
    block_abl = 0
    #at_least_one = 0

    print('seed: ' + str(seed))
    rng1 = np.random.RandomState(seed)
    rnd_pos_c = rng1.randint(0, img_w)
    rnd_pos_r = rng1.randint(0, img_h)

    #retained_labels = []

    print('Random starting position along columns: ' + str(rnd_pos_c))
    print('Random starting position along rows: ' + str(rnd_pos_r))
    print('Retention size when columns: ' + str(abl_size))
    print('Retention size when rows: ' + str(abl_size2))

    #input labels in xl,yl,w,h norm format
    h = img_h
    w = img_w

    denorm = [1, 1, w, h, w, h] #image num, cls_id, xl, yt, w, h
    labels_denorm = labels*torch.FloatTensor(denorm).to(device)

    gt_xl = labels_denorm[:, 2] - labels_denorm[:, 4] / 2
    gt_xr = labels_denorm[:, 2] + labels_denorm[:, 4] / 2
    gt_yt = h - (labels_denorm[:, 3] - labels_denorm[:, 5] / 2)
    gt_yb = h - (labels_denorm[:, 3] + labels_denorm[:, 5] / 2)

    gt_lefttop = (gt_xl, gt_yt)
    gt_rightbottom = (gt_xr, gt_yb)

    # switch among rows, cols and blocks
    if col_abl:

        if rnd_pos_c + abl_size <= w:
            rz_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz_xr = (rnd_pos_c + abl_size) * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yb = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yt = h * torch.ones(labels_denorm.size()[0]).to(device)

            rz_lefttop = (rz_xl, rz_yt)
            rz_rightbottom = (rz_xr, rz_yb)

            intersection_bool = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop, rz_rightbottom)
            # print('intersection_bool: ' + str(intersection_bool))

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

        else:
            # zone 1
            rz1_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_xr = (rnd_pos_c + abl_size - w) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yb = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yt = h * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone1_corners_list = [(rz1_xl, rz1_yt), (rz1_xr, rz1_yt), (rz1_xr, rz1_yb),
            #                              (rz1_xl, rz1_yb)]  # tuples for the corners

            #print('Retain zone 1 corners: ' + str(retain_zone1_corners_list))

            rz_lefttop1 = (rz1_xl, rz1_yt)
            rz_rightbottom1 = (rz1_xr, rz1_yb)

            intersection_bool1 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop1, rz_rightbottom1)

            # zone 2
            rz2_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yb = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yt = h * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone2_corners_list = [(rz2_xl, rz2_yt), (rz2_xr, rz2_yt), (rz2_xr, rz2_yb),
            #                              (rz2_xl, rz2_yb)]  # tuples for the corners
            #print('Retain zone 2 corners: ' + str(retain_zone2_corners_list))

            rz_lefttop2 = (rz2_xl, rz2_yt)
            rz_rightbottom2 = (rz2_xr, rz2_yb)

            intersection_bool2 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop2, rz_rightbottom2)

            intersection_bool = intersection_bool1 | intersection_bool2

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

    elif row_abl:

        if rnd_pos_r + abl_size2 <= h:
            rz_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yb = (h - (rnd_pos_r + abl_size2)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone_corners_list = [(rz_xl, rz_yt), (rz_xr, rz_yt), (rz_xr, rz_yb),
            #                             (rz_xl, rz_yb)]  # tuples for the corners
            #print('Retain zone corners: ' + str(retain_zone_corners_list))

            rz_lefttop = (rz_xl, rz_yt)
            rz_rightbottom = (rz_xr, rz_yb)

            intersection_bool = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop, rz_rightbottom)
            # print('intersection_bool: ' + str(intersection_bool))

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

        else:
            # zone 1
            rz1_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yb = (h - (rnd_pos_r + abl_size2 - h)) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yt = (h - 0) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone1_corners_list = [(rz1_xl, rz1_yt), (rz1_xr, rz1_yt), (rz1_xr, rz1_yb),
            #                              (rz1_xl, rz1_yb)]  # tuples for the corners
            #print('Retain zone 1 corners: ' + str(retain_zone1_corners_list))

            rz_lefttop1 = (rz1_xl, rz1_yt)
            rz_rightbottom1 = (rz1_xr, rz1_yb)

            intersection_bool1 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop1, rz_rightbottom1)

            # zone 2
            rz2_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yb = (h - h) * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone2_corners_list = [(rz2_xl, rz2_yt), (rz2_xr, rz2_yt), (rz2_xr, rz2_yb),
            #                              (rz2_xl, rz2_yb)]  # tuples for the corners
            #print('Retain zone 2 corners: ' + str(retain_zone2_corners_list))

            rz_lefttop2 = (rz2_xl, rz2_yt)
            rz_rightbottom2 = (rz2_xr, rz2_yb)

            intersection_bool2 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop2, rz_rightbottom2)

            intersection_bool = intersection_bool1 | intersection_bool2

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

    elif block_abl:
        print('block')
        no_ovlp_case = (rnd_pos_c + abl_size <= w) and (rnd_pos_r + abl_size2 <= h)
        ovlp_cls = (rnd_pos_c + abl_size > w) and (rnd_pos_r + abl_size2 <= h)
        ovlp_rws = (rnd_pos_c + abl_size <= w) and (rnd_pos_r + abl_size2 > h)
        ovlp_cls_and_rws = (rnd_pos_c + abl_size > w) and (rnd_pos_r + abl_size2 > h)

        if no_ovlp_case:
            print('no ovlp')
            rz_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz_xr = (rnd_pos_c + abl_size) * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz_yb = (h - (rnd_pos_r + abl_size)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone_corners_list = [(rz_xl, rz_yt), (rz_xr, rz_yt), (rz_xr, rz_yb),
            #                             (rz_xl, rz_yb)]  # tuples for the corners
            #print('Retain zone corners: ' + str(retain_zone_corners_list))

            rz_lefttop = (rz_xl, rz_yt)
            rz_rightbottom = (rz_xr, rz_yb)

            intersection_bool = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop, rz_rightbottom)
            # print('intersection_bool: ' + str(intersection_bool))

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

        elif ovlp_cls:
            print('ovlp_cls')
            # zone 1
            rz1_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_xr = (rnd_pos_c + abl_size - w) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yb = (h - (rnd_pos_r + abl_size2)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone1_corners_list = [(rz1_xl, rz1_yt), (rz1_xr, rz1_yt), (rz1_xr, rz1_yb),
            #                              (rz1_xl, rz1_yb)]  # tuples for the corners
            #print('Retain zone 1 corners: ' + str(retain_zone1_corners_list))

            rz_lefttop1 = (rz1_xl, rz1_yt)
            rz_rightbottom1 = (rz1_xr, rz1_yb)

            intersection_bool1 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop1, rz_rightbottom1)

            # zone 2
            rz2_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yb = (h - (rnd_pos_r + abl_size2)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone2_corners_list = [(rz2_xl, rz2_yt), (rz2_xr, rz2_yt), (rz2_xr, rz2_yb),
            #                              (rz2_xl, rz2_yb)]  # tuples for the corners
            #print('Retain zone 2 corners: ' + str(retain_zone2_corners_list))

            rz_lefttop2 = (rz2_xl, rz2_yt)
            rz_rightbottom2 = (rz2_xr, rz2_yb)

            intersection_bool2 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop2, rz_rightbottom2)

            intersection_bool = intersection_bool1 | intersection_bool2

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

        elif ovlp_rws:
            print('ovlp_rws')
            # zone 1
            rz1_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_xr = (rnd_pos_c + abl_size) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yt = (h - 0) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yb = (h - (rnd_pos_r + h - abl_size2)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone1_corners_list = [(rz1_xl, rz1_yt), (rz1_xr, rz1_yt), (rz1_xr, rz1_yb),
            #                              (rz1_xl, rz1_yb)]  # tuples for the corners
            #print('Retain zone 1 corners: ' + str(retain_zone1_corners_list))

            rz_lefttop1 = (rz1_xl, rz1_yt)
            rz_rightbottom1 = (rz1_xr, rz1_yb)

            intersection_bool1 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop1, rz_rightbottom1)

            # zone 2
            rz2_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_xr = (rnd_pos_c + abl_size) * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yb = (h - h) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone2_corners_list = [(rz2_xl, rz2_yt), (rz2_xr, rz2_yt), (rz2_xr, rz2_yb),
            #                              (rz2_xl, rz2_yb)]  # tuples for the corners
            #print('Retain zone 2 corners: ' + str(retain_zone2_corners_list))

            rz_lefttop2 = (rz2_xl, rz2_yt)
            rz_rightbottom2 = (rz2_xr, rz2_yb)

            intersection_bool2 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop2, rz_rightbottom2)

            intersection_bool = intersection_bool1 | intersection_bool2

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]


        elif ovlp_cls_and_rws:
            print('ovlp_cls_rws')
            # zone 1
            rz1_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_xr = (rnd_pos_c + abl_size - w) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz1_yb = (h - h) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone1_corners_list = [(rz1_xl, rz1_yt), (rz1_xr, rz1_yt), (rz1_xr, rz1_yb),
            #                              (rz1_xl, rz1_yb)]  # tuples for the corners
            #print('Retain zone 1 corners: ' + str(retain_zone1_corners_list))

            rz_lefttop1 = (rz1_xl, rz1_yt)
            rz_rightbottom1 = (rz1_xr, rz1_yb)

            intersection_bool1 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop1, rz_rightbottom1)

            # zone 2
            rz2_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yt = (h - rnd_pos_r) * torch.ones(labels_denorm.size()[0]).to(device)
            rz2_yb = (h - h) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone2_corners_list = [(rz2_xl, rz2_yt), (rz2_xr, rz2_yt), (rz2_xr, rz2_yb),
            #                              (rz2_xl, rz2_yb)]  # tuples for the corners
            #print('Retain zone 2 corners: ' + str(retain_zone2_corners_list))

            rz_lefttop2 = (rz2_xl, rz2_yt)
            rz_rightbottom2 = (rz2_xr, rz2_yb)

            intersection_bool2 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop2, rz_rightbottom2)

            # zone 3
            rz3_xl = rnd_pos_c * torch.ones(labels_denorm.size()[0]).to(device)
            rz3_xr = w * torch.ones(labels_denorm.size()[0]).to(device)
            rz3_yt = (h - 0) * torch.ones(labels_denorm.size()[0]).to(device)
            rz3_yb = (h - (rnd_pos_r + abl_size2 - h)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone3_corners_list = [(rz3_xl, rz3_yt), (rz3_xr, rz3_yt), (rz3_xr, rz3_yb),
            #                              (rz3_xl, rz3_yb)]  # tuples for the corners
            #print('Retain zone 3 corners: ' + str(retain_zone3_corners_list))

            rz_lefttop3 = (rz3_xl, rz3_yt)
            rz_rightbottom3 = (rz3_xr, rz3_yb)

            intersection_bool3 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop3, rz_rightbottom3)

            # zone 4
            rz4_xl = 0 * torch.ones(labels_denorm.size()[0]).to(device)
            rz4_xr = (rnd_pos_c + abl_size - w) * torch.ones(labels_denorm.size()[0]).to(device)
            rz4_yt = (h - 0) * torch.ones(labels_denorm.size()[0]).to(device)
            rz4_yb = (h - (rnd_pos_r + abl_size2 - h)) * torch.ones(labels_denorm.size()[0]).to(device)
            # retain_zone4_corners_list = [(rz4_xl, rz4_yt), (rz4_xr, rz4_yt), (rz4_xr, rz4_yb),
            #                              (rz4_xl, rz4_yb)]  # tuples for the corners
            #print('Retain zone 4 corners: ' + str(retain_zone4_corners_list))

            rz_lefttop4 = (rz4_xl, rz4_yt)
            rz_rightbottom4 = (rz4_xr, rz4_yb)

            intersection_bool4 = doOverlap(gt_lefttop, gt_rightbottom, rz_lefttop4, rz_rightbottom4)

            intersection_bool = intersection_bool1 | intersection_bool2 | intersection_bool3 | intersection_bool4

            intersection_bool = torch.where(intersection_bool == True, torch.ones(1).to(device), torch.zeros(1).to(device))
            # print(intersection_bool)
            intersection_bool = torch.unsqueeze(intersection_bool, 0)
            intersection_bool = torch.transpose(intersection_bool, 0, 1)
            # print(intersection_bool)

            labels = torch.cat((labels, intersection_bool), 1)
            # print(labels)

            labels = labels[labels[:, 6] > 0]

            # print(labels)
            retained_labels = labels[:, :6]

    print('\ngt_size after ablation: ' + str(retained_labels.size()))
    return retained_labels

# labels = torch.FloatTensor([[6.00000e+00, 8.00000e+00, 4.85415e-01, 9.80784e-02, 2.30160e-02, 3.49686e-02],
#         [6.00000e+00, 8.00000e+00, 6.44023e-01, 1.02031e-01, 1.74220e-02, 3.25934e-02],
#         [6.00000e+00, 8.00000e+00, 7.52344e-01, 1.11992e-01, 2.45940e-02, 1.36406e-02],
#         [6.00000e+00, 0.00000e+00, 9.40945e-01, 6.94946e-01, 1.11547e-01, 3.38904e-02],
#         [6.00000e+00, 8.00000e+00, 4.53430e-01, 8.41248e-02, 4.48589e-02, 7.71876e-02],
#         [6.00000e+00, 8.00000e+00, 5.05929e-01, 9.99845e-02, 1.65470e-02, 3.07187e-02],
#         [6.00000e+00, 8.00000e+00, 5.76454e-01, 1.00195e-01, 1.93120e-02, 3.22659e-02],
#         [6.00000e+00, 2.50000e+01, 1.53461e-01, 8.46922e-01, 1.97266e-01, 1.74344e-01],
#         [6.00000e+00, 0.00000e+00, 1.80071e-01, 9.22312e-01, 9.47340e-02, 1.55375e-01]])
#
# print(labels.size())
#
# rng1 = np.random.RandomState(50)
# rng2 = np.random.RandomState(100)
#
# print('gt_size before ablation: ' + str(labels.size()))
# print('gt before ablation: ' + str(labels))
# labels = labels_ablation(labels, 0.5*416, 300, rng2.randint(0,416))
# print('\ngt_size after ablation: ' + str(labels.size()))
# print('gt after ablation: ' + str(labels))
