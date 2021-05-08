import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()

list1=[76,820,411,798,951,946,761,416,498,516,264,884,282,296,526,495,468,130,590,751,624,41,223,541,797,741]
list2=[638,599,770,1144,60,1146,160,612,763,213,429,1142,588,760,358,364,1126,1085,1119,1125,1092,113,1120,1108,1097,489,1135,1121,13,1123,272,1095,1122,1136,35,336,1104,532,1139,1129,279,1128,1100,641,694,1116,1089,291,1103,497,750,142,168,93,795,568,84,387,344,230,542,1148,635]
list3=[983,767,548,549,58,629,101,1053,248,274,665,477,463,501,265,451,684,679,730,693,708,709,696,131,737,42,631,382,43,815,829,424,342,747,627,192,742,420,393,966,52]
list4=[823,1019,1025,994,981,1030,1024,1018,822,834,996,982,1027,1037,1023,986,826,832,1020,1008,953,1035,840,883,1046,920,921,869,841,1079,1051,1050,1054,927,1041,847,1043,1080,925,850,876,1070,917,1071,1059,877,867,1049,1048,899,858,870,802,1038,962,1039,803,801,1012,800,1002,970,959,811,813,1000]
list5=[348,215,772,603,166,249,649,449,134,322,242,686,442,331,253,194,69,593,784,579,186,569,185,768,91]
list6=[63,189,412,374,1031,214,200,764,765,759,567,361,407,349,611,605,188,74,48,613,175,161,439,363,1032,559,203,565,571,940,799,202,564,558,362,606,174,148,49,75,71,65,616,602,158,399,414,372,366,400,428,560,206,574,945,789,777,788,575,561,367,401,415,398,159,165,824,64,830,818,198,601,99,615,403,371,359,577,211,775,749,748,238,576,210,1009,589,402,628,614,98,600,73,9,698,14,28,129,673,317,471,288,511,263,504,510,276,464,302,458,672,100,666,896,116,670,328,472,300,466,506,260,704,705,513,507,261,473,103,117,659,842,16,12,675,661,339,503,259,729,700,258,270,310,890,106,660,674,39,879,689,104,110,676,138,306,312,500,528,703,717,918,267,273,475,313,461,139,677,105,38,691,108,646,493,444,295,281,518,530,901,929,732,243,531,257,519,280,323,479,135,109,36,692,137,889,651,645,123,484,453,447,725,731,724,718,240,268,297,283,452,644,136,687,683,27,654,668,456,318,287,244,522,536,250,720,734,735,251,245,292,286,480,669,655,133,26,32,682,680,18,24,30,496,327,441,333,469,1076,521,938,723,736,520,246,285,332,326,483,124,25,695,681,4,56,816,81,619,625,369,341,427,433,355,584,209,553,221,778,546,552,585,432,340,426,156,94,618,80,195,181,57,7,197,183,96,82,154,140,626,395,381,356,430,587,544,550,236,791,746,974,551,586,343,419,380,633,155,97,182,40,814,54,6,2,50,44,623,179,421,409,596,227,233,780,794,757,554,540,226,1017,583,408,346,434,352,391,92,144,150,45,51,79,191,47,53,146,620,634,608,90,422]
list7=[1145,1140,1141,1132,1091,1090,1084,1127,1133,1131,1093,1130,1124,1118,1134,1083,1096,1109,1094,1113,1098,1138,1110,1101,1115,1102,1088,1117,1149]
list8=[88,229,89,163,607,410,809,821,212,979,762,563,774,204,1021,370,172,459,303,505,909,1047,843,658,711,739,275,671,305,1069,851,448,1056,685,450,653,848,862,490,309,321,719,903,122,37,278,866,872,643,535,252,792,787,793,591,354,790,237,1006,83,169,78,743,597,557,345,392]
list9=[758,1033,1147,835,562,833,316,1087,852,307,844,861,1072,255,308,491,888,537,1114,871,977,817,632,961,960]
list10=[957,995,836,808,1026,954,968,997,955,941,438,819,831,825,992,993,987,950,944,978,1022,1036,1034,952,991,985,990,947,827,854,868,1052,908,882,855,857,894,1045,923,937,936,1044,1078,881,895,856,846,891,885,1040,926,932,933,1082,1055,853,845,886,892,931,1081,1042,893,887,878,875,849,1067,915,928,900,1066,874,1058,1064,1065,873,1075,1061,907,913,1060,864,1062,910,911,1077,1063,865,1004,1010,976,989,988,963,1005,1013,975,948,810,804,964,965,807,1015,1029,998,999,1028,1014,812,806]

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    img_fold_AB1 = os.path.join('./data1', sp)
    img_fold_AB2 = os.path.join('./data2', sp)
    img_fold_AB3 = os.path.join('./data3', sp)
    img_fold_AB4 = os.path.join('./data4', sp)
    img_fold_AB5 = os.path.join('./data5', sp)
    img_fold_AB6 = os.path.join('./data6', sp)
    img_fold_AB7 = os.path.join('./data7', sp)
    img_fold_AB8 = os.path.join('./data8', sp)
    img_fold_AB9 = os.path.join('./data9', sp)
    img_fold_AB10 = os.path.join('./data10', sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    if not os.path.isdir(img_fold_AB1):
        os.makedirs(img_fold_AB1)
    if not os.path.isdir(img_fold_AB2):
        os.makedirs(img_fold_AB2)
    if not os.path.isdir(img_fold_AB3):
        os.makedirs(img_fold_AB3)
    if not os.path.isdir(img_fold_AB4):
        os.makedirs(img_fold_AB4)
    if not os.path.isdir(img_fold_AB5):
        os.makedirs(img_fold_AB5)
    if not os.path.isdir(img_fold_AB6):
        os.makedirs(img_fold_AB6)
    if not os.path.isdir(img_fold_AB7):
        os.makedirs(img_fold_AB7)
    if not os.path.isdir(img_fold_AB8):
        os.makedirs(img_fold_AB8)
    if not os.path.isdir(img_fold_AB9):
        os.makedirs(img_fold_AB9)
    if not os.path.isdir(img_fold_AB10):
        os.makedirs(img_fold_AB10)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        flag=0
        num=0
        for i in name_A:
            if i>='0' and i<='9':
                flag=1
                num=num*10+int(i)
            else:
                if flag==1:
                    break
        flag=0
        if num in list1:
            mmm=img_fold_AB
        elif num in list2:
            mmm=img_fold_AB1
        elif num in list3:
            mmm=img_fold_AB2
        elif num in list4:
            mmm=img_fold_AB3
        elif num in list5:
            mmm=img_fold_AB4
        elif num in list6:
            mmm=img_fold_AB5
        elif num in list7:
            mmm=img_fold_AB6
        elif num in list8:
            mmm=img_fold_AB7
        elif num in list9:
            mmm=img_fold_AB8
        elif num in list10:
            mmm=img_fold_AB9
        else:
            mmm=img_fold_AB10
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(mmm, name_AB)
            im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            # print(path_A)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
