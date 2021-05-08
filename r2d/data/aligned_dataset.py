import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        list1=[416,495,516,946,
            307,308,316,817,833,835,844,852,861,871,888,960,961,977,1033,1072,
            1,8,42,43,52,58,101,192,248,265,274,342,382,420,424,451,463,477,501,627,629,737,742,747,767,
            813,850,1071,
            1094,1143,
            13,35,60,93,113,142,160,230,344,429,588,542,532,489,599,635,638,694,770,1085,1089,1095,1108,1104,1103,1100,1121,
            1125,1128,1136,1142,1144,1146,1148,
            854,887,936,937,1005,
            2,4,6,7,12,14,16,17,18,25,27,28,30,32,38,39,40,44,45,47,49,50,51,53,56,57,63,64,73,74,75,79,80,81,82,85,90,92,94,96,98,
            100,104,108,110,116,117,123,124,129,133,135,137,138,139,144,146,148,150,153,154,155,156,159,174,175,181,182,183,184,
            188,189,191,195,197,198,200,202,203,209,210,214,214,233,236,238,240,243,244,245,246,251,258,159,260,161,263,267,
            270,276,280,281,283,285,286,287,288,292,295,300,302,306,310,312,313,317,318,326,327,328,332,333,339,343,346,349,355,356,359,361,362,363,366,
            369,371,374,378,381,391,398,399,400,401,402,403,408,409,412,414,415,419,423,426,427,428,430,432,433,434,444,452,453,456,458,461,
            464,466,469,471,472,473,475,484,493,496,500,503,504,506,507,511,513,519,520,521,522,528,531,536,540,552,553,554,556,558,559,560,564,
            565,567,574,575,576,577,581,583,584,585,586,569,595,596,600,605,606,609,613,614,615,616,619,621,623,625,626,628,633,644,645,646,651,654,655,659,
            661,666,668,670,673,676,677,680,681,682,683,687,689,691,692,698,700,703,705,718,723,724,725,729,731,732,734,735,736,740,748,749,757,759,764,
            765,775,778,780,783,789,791,794,799,842,901,940,1032,
            1,8,85,153,184,378,423,556,581,595,609,621,740,783,1143,1177,1243,1249,1256,1272,1274,1284,1292]

        list2=[76,526,223,282,296,411,526,498,516,541,624,741,761,820,951,798,130,
            177,307,308,316,491,537,562,758,833,835,888,961,1072,1087,1114,
            131,393,501,627,693,
            801,802,803,811,822,847,867,870,876,899,902,920,925,959,981,982,986,1008,1018,1020,1024,1037,1039,1051,1070,1071,
            69,186,201,768,772,784,
            1083,1088,1090,1091,1094,1098,1102,1106,1107,1109,1111,1112,1113,1131,1132,1140,1145,1149,
            142,336,532,599,638,694,770,750,760,1126,
            804,806,807,808,819,827,846,849,853,865,886,887,892,895,900,906,908,910,913,915,933,941,954,965,968,976,978,988,990,993,997,998,
            1010,1014,1040,1042,1052,1055,1061,1065,1067,1075,1077,
            2,7,12,14,17,24,25,26,28,32,38,39,44,45,48,49,54,56,57,64,71,73,74,75,79,85,90,92,96,98,100,104,106,109,117,124,129,140,150,156,161,165,179,181,183,188,
            195,206,209,214,221,226,231,233,236,246,250,258,259,261,267,268,288,292,295,300,313,323,326,332,340,343,346,352,355,356,367,
            378,379,395,400,402,403,408,421,427,428,432,434,439,441,453,458,466,472,475,480,484,496,500,504,506,507,513,519,530,543,554,561,564,565,567,574,576,
            583,600,601,602,605,609,633,634,646,659,666,673,680,683,687,698,703,704,717,718,725,729,734,735,740,748,749,757,759,765,775,783,
            788,799,814,830,889,890,918,938,945,974,1001,1009,1017,1031,1076,1032,
            85,177,231,378,379,543,740,783,902,906,997,1001,1106,1107,1111,1112,1252,1284]
        # list3=[983,767,548,549,58,8,629,101,1053,114,248,274,665,477,463,501,265,451,684,679,730,693,708,709,696,131,737,42,631,382,43,815,829,424,342,747,627,192,742,420,1,393,966,52]
        # list4=[823,1019,1025,994,981,1030,1024,1018,822,834,996,982,1027,1037,1023,986,826,832,1020,1008,953,1035,840,883,1046,920,921,869,841,1079,1051,922,1050,1054,927,1041,847,1043,1080,925,850,876,1070,902,916,917,1071,1059,877,867,1049,1048,899,858,870,802,1038,962,1039,803,801,1012,800,1002,970,959,811,813,1000]
        # list5=[348,201,215,772,170,603,166,249,649,449,134,322,242,686,442,331,253,194,208,69,593,784,579,186,569,185,768,91]
        # list6=[63,189,412,374,1031,214,200,764,765,759,567,361,407,349,611,605,188,74,48,613,175,161,439,363,1032,559,203,565,571,940,799,202,564,558,362,606,174,148,49,75,71,65,616,602,158,399,414,372,366,400,428,560,206,574,945,789,777,788,575,561,367,401,415,398,159,165,824,64,830,818,198,601,99,615,403,371,359,577,211,775,749,748,238,576,210,1009,589,402,628,614,98,600,73,9,698,14,28,129,673,317,471,288,511,263,504,510,276,464,302,458,672,100,666,896,17,116,670,328,472,300,466,506,260,704,705,513,507,261,473,103,117,659,842,16,12,675,661,339,503,259,729,700,258,270,310,890,106,660,674,39,879,689,104,110,676,138,306,312,500,528,703,717,918,267,273,475,313,461,139,677,105,38,691,108,646,493,444,295,281,518,530,901,929,732,243,531,257,519,280,323,479,135,109,36,692,137,889,651,645,123,484,453,447,725,731,724,718,240,268,297,283,452,644,136,687,683,27,654,668,456,318,287,244,522,536,250,720,734,735,251,245,292,286,480,669,655,133,26,32,682,680,18,24,30,496,327,441,333,469,1076,521,938,723,736,520,246,285,332,326,483,124,25,695,681,4,56,816,81,619,625,369,341,427,433,355,584,209,553,221,778,546,552,585,432,340,426,156,94,618,80,195,181,57,7,197,183,96,82,154,140,626,395,381,356,430,587,544,550,236,791,746,974,551,586,343,419,380,633,155,97,182,40,814,54,6,2,50,44,623,179,421,409,596,227,233,780,794,757,554,540,226,1017,583,408,346,434,352,391,92,144,150,45,51,79,191,47,53,146,620,634,608,90,422,378,1001,595,581,556,783,740,543,231,379,423,609,85,153,621,184]
        # list7=[1145,957,995,836,808,1026,954,968,997,955,941,438,819,831,825,1143,992,993,987,950,944,978,1022,1036,1140,1034,952,991,985,990,947,1141,827,854,868,1132,1052,1091,908,1090,1084,1127,1133,882,855,857,894,1131,1045,923,937,936,1093,1044,1078,1130,1124,1118,881,895,856,846,891,885,1134,1040,1083,926,932,933,1082,1096,1055,1109,853,845,886,892,1094,931,1081,1042,893,887,878,875,849,1113,1107,1067,1098,915,928,900,1066,1106,1112,874,1138,1110,1058,1064,1065,1111,873,1101,1115,1075,1061,907,913,906,1060,864,1102,1062,910,904,905,911,1088,1077,1063,1117,865,1004,1010,976,989,988,963,1005,1013,975,948,810,804,964,965,807,1149,1015,1029,998,999,1028,1014,812,806]
        # list8=[88,229,89,163,607,410,809,821,212,979,762,563,774,204,1021,370,172,459,303,505,909,1047,843,658,711,739,275,671,305,1069,851,448,1056,685,450,653,848,862,490,309,321,719,903,122,37,278,866,872,643,535,252,792,787,793,591,354,790,237,1006,83,169,78,743,597,557,345,392]
        # list9=[758,177,1033,1147,835,562,833,316,1087,852,307,844,861,1072,255,308,491,888,537,1114,871,977,817,632,961,960]

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB1 = Image.open(AB_path).convert('L')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        A1 = AB1.crop((0, 0, w2, h))
        A1 = np.array(A1)
        A1 = np.stack((A1,)*3, axis=-1)
        A1 = Image.fromarray(np.uint8(A1))
        B1 = AB1.crop((w2, 0, w, h))
        B1 = np.array(B1)
        B1 = np.stack((B1,)*3, axis=-1)
        B1 = Image.fromarray(np.uint8(B1))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        A1 = A_transform(A1)
        B = B_transform(B)
        B1 = B_transform(B1)

        flag=0
        num=0
        for i in AB_path:
            if i>='0' and i<='9':
                flag=1
                num=num*10+int(i)
            else:
                if flag==1:
                    break
        flag=[]
        if num in list1 and num in list2:
            flag=[1,0,0,0]
        elif num in list1 and num not in list2:
            flag=[0,1,0,0]
        elif num not in list1 and num in list2:
            flag=[0,0,1,0]
        else:
            flag=[0,0,0,1]
        # elif num in list5:
        #     flag=5
        # elif num in list6:
        #     flag=6
        # elif num in list7:
        #     flag=7
        # elif num in list8:
        #     flag=8
        # elif num in list9:
        #     flag=9

        return {'A': A, 'B': B, 'Ag': A1, 'Bg': B1, 'A_paths': AB_path, 'B_paths': AB_path, 'flag':torch.FloatTensor(flag)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
