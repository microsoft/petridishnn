import argparse
import time
import traceback

from tensorpack.utils import logger
import petridish.data.openml as openml
from petridish.data.misc import preprocess_data_flow

def interval_from_str(ss):
    start_end = ss.strip().split('..')
    if len(start_end) == 1:
        return [ int(start_end[0]) ]
    elif len(start_end) == 2:
        start, end = start_end
        return list(range(int(start), int(end) + 1))
    else:
        raise ValueError("Invalid interval {}".format(ss))

def indices_from_str(ss):
    indices = []
    for interval_str in ss.strip().split(','):
        interval_list = interval_from_str(interval_str)
        indices.extend(interval_list)
    return indices


# copy paste from the paper: https://arxiv.org/pdf/1802.04064.pdf
dataset_id_strs = \
"""3, 6, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23, 26, 28, 30, 31, 32, 36, 37, 39, 40, 41,
43, 44, 46, 48, 50,
53, 54, 59, 60, 61, 62, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 180,
181, 182, 183,
184, 187, 189, 197, 209, 223, 227, 273, 275, 276, 277, 278, 279, 285, 287, 292, 293, 294, 298, 300,
307, 310, 312, 313, 329, 333, 334, 335, 336, 337, 338, 339, 343, 346, 351, 354, 357, 375, 377, 383,
384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 444, 446,
448, 450, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 472, 475, 476, 477, 478, 479, 480,
554, 679, 682, 683, 685, 694, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726,
727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746, 747, 748,
749, 750, 751, 752, 753, 754, 755, 756, 758, 759, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770,
771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 782, 783, 784, 785, 787, 788, 789, 790, 791, 792,
793, 794, 795, 796, 797, 799, 800, 801, 803, 804, 805, 806, 807, 808, 811, 812, 813, 814, 815, 816,
817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 832, 833, 834, 835, 836, 837,
838, 841, 843, 845, 846, 847, 848, 849, 850, 851, 853, 855, 857, 859, 860, 862, 863, 864, 865, 866,
867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 884, 885, 886, 888,
891, 892, 893, 894, 895, 896, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913,
914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 932, 933, 934,
935, 936, 937, 938, 941, 942, 943, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 958,
959, 962, 964, 965, 969, 970, 971, 973, 974, 976, 977, 978, 979, 980, 983, 987, 988, 991, 994, 995,
996, 997, 1004, 1005, 1006, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1019, 1020, 1021, 1022,
1025, 1026, 1036, 1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1054, 1055, 1056,
1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1077, 1078,
1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1100, 1104, 1106, 1107, 1110, 1113,
1115, 1116, 1117, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132,
1133, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149,
1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165,
1166, 1169, 1216, 1217, 1218, 1233, 1235, 1236, 1237, 1238, 1241, 1242, 1412, 1413, 1441, 1442,
1443, 1444, 1449, 1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1470, 1471, 1472, 1473,
1475, 1481, 1482, 1483, 1486, 1487, 1488, 1489, 1496, 1498"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openml_idx', type=str, default=None)
    parser.add_argument('--cbb_openml_idx', type=str, default=None)
    parser.add_argument('--data_dir', default='/home/hanzhang/data')
    parser.add_argument('--do_validation', default=False, action='store_true')

    args = parser.parse_args()
    args.batch_size = 16
    args.nr_gpu = 1

    assert not (args.openml_idx is None and args.cbb_openml_idx is None), \
        'should specify eith openml or cbb_openml '

    cbb_idx_to_openml_idx = \
        list(map(int, dataset_id_strs.replace('\n', ' ').replace(',', ' ').strip().split()))

    if args.openml_idx is not None:
        args.openml_idx = indices_from_str(args.openml_idx)
    else:
        args.openml_idx = []
    if args.cbb_openml_idx is not None:
        args.cbb_openml_idx = indices_from_str(args.cbb_openml_idx)
        args.openml_idx.extend(map(lambda ci : cbb_idx_to_openml_idx[ci],
            args.cbb_openml_idx))

    failed_indices = []
    failed_message = []
    for cbb_idx, openml_idx in enumerate(args.openml_idx):
        logger.info("Doing {} / {}".format(cbb_idx, len(args.openml_idx)))
        start_time = time.time()
        try:
            (l_ds, args.mlp_input_types,
                args.mlp_input_dims, n_data, args.num_classes,
                args.mlp_feat_means, args.mlp_feat_stds
            ) = openml.get_openml_dataflow(
                openml_idx, args.data_dir, splits=['train','val'],
                do_validation=args.do_validation
            )

            ds_train = preprocess_data_flow(l_ds['train'], args, True)
            ds_val = preprocess_data_flow(l_ds['val'], args, False)
        except:
            failed_indices.append(openml_idx)
            traceback.print_exc()
            logger.info("Failed {}".format(openml_idx))

        end_time = time.time()
        logger.info("openml_{} finished loading in {:2.2f} sec".format(
            openml_idx, end_time - start_time))

    logger.info("The following openml_idx failed: \n{}".format(failed_indices))