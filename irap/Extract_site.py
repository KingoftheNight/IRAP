# import packages
import os
import math
from Feature_site import feature_sw_site, feature_kmer_site, feature_dtpssm_site
from Visual import easy_time, visual_mulbox, visual_aa
from Read import read_raac, read_pssm_site
extract_path = os.path.dirname(__file__)


# 矩阵列相加
def extract_col_plus(type_box, eachfile, i, j):
    for m in range(len(type_box)):
        type_box[m][j] += eachfile[m][i]
    return type_box


# 平均
def extract_average(type_box, raa_box):
    for i in range(len(type_box)):
        for j in range(len(type_box[i])):
            type_box[i][j] = round(type_box[i][j] / len(raa_box[j]), 4)
    return type_box


# 获取raac
def extract_raa(raa, outfolder, now_path):
    # 获取氨基酸约化密码表
    raa_path = os.path.join(extract_path, 'raacDB')
    raa_file = os.path.join(raa_path, raa)
    if raa in os.listdir(raa_path):
        raacode = read_raac(raa_file)
        if outfolder not in os.listdir(now_path):
            outfolder = os.path.join(now_path, outfolder)
            os.makedirs(outfolder)
        else:
            outfolder = os.path.join(now_path, outfolder)
    else:
        with open(raa_file, 'w') as f:
            f.write('type 1 size ' + str(len(raa.split('-'))) + ' ' + raa)
        raacode = read_raac(raa_file)
        if outfolder not in os.listdir(now_path):
            outfolder = now_path
        else:
            outfolder = now_path
        os.remove(raa_file)
    return raacode, outfolder


# str to float
def extract_change(pssm_matrixes):
    out_box = []
    for i in range(len(pssm_matrixes)):
        mid_box = []
        for j in range(len(pssm_matrixes[i])):
            next_box = []
            for k in range(len(pssm_matrixes[i][j])):
                next_box.append(float(pssm_matrixes[i][j][k]))
            mid_box.append(next_box)
        out_box.append(mid_box)
    return out_box


# 提取矩阵特征
def extract_features(pssm_matrixes, pssm_aaid):
    all_features = []
    start_e = 0
    for i in range(len(pssm_matrixes)):
        start_e += 1
        easy_time(start_e, len(pssm_matrixes))
        each_matrix = pssm_matrixes[i]
        matrix_400 = []
        aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for aa in aa_index:
            aa_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0]
            for j in range(len(each_matrix)):
                line = each_matrix[j]
                if pssm_aaid[i][j] == aa:
                    for k in range(len(line)):
                        aa_score[k] = aa_score[k] + line[k]
            matrix_400.append(aa_score)
        all_features.append(matrix_400)
    return all_features


# long to short
def extract_short(pssm_features):
    out_box = []
    for i in range(len(pssm_features)):
        mid_box = []
        for j in range(len(pssm_features[i])):
            next_box = []
            for k in range(len(pssm_features[i][j])):
                next_box.append(float('%.3f' % pssm_features[i][j][k]))
            mid_box.append(next_box)
        out_box.append(mid_box)
    return out_box


# 约化矩阵
def extract_reduce(pssm_matrixes, raacode, pssm_type):
    all_features = []
    aa_index = visual_aa()
    start_e = 0
    for raa in raacode[1][:5]:
        start_e += 1
        easy_time(start_e, len(raacode[1]))
        raa_box = raacode[0][raa]
        mid_box = []
        for k in range(len(pssm_matrixes)):
            eachfile = pssm_matrixes[k]
            eachtype = pssm_type[k]
            # 列合并
            type_box = visual_mulbox(len(eachfile), len(raa_box))
            for i in range(len(aa_index)):
                for j in range(len(raa_box)):
                    if aa_index[i] in raa_box[j]:
                        type_box = extract_col_plus(type_box, eachfile, i, j)
            # 平均
            # type_box = extract_average(type_box, raa_box)
            mid_box.append([eachtype, type_box])
        all_features.append(mid_box)
    return all_features


# 矩阵转置
def extract_transform(data):
    new_data = []
    for i in data:
        new_data += i
    return new_data


# 归一化
def extract_scale(data):
    new_data = visual_mulbox(len(data), len(data[0]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] >= - 709:
                f = 1 / (1 + math.exp(-data[i][j]))
                new_data[i][j] = round(f, 4)
            else:
                f = 1 / (1 + math.exp(-709))
                new_data[i][j] = round(f, 4)
                # new_data.append('%.6f' % f)
    return new_data


# combine features
def extract_combine(line, tp):
    out = tp
    for i in range(len(line)):
        out += ' ' + str(i + 1) + ':' + str(line[i])
    return out


# expand
def extract_expand_site(data1, data2):
    for i in range(len(data1)):
        data1[i] += data2[i]
    return data1


# 保存特征
def extract_save(raa_features, sw_features, kmer_features, dtpssm_features, outfolder, raa_list):
    start_e = 0
    for k in range(len(raa_features)):
        start_e += 1
        easy_time(start_e, len(raa_features))
        eachraa = raa_features[k]
        out_file = ''
        for i in range(len(eachraa)):
            eachfile = eachraa[i]
            type_m = eachfile[0]
            data_m = extract_expand_site(eachfile[1], sw_features[k][i])
            data_m = extract_expand_site(data_m, kmer_features[k][i])
            data_m = extract_expand_site(data_m, dtpssm_features[k][i])
            # 归一化
            data_m = extract_scale(data_m)
            for j in range(len(data_m)):
                line = extract_combine(data_m[j], type_m[j])
                out_file += line + '\n'
        path = os.path.join(outfolder, raa_list[k] + '.rap')
        with open(path, 'w') as f2:
            f2.write(out_file[:-1])
            f2.close()


# extract main
def extract_main_site(pssm_name, site_name, outfolder, raa, lmda, now_path):
    # 获取raac
    raacode, outfolder = extract_raa(raa, outfolder, now_path)
    # 处理地址
    sequence = os.path.join(os.path.join(now_path, 'PSSMs'), pssm_name)
    site = os.path.join(now_path, site_name)
    # PSSM
    pssm_matrixes, pssm_aaid, pssm_type = read_pssm_site(sequence, [], [], [], site)
    # 矩阵约化
    raa_features = extract_reduce(pssm_matrixes, raacode, pssm_type)
    # PSSM位点滑窗
    sw_features = feature_sw_site(raa_features, lmda)
    # kmer字典
    k = 2
    kmer_features = feature_kmer_site(pssm_aaid, lmda, k, len(raa_features))
    # dtpssm 共识
    k = 2
    dtpssm_features = feature_dtpssm_site(pssm_matrixes, lmda, k, len(raa_features))
    # 生成特征文件
    extract_save(raa_features, sw_features, kmer_features, dtpssm_features, outfolder, raacode[1])
