from Visual import easy_time, visual_box, visual_mulbox, visual_box_site, visual_aa


# SW ##########################################################################
# zero
def sw_zero_site(eachmatrix, lmda):
    l = int((lmda - 1) / 2)
    mid = visual_mulbox(l, len(eachmatrix[0]))
    return mid + eachmatrix + mid


# expand
def sw_expand_site(mid):
    out = []
    for i in mid:
        out += i
    return out


# extract site
def sw_extract_site(eachtype, eachmatrix, lmda):
    site_box = []
    for i in range(len(eachtype)):
        mid = eachmatrix[i:i + lmda]
        mid = sw_expand_site(mid)
        site_box.append(mid)
    return site_box
    
    
# Sliding window main
def feature_sw_site(raa_features, lmda):
    sw_features = []
    start_e = 0
    for i in range(len(raa_features)):
        start_e += 1
        easy_time(start_e, len(raa_features))
        eachraa = raa_features[i]
        mid = []
        for j in range(len(eachraa)):
            eachfile = eachraa[j]
            # zero factor
            eachmatrix = sw_zero_site(eachfile[1], int(lmda))
            # matrix faeature
            eachsite_fs = sw_extract_site(eachfile[0], eachmatrix, int(lmda))
            mid.append(eachsite_fs)
        sw_features.append(mid)
    return sw_features


# KMER ########################################################################
# combine
def kmer_com_site(li):
    out = ''
    for i in li:
        out += i
    return out


# dictionary
def kmer_dic_site(pssm_aaid, lmda):
    kmer_dic = []
    start_e = 0
    for i in pssm_aaid:
        start_e += 1
        easy_time(start_e, len(pssm_aaid))
        for j in range(len(i[:-lmda])):
            mid = kmer_com_site(i[j:j + lmda])
            if mid not in kmer_dic:
                kmer_dic.append(mid)
    return kmer_dic
            

# extract
def kmer_extract_site(eachfile, kmer_dic, lmda, k):
    l = int((lmda - 1) / 2)
    mid = visual_box_site(l, 'A')
    eachfile = mid + eachfile + mid
    site_box = []
    for i in range(len(eachfile)-lmda + 1):
        my_site = visual_box(len(kmer_dic))
        mid = kmer_com_site(eachfile[i:i+lmda])
        for j in range(len(mid) - k + 1):
            if mid[j:j + k] in kmer_dic:
                my_site[kmer_dic.index(mid[j:j + k])] += 1
        site_box.append(my_site)
    return site_box
        

# kmer main
def feature_kmer_site(pssm_aaid, lmda, k, raa):
    kmer_features = []
    # kmer 字典
    kmer_dic = kmer_dic_site(pssm_aaid, k)
    start_e = 0
    for i in range(raa):
        start_e += 1
        easy_time(start_e, raa)
        mid = []
        for j in range(len(pssm_aaid)):
            eachfile = pssm_aaid[j]
            # kmer feature
            eachsite_fs = kmer_extract_site(eachfile, kmer_dic, lmda, k)
            mid.append(eachsite_fs)
        kmer_features.append(mid)
    return kmer_features


# DTPSSM ######################################################################
# turn
def dtpssm_turn_site(pssm_matrixes):
    aa = visual_aa()
    pssm_aaid = []
    for i in pssm_matrixes:
        mid = []
        for j in i:
            mid.append(aa[j.index(max(j))])
        pssm_aaid.append(mid)
    return pssm_aaid
    
    
# combine
def dtpssm_com_site(li):
    out = ''
    for i in li:
        out += i
    return out


# dictionary
def dtpssm_dic_site(pssm_aaid, lmda):
    kmer_dic = []
    start_e = 0
    for i in pssm_aaid:
        start_e += 1
        easy_time(start_e, len(pssm_aaid))
        for j in range(len(i[:-lmda])):
            mid = dtpssm_com_site(i[j:j + lmda])
            if mid not in kmer_dic:
                kmer_dic.append(mid)
    return kmer_dic
            

# extract
def dtpssm_extract_site(eachfile, kmer_dic, lmda, k):
    l = int((lmda - 1) / 2)
    mid = visual_box_site(l, 'A')
    eachfile = mid + eachfile + mid
    site_box = []
    for i in range(len(eachfile)-lmda + 1):
        my_site = visual_box(len(kmer_dic))
        mid = dtpssm_com_site(eachfile[i:i+lmda])
        for j in range(len(mid) - k + 1):
            if mid[j:j + k] in kmer_dic:
                my_site[kmer_dic.index(mid[j:j + k])] += 1
        site_box.append(my_site)
    return site_box


# dtpssm main
def feature_dtpssm_site(pssm_matrixes, lmda, k, raa):
    dtpssm_features = []
    # dtpssm 共识序列
    pssm_aaid = dtpssm_turn_site(pssm_matrixes)
    # kmer 字典
    kmer_dic = dtpssm_dic_site(pssm_aaid, k)
    start_e = 0
    for i in range(raa):
        start_e += 1
        easy_time(start_e, raa)
        mid = []
        for j in range(len(pssm_aaid)):
            eachfile = pssm_aaid[j]
            # kmer feature
            eachsite_fs = dtpssm_extract_site(eachfile, kmer_dic, lmda, k)
            mid.append(eachsite_fs)
        dtpssm_features.append(mid)
    return dtpssm_features
