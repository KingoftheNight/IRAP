# import packages
import os
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
raac_path = os.path.join(file_path, 'raacDB')
import sys
sys.path.append(file_path)
import Load as iload
import Blast as iblast
import Extract as iextra
import SVM as isvm
import Evaluate as ieval


def read_read(file, out):
    pos_sq = iload.load_fasta_folder(file=file, out=out)

def read_blast(path, db, n, ev, out):
    pos_sq = iload.load_fasta_folder(file=None, out=os.path.join(os.path.join(now_path, 'Reads'), path))
    pos_pssm = iblast.blast_psiblast_folder(pos_sq, db, n, ev, name=path, vi=False)

def read_extract_raabook(path1, path2, out, raa):
    pos_pssm = iload.load_reload_folder(os.path.join(os.path.join(now_path, 'PSSMs'), path1))
    neg_pssm = iload.load_reload_folder(os.path.join(os.path.join(now_path, 'PSSMs'), path2))
    pssm_path = iextra.extract_pssm(pos=pos_pssm, neg=neg_pssm, reduce=True, raaBook=raa, out=os.path.join(now_path, out))

def read_extract_selfraa(path1, path2, out, raa):
    s = len(raa.split('-'))
    file = 'type 1 size ' + str(s) + ' ' + raa
    with open(os.path.join(raac_path, 'test.txt'), 'w') as f:
        f.write(file)
    pos_pssm = iload.load_reload_folder(os.path.join(os.path.join(now_path, 'PSSMs'), path1))
    neg_pssm = iload.load_reload_folder(os.path.join(os.path.join(now_path, 'PSSMs'), path2))
    pssm_path = iextra.extract_pssm(pos=pos_pssm, neg=neg_pssm, reduce=True, raaBook=os.path.join(raac_path, 'test.txt'), out=os.path.join(now_path, out))
    os.remove(os.path.join(raac_path, 'test.txt'))

def read_grid_file(file):
    best_c, best_g = isvm.svm_grid(file)
    print('Best_C:\t' + str(best_c) + '\tBest_Gamma:\t' + str(best_g))

def read_grid_folder(folder):
    pssm_path = iload.load_reload_folder(os.path.join(now_path, folder))
    isvm.svm_grid_folder(pssm_path, out=os.path.join(now_path, 'Hys_' + folder + '.txt'))

def read_extract_folder(folder, cg, cv, out):
    pssm_path = iload.load_reload_folder(os.path.join(now_path, folder))
    ieval.evaluate_folder(pssm_path, cg=cg, cv=cv, out=os.path.join(now_path, out))

def read_model_save_folder(folder, cg, out):
    pssm_path = iload.load_reload_folder(os.path.join(now_path, folder))
    iload.load_model_save_folder(pssm_path, cg=cg, out=os.path.join(now_path, out))

