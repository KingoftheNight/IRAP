# import packages
from math import sqrt


def eval_score(true, tfpn):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(true)):
        if tfpn[j] == 0.0 and true[j] == 0.0:
            tp += 1
        if tfpn[j] == 1.0 and true[j] == 1.0:
            tn += 1
        if tfpn[j] == 0.0 and true[j] == 1.0:
            fp += 1
        if tfpn[j] == 1.0 and true[j] == 0.0:
            fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    if (tp + fn) == 0 or (tn + fp) == 0 or (tp + fp) == 0 or (tn + fn) == 0:
        return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % 0), float('%.3f' % 0), float('%.3f' % 0), float(
            '%.3f' % 0)
    else:
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        ppv = tp / (tp + fp)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
        return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % sn), float('%.3f' % sp), float(
            '%.3f' % ppv), float('%.3f' % mcc)