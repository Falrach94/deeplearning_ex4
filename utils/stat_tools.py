import numpy as np
import torch


def calc_profile_f1(profile):

    f1_raw = [s.f1 for s in profile.sessions]
    f1_c_raw = [s.f1_crack for s in profile.sessions]
    f1_i_raw = [s.f1_inactive for s in profile.sessions]

    return calc_mean(profile, f1_raw), calc_mean(profile, f1_c_raw), calc_mean(profile, f1_i_raw),


def categorize_data(data):
    fine = []
    not_fine = []
    data = np.array(data)
    for l in data:
        if l[1] == 1 or l[2] == 1:
            not_fine.append(l)
        else:
            fine.append(l)
    return fine, not_fine


def fully_categorize_data(data):
    cat = [[],[],[],[]]
    data = np.array(data)
    for l in data:
        if l[1] == 1 and l[2] == 1:
            cat[3].append(l)
        elif l[1] == 1:
            cat[1].append(l)
        elif l[2] == 1:
            cat[2].append(l)
        else:
            cat[0].append(l)
    return cat

def calc_mean(profile, raw):

    if len(raw) == 0:
        return np.zeros((2,1))

    max_len = max([len(ar) for ar in raw])
    session_cnt = len(profile.sessions)
    acc = np.full((session_cnt, max_len), np.nan)

    for i in range(session_cnt):
        acc[i, :len(raw[i])] = raw[i]

    means = np.nanmean(acc, axis=0)
 #   std = np.nanstd(acc, axis=0)
 #   result = np.stack((means, std)).transpose()
    return means


def calc_data_stats(session):
    tr_data = session.tr_data
    v_data = session.val_data

    tr_cat = fully_categorize_data(tr_data)
    val_cat = fully_categorize_data(v_data)

    res = [[len(tr_data), len(v_data)],
           [len(tr_cat[0]), len(val_cat[0])],
           [len(tr_cat[1]), len(val_cat[1])],
           [len(tr_cat[2]), len(val_cat[2])],
           [len(tr_cat[3]), len(val_cat[3])]]
    return res


def calc_multi_f1_conf(prediction, label):

    f1 = calc_multi_f1(prediction, label)
    conf = calc_multi_conf(prediction, label)

    f1['crack'].update(conf['crack'])
    f1['inactive'].update(conf['inactive'])
    return f1


def calc_f1_from_4class(pred, label):
    pred = pred > 0.5
    pred = torch.stack((pred[:, 1] | pred[:, 3],
                        pred[:, 2] | pred[:, 3])).transpose(0,1)
    label = label > 0.5
    label = torch.stack((label[:, 1] | label[:, 3],
                          label[:, 2] | label[:, 3])).transpose(0,1)

    stats = calc_stats(pred, label)
    return {'stats': stats,
            'mean': np.mean([stat['f1'] for stat in stats])}


def calc_stats(bool_pred, bool_label):
    tp = np.logical_and(bool_pred, bool_label)
    tn = np.logical_and(np.invert(bool_pred), np.invert(bool_label))
    fp = np.logical_and(bool_pred, np.invert(bool_label))
    fn = np.logical_and(np.invert(bool_pred), bool_label)

    tp = tp.sum(dim=0)
    tn = tn.sum(dim=0)
    fp = fp.sum(dim=0)
    fn = fn.sum(dim=0)

    accuracy = [((tpv + tnv) / (bool_pred.shape[0])).item() for tpv, tnv in zip(tp, tn)]

    precision = [np.nan if tpv+fpv == 0 else (tpv/(tpv+fpv)).item()
                 for tpv, fpv in zip(tp, fp)]

    recall = [np.nan if tpv+fnv == 0 else (tpv/(tpv+fnv)).item()
              for tpv, fnv in zip(tp, fn)]

    nan = [np.isnan(r) or np.isnan(p) or p + r == 0
           for p, r in zip(precision, recall)]

    stats = [{'tp': tpv.item(), 'tn': tnv.item(), 'fp': fpv.item(), 'fn': fnv.item(),
              'precision': p, 'recall': r,
              'accuracy': acc,
              'f1': (0 if nanv else 2*p*r/(p+r))}
             for tpv, tnv, fpv, fnv, nanv, p, r, acc in zip(tp, tn, fp, fn, nan, precision, recall, accuracy)]

    return stats


def calc_f1_m(pred, label):

    if pred.shape[0] == 4:
        classical_stats = calc_f1_from_4class(pred, label)
    else:
        classical_stats = None

    stats = calc_stats(pred > 0.5, label > 0.5)
    return {
        'stats': stats,
        'mean': np.mean([stat['f1'] for stat in stats]),
        'classical': classical_stats
    }

'''
def calc_f1_pure(pred, label):
    pred = pred > 0.5
    label = label > 0.5

    tp = np.logical_and(pred, label)
    tn = np.logical_and(np.invert(pred), np.invert(label))
    fp = np.logical_and(pred, np.invert(label))
    fn = np.logical_and(np.invert(pred), label)

    tp = tp.sum(dim=0)
    tn = tn.sum(dim=0)
    fp = fp.sum(dim=0)
    fn = fn.sum(dim=0)

    precision = [np.nan if tpv+fpv == 0 else (tpv/(tpv+fpv)).item()
                 for tpv, fpv in zip(tp, fp)]

    recall = [np.nan if tpv+fnv == 0 else (tpv/(tpv+fnv)).item()
              for tpv, fnv in zip(tp, fn)]

    nan = [np.isnan(r) or np.isnan(p) or p + r == 0
           for p, r in zip(precision, recall)]

    stats = [{'tp': tpv.item(), 'tn': tnv.item(), 'fp': fpv.item(), 'fn': fnv.item(),
              'precision': p, 'recall': r,
              'f1': (0 if nanv else 2*p*r/(p+r))}
             for tpv, tnv, fpv, fnv, nanv, p, r in zip (tp, tn, fp, fn, nan, precision, recall)]
    return {
        'stats': stats,
        'mean': np.mean([stat['f1'] for stat in stats]),
        'classical': None
    }
'''


def calc_multi_f1(prediction, label):

    if prediction.size(1) == 1:
        stat_i = calc_f1(prediction, label)
        stat_c = stat_i
    else:
        stat_c = calc_f1(prediction[:, 0], label[:, 0])
        stat_i = calc_f1(prediction[:, 1], label[:, 1])

    f1 = (stat_c['f1'] + stat_i['f1'])/2

    return {'crack': stat_c, 'inactive': stat_i, 'mean': f1}


def calc_multi_conf(pred, label):
    stat_c = calc_conf(pred[:, 0], label[:, 0])
    stat_i = calc_conf(pred[:, 1], label[:, 1])

    return {'crack': stat_c, 'inactive': stat_i}


def calc_f1(pred, label):
    pred = pred > 0.5
    label = label > 0.5

    tp = np.logical_and(pred, label)
    tn = np.logical_and(np.invert(pred), np.invert(label))
    fp = np.logical_and(pred, np.invert(label))
    fn = np.logical_and(np.invert(pred), label)

    tp = tp.sum().item()
    tn = tn.sum().item()
    fp = fp.sum().item()
    fn = fn.sum().item()

    if tp + fp == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = np.nan
    else:
        recall = tp/(tp + fn)

    if np.isnan(recall) or np.isnan(precision) or precision + recall == 0:
        return {'f1': 0,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    return {'f1': 2*precision*recall/(precision+recall),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def calc_conf(pred, label, threshold=0.25):

    pred_bin = pred > 0.5
    label_bin = label > 0.5

    fp = np.logical_and(pred_bin, np.invert(label_bin))
    fn = np.logical_and(np.invert(pred_bin), label_bin)

    hc_fp = np.logical_and(fp, (pred > 1-threshold)).sum().item()
    hc_fn = np.logical_and(fn, (pred < threshold)).sum().item()

    return {'hc_fp': hc_fp, 'hc_fn': hc_fn}
