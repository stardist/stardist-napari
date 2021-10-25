import numpy as np
from skimage.measure import regionprops

def match_labels(x, y, iou_threshold=.2):
    """ ensure that matching objects in x and y have some id, return a relabeled y"""
    from stardist.matching import matching
    
    res = matching(x,y, report_matches=True, thresh=0)

    pairs = tuple(p for p,s in zip(res.matched_pairs, res.matched_scores) if s>= iou_threshold)
    map_dict = dict((i2,i1) for i1,i2 in pairs)
    
    # # the indices in y that should be mapped
    # ind_y_plus = set(map_dict.keys())
    # # the indices in y that have no matching partner
    # ind_y_minus = set(np.unique(y)) - {0} - ind_y_plus
    
    y2 = np.zeros_like(y)

    y_labels = set(np.unique(y)) - {0}

    # labels that can be used for non-matched regions
    label_reservoir = list(set(np.arange(1,len(y_labels)+1)) - set(map_dict.values()))

    for r in regionprops(y):
        m = (y[r.slice] == r.label)
        if r.label in map_dict:
            y2[r.slice][m] = map_dict[r.label]
        else:
            y2[r.slice][m] = label_reservoir.pop(0)

    return y2


# def match_labels(x, y, iou_threshold=.2):
    
#     return y2
