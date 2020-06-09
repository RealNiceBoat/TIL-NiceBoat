import numpy as np
def encode_label(label_arr, dims_list, aspect_ratios, iou_fn, sampling_fn, cat_list):
    num_entries = 7 + len(cat_list) # objectness, ... len(cat_list) ..., dx, dy, dw, dh, obj_indicator, catloc_indicator
    np_labels = {}
    for dims in dims_list:
        dimkey = '{}x{}'.format(*dims)
        np_labels[dimkey] = np.zeros( (*dims, len(aspect_ratios), num_entries ) )

    for label in label_arr:
        gtclass, gtx, gty, gtw, gth = label
        gtclass = int(gtclass)
        gt_bbox = [gtx, gty, gtw, gth]
        
        iou_scores_dict = {}

        for dims in dims_list:
            key = '{}x{}'.format(*dims)
        
            kx,ky = dims
            gapx = 1.0 / kx
            gapy = 1.0 / ky
            '''
            There are kx x ky tiles. 
            For now, all have the same w,h of gapx,gapy. 
            For the (i,j)-th tile, x = 0.5*gapx + i*gapx = (0.5+i)*gapx | y = (0.5+j)*gapy
            '''
            for i in range(kx):
                for j in range(ky):
                    for k in range( len(aspect_ratios) ):
                        dims_aspect_key = (*dims, k) # a 3-tuple: (dim1,dim2,ar)
                        if dims_aspect_key not in iou_scores_dict:
                            iou_scores_dict[dims_aspect_key] = []
                        x = (0.5+i)*gapx
                        y = (0.5+j)*gapy

                        # Different aspect ratios alter the anchor box default dimensions
                        w = gapx * aspect_ratios[k][0]
                        h = gapy * aspect_ratios[k][1]
                        cand_bbox = [x,y,w,h]

                        # SSD formulation
                        dx = (gtx - x) / w 
                        dy = (gty - y) / h
                        dw = np.log( gtw / w )
                        dh = np.log( gth / h )
                        
                        int_over_union = iou_fn( cand_bbox, gt_bbox )
                        iou_scores_dict[dims_aspect_key].append( (int_over_union, key, i, j, k, dx, dy, dw, dh) )
            sampling_fn( iou_scores_dict, np_labels, gtclass, cat_list )
    return np_labels

def decode_tensor(pred_dict, aspect_ratios):
    results = []
    for dim_str, pred_tensor in pred_dict.items():
        pred_tensor = pred_tensor[0] # remove the batch
        kx, ky = [int(g) for g in dim_str.split('x')]
        gapx = 1. / kx
        gapy = 1. / ky

        # We trained without activations, so we need to process the logits into probabilities/scores
        pred_arr = np.array(pred_tensor)
        obj_logits = pred_arr[:,:,:,0]
        obj_scores = 1. / (1 + np.exp(-obj_logits))
        pred_arr[:,:,:,0] = obj_scores
        
        cls_logits = pred_arr[:,:,:,1:-4]
        cls_scores = np.exp(cls_logits)
        cls_scores = cls_scores / cls_scores.sum(axis=-1)[...,np.newaxis]
        pred_arr[:,:,:,1:-4] = cls_scores

        for k, ar in enumerate(aspect_ratios):
            for i in range(kx):
                for j in range(ky):
                    cx = (0.5+i)*gapx
                    cy = (0.5+j)*gapy
                    w = gapx * ar[0]
                    h = gapy * ar[1]

                    payload = pred_arr[i,j,k]
                    obj_score = payload[0]
                    dx, dy, dw, dh = payload[-4:]
                    cls_probs = payload[1:-4]

                    predx = (dx * w) + cx
                    predy = (dy * h) + cy
                    predw = w * np.exp( dw )
                    predh = h * np.exp( dh )
                    max_cls_idx = np.argmax( cls_probs )
                    max_cls_prob = cls_probs[max_cls_idx]
                    category_id = max_cls_idx + 1
                    det_score = obj_score * max_cls_prob
                    results.append( (det_score, category_id, predx, predy, predw, predh) )
    return results