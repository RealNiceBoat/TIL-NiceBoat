# Computes the intersection-over-union (IoU) of two bounding boxes
def iou(bb1, bb2):
  x1,y1,w1,h1 = bb1
  xmin1 = x1 - w1/2
  xmax1 = x1 + w1/2
  ymin1 = y1 - h1/2
  ymax1 = y1 + h1/2

  x2,y2,w2,h2 = bb2
  xmin2 = x2 - w2/2
  xmax2 = x2 + w2/2
  ymin2 = y2 - h2/2
  ymax2 = y2 + h2/2

  area1 = w1*h1
  area2 = w2*h2

  # Compute the boundary of the intersection
  xmin_int = max( xmin1, xmin2 )
  xmax_int = min( xmax1, xmax2 )
  ymin_int = max( ymin1, ymin2 )
  ymax_int = min( ymax1, ymax2 )
  intersection = max(xmax_int - xmin_int, 0) * max( ymax_int - ymin_int, 0 )

  # Remove the double counted region
  union = area1+area2-intersection

  return intersection / union


# Sampling schemes
def yolo_posneg_sampling(iou_scores_dict, label_tensor, gtclass, cat_list, iou_threshold=0.5):
  iou_scores = []
  for _, scores in iou_scores_dict.items():
    iou_scores.extend(scores)
  iou_scores.sort( key=lambda x: x[0], reverse=True )
  
  top_iou_score = iou_scores.pop(0)
  _, key, i, j, k, dx, dy, dw, dh = top_iou_score
  zeros = [0] * len(cat_list)
  payload = [1, *zeros, dx,dy,dw,dh]
  payload[gtclass + 1] = 1
  # Train objectness, class and loc for the positive
  label_tensor[key][i,j,k,-2:] = 1
  label_tensor[key][i,j,k,:len(payload)] = payload

  # Train objectness only for the negatives
  low_iou_scores = [iou_score for iou_score in iou_scores if iou_score[0] < iou_threshold]
  for _, key, i, j, k, _, _, _, _ in low_iou_scores:
    label_tensor[key][i,j,k,-2] = 1

def modified_yolo_posneg_sampling(iou_scores_dict, label_tensor, gtclass, cat_list, iou_threshold=0.5):
  iou_scores = []
  zeros = [0] * len(cat_list)

  for _, scores in iou_scores_dict.items():
    iou_scores.extend(scores)
  iou_scores.sort( key=lambda x: x[0], reverse=True )
  
  top_iou_score = iou_scores.pop(0)
  _, key, i, j, k, dx, dy, dw, dh = top_iou_score
  payload = [1, *zeros, dx,dy,dw,dh]
  payload[gtclass + 1] = 1
  # Train objectness, class and loc for the positive
  label_tensor[key][i,j,k,-2:] = 1
  label_tensor[key][i,j,k,:len(payload)] = payload

  # Train objectness only for the negatives
  low_iou_scores = [iou_score for iou_score in iou_scores if iou_score[0] < iou_threshold]
  for _, key, i, j, k, _, _, _, _ in low_iou_scores:
    label_tensor[key][i,j,k,-2] = 1

  # Train cat/loc only for the in-betweens - those with high IoU but not positive
  high_iou_scores = [iou_score for iou_score in iou_scores if iou_score[0] >= iou_threshold]
  for _, key, i, j, k, dx, dy, dw, dh in high_iou_scores:
    label_tensor[key][i,j,k,-1] = 1
    payload = [0,*zeros,dx,dy,dw,dh]
    payload[gtclass + 1] = 1
    label_tensor[key][i,j,k,:len(payload)] = payload

def top_ratio_sampling(iou_scores_dict, label_tensor, gtclass, cat_list, positive_ratio=0.25):
  iou_scores = []
  # Let all tensors learn objectness score
  for v in label_tensor.values():
    v[:,:,:,-2] = 1
  
  for _, iou_score_list in iou_scores_dict.items():
    iou_score_list.sort( key=lambda x: x[0], reverse=True )
    top_percentile_iou_scores = iou_score_list[:round(len(iou_score_list) * positive_ratio)]
    # Include the rest that cross the IoU threshold
    iou_score_list = top_percentile_iou_scores + [iou_score for iou_score in iou_score_list[len(top_percentile_iou_scores):] if iou_score[0] >= self.iou_threshold]
    iou_scores.extend( iou_score_list )

  for iou_score in iou_scores:
    IoU, key, i, j, k, dx, dy, dw, dh = iou_score
    zeros = [0] * len(cat_list)
    payload = [IoU, *zeros, dx,dy,dw,dh]
    payload[gtclass + 1] = 1
    label_tensor[key][i,j,k,:len(payload)] = payload
    # Set the classification/localisation indicator at this location to positive
    label_tensor[key][i,j,k,-1] = 1
