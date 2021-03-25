import tensorflow as tf


def bbox_overlap_tf(bboxes1, bboxes2):
    """Calculate Intersection over Union (IoU) between two sets of bounding
    boxes.
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    #with tf.name_scope('bbox_overlap'):
    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    intersection = (
        tf.maximum(xI2 - xI1 + 1., 0.) *
        tf.maximum(yI2 - yI1 + 1., 0.)
    )

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - intersection

    iou = tf.maximum(intersection / union, 0)

    return iou


def get_width_upright(bboxes):
    #with tf.name_scope('BoundingBoxTransform/get_width_upright'):
    bboxes = tf.cast(bboxes, tf.float32)
    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
    width = x2 - x1 + 1.
    height = y2 - y1 + 1.

    # Calculate up right point of bbox (urx = up right x)
    urx = x1 + .5 * width
    ury = y1 + .5 * height

    return width, height, urx, ury


def encode(bboxes, gt_boxes, variances=None):
    #with tf.name_scope('BoundingBoxTransform/encode'):
    (bboxes_width, bboxes_height,
     bboxes_urx, bboxes_ury) = get_width_upright(bboxes)

    (gt_boxes_width, gt_boxes_height,
     gt_boxes_urx, gt_boxes_ury) = get_width_upright(gt_boxes)

    if variances is None:
        variances = [1., 1.]

    targets_dx = (gt_boxes_urx - bboxes_urx)/(bboxes_width * variances[0])
    targets_dy = (gt_boxes_ury - bboxes_ury)/(bboxes_height * variances[0])

    targets_dw = tf.log(gt_boxes_width / bboxes_width) / variances[1]
    targets_dh = tf.log(gt_boxes_height / bboxes_height) / variances[1]

    targets = tf.concat(
        [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)

    return targets


