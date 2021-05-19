# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import namedtuple
import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
import importlib


def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with
    python modules used in the evaluation.
    """
    return {
        'math': 'math',
        'numpy': 'np'
    }


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and
    evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT': 0.8,
        'AREA_PRECISION_CONSTRAINT': 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O': 1.,
        'MTYPE_OM_O': 0.8,
        'MTYPE_OM_M': 1.,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'CRLF': False  # Lines are delimited by Windows CRLF format
    }


def validate_data(gt_file_path, sub_file_path, evaluation_params):
    """
    Method validate_data: validates that all files in the results folder are
    correct (have the correct name contents).
                            Validates also that there are no missing files in
                            the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(
        gt_file_path, evaluation_params['GT_SAMPLE_NAME_2_ID'])

    sub = rrc_evaluation_funcs.load_zip_file(
        sub_file_path, evaluation_params['DET_SAMPLE_NAME_2_ID'], True)

    # Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(
            k, gt[k], evaluation_params['CRLF'], True, True)

    # Validate format of results
    for k in sub:
        if not (k in gt):
            raise Exception("The sample %s not present in GT" % k)

        rrc_evaluation_funcs.validate_lines_in_file(
            k, sub[k], evaluation_params['CRLF'], True, False)


def evaluate_method(gt_file_path, sub_file_path, evaluation_param):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex:
        { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex:
        {'sample1' : { 'Precision':0.8,'Recall':0.9 },
        'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    for module, alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recall_mat[0])):
            if recall_mat[row, j] >= evaluation_param['AREA_RECALL_CONSTRAINT']\
                    and precision_mat[row, j] >= \
                    evaluation_param['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if cont != 1:
            return False
        cont = 0
        for i in range(len(recall_mat)):
            if recall_mat[i, col] >= evaluation_param['AREA_RECALL_CONSTRAINT']\
                    and precision_mat[i, col] >= \
                    evaluation_param['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if cont != 1:
            return False

        if recall_mat[row, col] >= evaluation_param['AREA_RECALL_CONSTRAINT']\
                and precision_mat[row, col] >= \
                evaluation_param['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def num_overlaps_gt(gt_num_):
        cont = 0
        for det_num_ in range(len(det_rects)):
            if det_num_ not in det_dont_care_rects_num:
                if recall_mat[gt_num_, det_num_] > 0:
                    cont = cont + 1
        return cont

    def num_overlaps_det(det_num_):
        cont = 0
        for gt_num_ in range(len(recall_mat)):
            if gt_num_ not in gt_dont_care_rects_num:
                if recall_mat[gt_num_, det_num_] > 0:
                    cont = cont + 1
        return cont

    def is_single_overlap(row, col):
        if num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1:
            return True
        else:
            return False

    def one_to_many_match(gt_num_):
        many_sum = 0
        det_rects_ = []
        for det_num_ in range(len(recall_mat[0])):
            if gt_rect_mat[gt_num_] == 0 and det_rect_mat[det_num_] == 0 \
                    and det_num_ not in det_dont_care_rects_num:
                if precision_mat[gt_num_, det_num_] >= \
                        evaluation_param['AREA_PRECISION_CONSTRAINT']:
                    many_sum += recall_mat[gt_num_, det_num_]
                    det_rects_.append(det_num_)
        if round(many_sum, 4) >= evaluation_param['AREA_RECALL_CONSTRAINT']:
            return True, det_rects_
        else:
            return False, []

    def many_to_one_match(det_num_):
        many_sum = 0
        gt_rect_list = []
        for gt_num_ in range(len(recall_mat)):
            if gt_rect_mat[gt_num_] == 0 and det_rect_mat[det_num_] == 0 \
                    and gt_num_ not in gt_dont_care_rects_num:
                if recall_mat[gt_num_, det_num_] >= \
                        evaluation_param['AREA_RECALL_CONSTRAINT']:
                    many_sum += precision_mat[gt_num_, det_num_]
                    gt_rect_list.append(gt_num_)
        if round(many_sum, 4) >= evaluation_param['AREA_PRECISION_CONSTRAINT']:
            return True, gt_rect_list
        else:
            return False, []

    def area(a, b):
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0.

    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x, y)

    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty)

    def center_distance(r1, r2):
        return point_distance(center(r1), center(r2))

    def diag(r):
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    per_sample_metrics = {}

    method_recall_sum = 0
    method_precision_sum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    gt = rrc_evaluation_funcs.load_zip_file(
        gt_file_path, evaluation_param['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(
        sub_file_path, evaluation_param['DET_SAMPLE_NAME_2_ID'], True)

    num_gt = 0
    num_det = 0

    for resFile in gt:

        gt_file = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0
        recall_accum = 0.
        precision_accum = 0.
        gt_rects = []
        det_rects = []
        gt_pol_points = []
        det_pol_points = []
        # Array of Ground Truth Rectangles' keys marked as don't Care
        gt_dont_care_rects_num = []
        # Array of Detected Rectangles' matched with a don't Care GT
        det_dont_care_rects_num = []
        pairs = []
        evaluation_log = ""

        recall_mat = np.empty([1, 1])
        precision_mat = np.empty([1, 1])

        points_list, _, transcriptions_list = \
            rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
                gt_file, evaluation_param['CRLF'], True, True, False)
        for n in range(len(points_list)):
            points = points_list[n]
            transcription = transcriptions_list[n]
            dont_care = transcription == "###"
            gt_rect = Rectangle(*points)
            gt_rects.append(gt_rect)
            gt_pol_points.append(points)
            if dont_care:
                gt_dont_care_rects_num.append(len(gt_rects) - 1)

        evaluation_log += "GT rectangles: " + str(len(gt_rects)) \
                          + (" (" + str(len(gt_dont_care_rects_num))
                             + " don't care)\n" if
                             len(gt_dont_care_rects_num) > 0 else "\n")

        if resFile in subm:
            det_file = rrc_evaluation_funcs.decode_utf8(subm[resFile])
            points_list, _, _ = \
                rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
                    det_file, evaluation_param['CRLF'], True, False, False)
            for n in range(len(points_list)):
                points = points_list[n]
                det_rect = Rectangle(*points)
                det_rects.append(det_rect)
                det_pol_points.append(points)
                if len(gt_dont_care_rects_num) > 0:
                    for dontCareRectNum in gt_dont_care_rects_num:
                        dont_care_rect = gt_rects[dontCareRectNum]
                        intersected_area = area(dont_care_rect, det_rect)
                        rd_dimensions = ((det_rect.xmax - det_rect.xmin + 1) *
                                         (det_rect.ymax - det_rect.ymin + 1))
                        if rd_dimensions == 0:
                            precision = 0
                        else:
                            precision = intersected_area / rd_dimensions
                        if (precision >
                                evaluation_param['AREA_PRECISION_CONSTRAINT']):
                            det_dont_care_rects_num.append(len(det_rects) - 1)
                            break

            evaluation_log += "DET rectangles: " + str(len(det_rects)) + \
                              (" (" + str(len(det_dont_care_rects_num)) +
                               " don't care)\n" if
                               len(det_dont_care_rects_num) > 0 else "\n")

            if len(gt_rects) == 0:
                recall = 1
                precision = 0 if len(det_rects) > 0 else 1

            if len(det_rects) > 0:
                # Calculate recall and precision matrixs
                output_shape = [len(gt_rects), len(det_rects)]
                recall_mat = np.empty(output_shape)
                precision_mat = np.empty(output_shape)
                gt_rect_mat = np.zeros(len(gt_rects), np.int8)
                det_rect_mat = np.zeros(len(det_rects), np.int8)
                for gt_num in range(len(gt_rects)):
                    for det_num in range(len(det_rects)):
                        r_g = gt_rects[gt_num]
                        r_d = det_rects[det_num]
                        intersected_area = area(r_g, r_d)
                        rg_dimensions = ((r_g.xmax - r_g.xmin + 1) *
                                         (r_g.ymax - r_g.ymin + 1))
                        rd_dimensions = ((r_d.xmax - r_d.xmin + 1) *
                                         (r_d.ymax - r_d.ymin + 1))
                        recall_mat[gt_num, det_num] = 0 if rg_dimensions == 0 \
                            else intersected_area / rg_dimensions
                        precision_mat[gt_num, det_num] = 0 \
                            if rd_dimensions == 0 \
                            else intersected_area / rd_dimensions

                # Find one-to-one matches
                evaluation_log += "Find one-to-one matches\n"
                for gt_num in range(len(gt_rects)):
                    for det_num in range(len(det_rects)):
                        if gt_rect_mat[gt_num] == 0 and \
                                det_rect_mat[det_num] == 0 and \
                                gt_num not in gt_dont_care_rects_num and \
                                det_num not in det_dont_care_rects_num:
                            match = one_to_one_match(gt_num, det_num)
                            if match is True:
                                # in deteval we have to make other validation
                                # before mark as one-to-one
                                if is_single_overlap(gt_num, det_num) is True:
                                    r_g = gt_rects[gt_num]
                                    r_d = det_rects[det_num]
                                    normDist = center_distance(r_g, r_d)
                                    normDist /= diag(r_g) + diag(r_d)
                                    normDist *= 2.0
                                    if normDist < evaluation_param['EV_PARAM_IND_CENTER_DIFF_THR']:
                                        gt_rect_mat[gt_num] = 1
                                        det_rect_mat[det_num] = 1
                                        recall_accum += evaluation_param['MTYPE_OO_O']
                                        precision_accum += evaluation_param['MTYPE_OO_O']
                                        pairs.append(
                                            {'gt': gt_num, 'det': det_num, 'type': 'OO'})
                                        evaluation_log += "Match GT #" + \
                                            str(gt_num) + " with Det #" + str(det_num) + "\n"
                                    else:
                                        evaluation_log += "Match Discarded GT #" + \
                                            str(gt_num) + " with Det #" + str(det_num) + " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluation_log += "Match Discarded GT #" + \
                                        str(gt_num) + " with Det #" + str(det_num) + " not single overlap\n"
                # Find one-to-many matches
                evaluation_log += "Find one-to-many matches\n"
                for gt_num in range(len(gt_rects)):
                    if gt_num not in gt_dont_care_rects_num:
                        match, matchesDet = one_to_many_match(gt_num)
                        if match is True:
                            evaluation_log += "num_overlaps_gt=" + \
                                str(num_overlaps_gt(gt_num))
                            # in deteval we have to make other validation
                            # before mark as one-to-one
                            if num_overlaps_gt(gt_num) >= 2:
                                gt_rect_mat[gt_num] = 1
                                recall_accum += (evaluation_param['MTYPE_OO_O'] if len(
                                    matchesDet) == 1 else evaluation_param['MTYPE_OM_O'])
                                precision_accum += (evaluation_param['MTYPE_OO_O'] if len(
                                    matchesDet) == 1 else evaluation_param['MTYPE_OM_O'] * len(matchesDet))
                                pairs.append({'gt': gt_num, 'det': matchesDet, 'type': 'OO' if len(
                                    matchesDet) == 1 else 'OM'})
                                for det_num in matchesDet:
                                    det_rect_mat[det_num] = 1
                                evaluation_log += "Match GT #" + \
                                    str(gt_num) + " with Det #" + str(matchesDet) + "\n"
                            else:
                                evaluation_log += "Match Discarded GT #" + \
                                    str(gt_num) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # Find many-to-one matches
                evaluation_log += "Find many-to-one matches\n"
                for det_num in range(len(det_rects)):
                    if det_num not in det_dont_care_rects_num:
                        match, matchesGt = many_to_one_match(det_num)
                        if match is True:
                            # in deteval we have to make other validation
                            # before mark as one-to-one
                            if num_overlaps_det(det_num) >= 2:
                                det_rect_mat[det_num] = 1
                                recall_accum += (evaluation_param['MTYPE_OO_O'] if len(
                                    matchesGt) == 1 else evaluation_param['MTYPE_OM_M'] * len(matchesGt))
                                precision_accum += (evaluation_param['MTYPE_OO_O'] if len(
                                    matchesGt) == 1 else evaluation_param['MTYPE_OM_M'])
                                pairs.append(
                                    {'gt': matchesGt, 'det': det_num, 'type': 'OO' if len(matchesGt) == 1 else 'MO'})
                                for gt_num in matchesGt:
                                    gt_rect_mat[gt_num] = 1
                                evaluation_log += "Match GT #" + \
                                    str(matchesGt) + " with Det #" + str(det_num) + "\n"
                            else:
                                evaluation_log += "Match Discarded GT #" + \
                                    str(matchesGt) + " with Det #" + str(det_num) + " not single overlap\n"

                num_gt_care = (len(gt_rects) - len(gt_dont_care_rects_num))
                if num_gt_care == 0:
                    recall = float(1)
                    precision = float(0) if len(det_rects) > 0 else float(1)
                else:
                    recall = float(recall_accum) / num_gt_care
                    precision = float(0) if (
                        len(det_rects) - len(det_dont_care_rects_num)) == 0 \
                        else float(precision_accum) / (
                            len(det_rects) - len(det_dont_care_rects_num))
                hmean = 0 if (precision + recall) == 0 else 2.0 * \
                    precision * recall / (precision + recall)

        method_recall_sum += recall_accum
        method_precision_sum += precision_accum
        num_gt += len(gt_rects) - len(gt_dont_care_rects_num)
        num_det += len(det_rects) - len(det_dont_care_rects_num)

        per_sample_metrics[resFile] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recallMat': [] if len(det_rects) > 100 else recall_mat.tolist(),
            'precision_mat': [] if len(det_rects) > 100 else precision_mat.tolist(),
            'gt_pol_points': gt_pol_points,
            'det_pol_points': det_pol_points,
            'gtDontCare': gt_dont_care_rects_num,
            'detDontCare': det_dont_care_rects_num,
            'evaluation_params': evaluation_param,
            'evaluation_log': evaluation_log}

    method_recall = 0 if num_gt == 0 else method_recall_sum / num_gt
    method_precision = 0 if num_det == 0 else method_precision_sum / num_det
    method_hmean = 0 if method_recall + method_precision == 0 else 2 * \
        method_recall * method_precision / (method_recall + method_precision)

    method_metrics = {
        'precision': method_precision,
        'recall': method_recall,
        'hmean': method_hmean}

    res_dict = {
        'calculated': True,
        'Message': '',
        'method': method_metrics,
        'per_sample': per_sample_metrics}

    return res_dict


if __name__ == '__main__':

    rrc_evaluation_funcs.main_evaluation(
        None,
        default_evaluation_params,
        validate_data,
        evaluate_method)
