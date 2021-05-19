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
import os
import re
import zipfile
import json
import sys

sys.path.append('./')


def print_help():
    sys.stdout.write('Usage: python %s.py -g=<gtFile> -s=<submFile> '
                     '[-o=<outputFolder> -p=<jsonParams>]' % sys.argv[0])
    sys.exit(2)


def load_zip_file_keys(file, file_name_reg_exp=''):
    """
    Returns an array with the entries of the ZIP file that match with
    the regular expression.
    The key's are the names or the file or the capturing group definied
    in the file_name_reg_exp
    """
    try:
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except (RuntimeError, zipfile.BadZipFile):
        raise Exception('Error loading the ZIP archive.')

    pairs = []

    for name in archive.namelist():
        add_file = True
        key_name = name
        if file_name_reg_exp != "":
            m = re.match(file_name_reg_exp, name)
            if m is None:
                add_file = False
            else:
                if len(m.groups()) > 0:
                    key_name = m.group(1)

        if add_file:
            pairs.append(key_name)

    return pairs


def load_zip_file(file, file_name_reg_exp='', all_entries=False):
    """
    Returns an array with the contents (filtered by file_name_reg_exp) of
    a ZIP file.
    The key's are the names or the file or the capturing group definied in
    the file_name_reg_exp
    all_entries validates that all entries in the ZIP file pass the
    file_name_reg_exp
    """
    try:
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except (RuntimeError, zipfile.BadZipFile):
        raise Exception('Error loading the ZIP archive')

    pairs = []
    for name in archive.namelist():
        add_file = True
        key_name = name
        if file_name_reg_exp != "":
            m = re.match(file_name_reg_exp, name)
            if m is None:
                add_file = False
            else:
                if len(m.groups()) > 0:
                    key_name = m.group(1)

        if add_file:
            pairs.append([key_name, archive.read(name)])
        else:
            if all_entries:
                raise Exception('ZIP entry not valid: %s' % name)

    return dict(pairs)


def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        return raw.decode('utf-8-sig', errors='replace')
    except BaseException:
        return None


def validate_lines_in_file(file_name, file_contents, cr_lf=True, lt_rb=True,
                           with_transcription=False, with_confidence=False,
                           im_width=0, im_height=0):
    """
    This function validates that all lines of the file calling the Line
    validation function for each line
    """
    utf8_file = decode_utf8(file_contents)
    if utf8_file is None:
        raise Exception("The file %s is not UTF-8" % file_name)

    lines = utf8_file.split("\r\n" if cr_lf else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line:
            try:
                validate_tl_line(
                    line,
                    lt_rb,
                    with_transcription,
                    with_confidence,
                    im_width,
                    im_height)
            except Exception as e:
                raise Exception(("Line in sample not valid. Sample: %s Line: "
                                 "%s Error: %s" % (file_name, line, str(e))).
                                encode('utf-8', 'replace'))


def validate_tl_line(line, lt_rb=True, with_transcription=True,
                     with_confidence=True, im_width=0, im_height=0):
    """
    Validate the format of the line. If the line is not valid an exception
    will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the
    imgage bounds.
    Posible values are:
    lt_rb=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    lt_rb=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    """
    get_tl_line_values(
        line,
        lt_rb,
        with_transcription,
        with_confidence,
        im_width,
        im_height)


def get_tl_line_values(line, lt_rb=True, with_transcription=False,
                       with_confidence=False, im_width=0, im_height=0):
    """
    Validate the format of the line. If the line is not valid an exception
    will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the
    imgage bounds.
    Posible values are:
    lt_rb=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    lt_rb=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = ""
    if lt_rb:
        num_points = 4
        if with_transcription and with_confidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+'
                r')\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,"
                    "confidence,transcription")
        elif with_confidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+'
                r')\s*,\s*([0-1].?[0-9]*)\s*$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,"
                    "confidence")
        elif with_transcription:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+'
                r')\s*,(.*)$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,"
                    "transcription")
        else:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+'
                r')\s*,?\s*$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax")

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if xmax < xmin:
            raise Exception(
                "Xmax value (%s) not valid (Xmax < Xmin)." %
                xmax)
        if ymax < ymin:
            raise Exception(
                "Ymax value (%s)  not valid (Ymax < Ymin)." % ymax)

        points = [float(m.group(i)) for i in range(1, (num_points + 1))]

        if im_width > 0 and im_height > 0:
            validate_point_inside_bounds(xmin, ymin, im_width, im_height)
            validate_point_inside_bounds(xmax, ymax, im_width, im_height)

    else:

        num_points = 8

        if with_transcription and with_confidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?'
                r'[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)'
                r'\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,"
                    "confidence,transcription")
        elif with_confidence:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?'
                r'[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)'
                r'\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,"
                    "confidence")
        elif with_transcription:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?'
                r'[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)'
                r'\s*,\s*(-?[0-9]+)\s*,(.*)$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,"
                    "transcription")
        else:
            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?'
                r'[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)'
                r'\s*,\s*(-?[0-9]+)\s*$',
                line)
            if m is None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")

        points = [float(m.group(i)) for i in range(1, (num_points + 1))]

        validate_clockwise_points(points)

        if im_width > 0 and im_height > 0:
            validate_point_inside_bounds(points[0], points[1], im_width,
                                         im_height)
            validate_point_inside_bounds(points[2], points[3], im_width,
                                         im_height)
            validate_point_inside_bounds(points[4], points[5], im_width,
                                         im_height)
            validate_point_inside_bounds(points[6], points[7], im_width,
                                         im_height)

    if with_confidence:
        try:
            confidence = float(m.group(num_points + 1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    if with_transcription:
        pos_transcription = num_points + (2 if with_confidence else 1)
        transcription = m.group(pos_transcription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
        # Transcription with double quotes, we extract the value and replace
        # escaped characters
        if m2 is not None:
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"",
                                                                      "\"")

    return points, confidence, transcription


def get_tl_dict_values(
        detection,
        with_transcription=False,
        with_confidence=False,
        im_width=0,
        im_height=0,
        valid_num_points=[],
        validate_cw=True):
    """
    Validate the format of the dictionary. If the dictionary is not valid an
    exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the
    imgage bounds.
    Posible values:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]]}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"illegibility":false}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"dontCare":false}
    Returns values from the dictionary. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = ""
    points = []

    if not isinstance(detection, dict):
        raise Exception("Incorrect format. Object has to be a dictionary")

    if 'points' not in detection:
        raise Exception("Incorrect format. Object has no points key)")

    if not isinstance(detection['points'], list):
        raise Exception(
            "Incorrect format. Object points key have to be an array)")

    num_points = len(detection['points'])

    if num_points < 3:
        raise Exception(
            "Incorrect format. Incorrect number of points. At least 3 points "
            "are necessary. Found: " +
            str(num_points))

    if len(valid_num_points) > 0 and num_points not in valid_num_points:
        raise Exception(
            "Incorrect format. Incorrect number of points. Only allowed 4,8 "
            "or 12 points)")

    for i in range(num_points):
        if not isinstance(detection['points'][i], list):
            raise Exception("Incorrect format. Point #" + str(
                i + 1) + " has to be an array)")

        if len(detection['points'][i]) != 2:
            raise Exception("Incorrect format. Point #" + str(
                i + 1) + " has to be an array with 2 objects(x,y) )")

        if isinstance(detection['points'][i][0],
                      (int, float)) is False or isinstance(
                detection['points'][i][1], (int, float)) is False:
            raise Exception("Incorrect format. Point #" + str(
                i + 1) + " childs have to be Integers)")

        if im_width > 0 and im_height > 0:
            validate_point_inside_bounds(detection['points'][i][0],
                                         detection['points'][i][1], im_width,
                                         im_height)

        points.append(float(detection['points'][i][0]))
        points.append(float(detection['points'][i][1]))

    if validate_cw:
        validate_clockwise_points(points)

    if with_confidence:
        if 'confidence' not in detection:
            raise Exception("Incorrect format. No confidence key)")

        if not isinstance(detection['confidence'], (int, float)):
            raise Exception(
                "Incorrect format. Confidence key has to be a float)")

        if detection['confidence'] < 0 or detection['confidence'] > 1:
            raise Exception(
                "Incorrect format. Confidence key has to be a float "
                "between 0.0 and 1.0")

        confidence = detection['confidence']

    if with_transcription:
        if 'transcription' not in detection:
            raise Exception("Incorrect format. No transcription key)")

        if not isinstance(detection['transcription'], str):
            raise Exception(
                "Incorrect format. Transcription has to be a string. Detected: "
                + type(detection['transcription']).__name__)

        transcription = detection['transcription']

        # Ensures that if illegibility atribute is present and is True the
        # transcription is set to ### (don't care)
        if 'illegibility' in detection:
            if detection['illegibility']:
                transcription = "###"

        # Ensures that if dontCare atribute is present and is True the
        # transcription is set to ### (don't care)
        if 'dontCare' in detection:
            if detection['dontCare']:
                transcription = "###"

    return points, confidence, transcription


def validate_point_inside_bounds(x, y, im_width, im_height):
    if x < 0 or x > im_width:
        raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" % (
            xmin, im_width, im_height))
    if y < 0 or y > im_height:
        raise Exception(
            "Y value (%s)  not valid. Image dimensions: (%s,%s)" %
            (ymin, im_width, im_height))


def validate_clockwise_points(points):
    """
    Validates that the points are in clockwise order.
    """
    edge = []
    for i in range(len(points) // 2):
        edge.append(
            (int(points[(i + 1) * 2 % len(points)]) - int(points[i * 2])) * (
                int(points[((i + 1) * 2 + 1) % len(points)]) + int(
                    points[i * 2 + 1])))
    if sum(edge) > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding points have "
            "to be given in clockwise order. Regarding the correct "
            "interpretation of 'clockwise' remember that the image coordinate "
            "system used is the standard one, with the image origin at the "
            "upper left, the X axis extending to the right and Y axis "
            "extending downwards.")


def get_tl_line_values_from_file_contents(
        content,
        cr_lf=True,
        lt_rb=True,
        with_transcription=False,
        with_confidence=False,
        im_width=0,
        im_height=0,
        sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists.
    Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    points_list = []
    transcriptions_list = []
    confidences_list = []

    lines = content.split("\r\n" if cr_lf else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            points, confidence, transcription = get_tl_line_values(
                line, lt_rb, with_transcription, with_confidence, im_width,
                im_height)
            points_list.append(points)
            transcriptions_list.append(transcription)
            confidences_list.append(confidence)

    if with_confidence and len(confidences_list) > 0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidences_list))
        confidences_list = [confidences_list[i] for i in sorted_ind]
        points_list = [points_list[i] for i in sorted_ind]
        transcriptions_list = [transcriptions_list[i] for i in sorted_ind]

    return points_list, confidences_list, transcriptions_list


def get_tl_dict_values_from_array(array, with_transcription=False,
                                  with_confidence=False, im_width=0,
                                  im_height=0, sort_by_confidences=True,
                                  valid_num_points=[], validate_cw=True):
    """
    Returns all points, confindences and transcriptions of a file in lists.
    Valid dict formats:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4}
    """
    points_list = []
    transcriptions_list = []
    confidences_list = []

    for n in range(len(array)):
        object_dict = array[n]
        points, confidence, transcription = \
            get_tl_dict_values(object_dict, with_transcription,
                               with_confidence, im_width, im_height,
                               valid_num_points, validate_cw)
        points_list.append(points)
        transcriptions_list.append(transcription)
        confidences_list.append(confidence)

    if with_confidence and len(confidences_list) > 0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidences_list))
        confidences_list = [confidences_list[i] for i in sorted_ind]
        points_list = [points_list[i] for i in sorted_ind]
        transcriptions_list = [transcriptions_list[i] for i in sorted_ind]

    return points_list, confidences_list, transcriptions_list


def main_evaluation(p, default_evaluation_params_fn, validate_data_fn,
                    evaluate_method_fn, show_result=True, per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a
    ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is
    passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary
     with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the
    submission
    evaluate_method_fn: points to a function that evaluated the submission and
    return a Dictionary with the results
    """

    if p is None:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        if len(sys.argv) < 3:
            print_help()

    eval_params = default_evaluation_params_fn()
    if 'p' in p.keys():
        eval_params.update(
            p['p'] if isinstance(p['p'], dict) else json.loads(p['p']))

    res_dict = {'calculated': True, 'Message': '', 'method': '{}',
                'per_sample': '{}'}
    try:
        validate_data_fn(p['g'], p['s'], eval_params)
        eval_data = evaluate_method_fn(p['g'], p['s'], eval_params)
        res_dict.update(eval_data)
        if not res_dict['calculated']:
            if show_result:
                sys.stderr.write('Error!\n' + res_dict['Message'] + '\n\n')
            return res_dict

        if 'o' in p:
            if not os.path.exists(p['o']):
                os.makedirs(p['o'])

            results_output_name = p['o'] + '/results.zip'
            out_zip = zipfile.ZipFile(results_output_name, mode='w',
                                      allowZip64=True)

            del res_dict['per_sample']
            if 'output_items' in res_dict.keys():
                del res_dict['output_items']

            out_zip.writestr('method.json', json.dumps(res_dict))

            if per_sample:
                for k, v in eval_data['per_sample'].items():
                    out_zip.writestr(k + '.json', json.dumps(v))

            if 'output_items' in eval_data.keys():
                for k, v in eval_data['output_items'].items():
                    out_zip.writestr(k, v)
            out_zip.close()

        if show_result:
            sys.stdout.write("Calculated!")
            sys.stdout.write(json.dumps(res_dict['method']))
        return res_dict

    except Exception as e:
        res_dict['Message'] = str(e)
        res_dict['calculated'] = False
        raise Exception(res_dict['Message'])


def main_validation(default_evaluation_params_fn, validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary
     with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the
    submission
    """
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        eval_params = default_evaluation_params_fn()
        if 'p' in p.keys():
            eval_params.update(
                p['p'] if isinstance(p['p'], dict) else json.loads(p['p']))

        validate_data_fn(p['g'], p['s'], eval_params)
        print('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(101)
