# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
// Post processing for adveast
//
// Created by yang xuehang on 20200716.
// Copyright @ 2019年 yang xuehang. All rights reserved.

#include <math.h>
#include <map>
#include <vector>
//#include <queue>
#include <algorithm>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

namespace py = pybind11;

namespace advEast{	
    // quads, conf = pse_cpp(score_map, label_map, label_num, head_mask, tail_mask, 
    //                        is_vert, geo_quad_map, xyscale=4)
	std::vector<std::vector<float> > pse(
        py::array_t<float, py::array::c_style> score_map,
		py::array_t<int32_t, py::array::c_style> label_map,
        int label_num,
        py::array_t<int32_t, py::array::c_style> head_mask,
        py::array_t<int32_t, py::array::c_style> tail_mask,
        py::array_t<int32_t, py::array::c_style> is_vert,
        py::array_t<float, py::array::c_style> geo_quad_map,
        int xyscale=4)
	{
        auto pbuf_score_map = score_map.request();
        auto pbuf_label_map = label_map.request();
        auto pbuf_head_mask = head_mask.request();
        auto pbuf_tail_mask = tail_mask.request();
        auto pbuf_is_vert = is_vert.request();
        auto pbuf_geo_quad_map = geo_quad_map.request();
		
		if (pbuf_label_map.ndim !=2 || pbuf_label_map.shape[0]==0 || pbuf_label_map.shape[1]==0)
			throw std::runtime_error("label map must have a shape of (h>0, w>0)");
		int h = pbuf_label_map.shape[0];
		int w = pbuf_label_map.shape[1];
		if (pbuf_score_map.ndim != 2 || pbuf_score_map.shape[0]!=h \
                || pbuf_score_map.shape[1]!=w)
			throw std::runtime_error("score_map must have a shape of (h,w)");
		if (pbuf_head_mask.ndim != 2 || pbuf_head_mask.shape[0]!=h \
                || pbuf_head_mask.shape[1]!=w)
			throw std::runtime_error("head_mask must have a shape of (h,w)");
		if (pbuf_tail_mask.ndim != 2 || pbuf_tail_mask.shape[0]!=h \
                || pbuf_tail_mask.shape[1]!=w)
			throw std::runtime_error("tail_mask must have a shape of (h,w)");
		if (pbuf_geo_quad_map.ndim != 3 || pbuf_geo_quad_map.shape[0]!=h \
			|| pbuf_geo_quad_map.shape[1]!=w || pbuf_geo_quad_map.shape[2]!=8)
			throw std::runtime_error("geo_quad_map must have a shape of (h,w,8)");
	    if (pbuf_is_vert.ndim != 1 || pbuf_is_vert.shape[0]!=label_num)
            throw std::runtime_error("is_vert must have a shape of (label_num,)");

		// get ptr of label_map, score_map, geo_rbox_map, geo_quad_map
		auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
		auto ptr_score_map = static_cast<float *>(pbuf_score_map.ptr);
        auto ptr_head_mask = static_cast<int32_t *>(pbuf_head_mask.ptr);
        auto ptr_tail_mask = static_cast<int32_t *>(pbuf_tail_mask.ptr);
		auto ptr_geo_quad_map = static_cast<float *>(pbuf_geo_quad_map.ptr);
        auto ptr_is_vert = static_cast<int32_t *>(pbuf_is_vert.ptr);
		
		// get 4 corner points of each kernel.	
        std::vector<std::vector<float> > quad_vector;
		for(int i=0; i<label_num; i++)
		{
            std::vector<float> tmp = {0., 0., 0., 0.,  0., 0., 0., 0.,  0.};
			quad_vector.push_back(tmp);				
		}
        //count of head, tail.
		float head_count[label_num][2] = {0};
        float tail_count[label_num][2] = {0}; 
		
		// Traverse over image, analysis on kernel points.
		for(int i=0; i<h; i++)
		{
			auto p_label_map = ptr_label_map + i*w;
			auto p_score_map = ptr_score_map + i*w;
            auto p_head_mask = ptr_head_mask + i*w;
            auto p_tail_mask = ptr_tail_mask + i*w;
            auto p_geo_quad_map = ptr_geo_quad_map + i*w*8;

			for(int j=0, m=0; j<w && m<w*8; j++,m+=8)
			{
				int32_t label = p_label_map[j];
                float score = p_score_map[j];
				int32_t head = p_head_mask[j];
				int32_t tail = p_tail_mask[j];
                int32_t isVert = ptr_is_vert[label];

				if (label>0)
				{
					if (head>0)
					{
                        if (isVert) {
                            quad_vector[label][0] += p_geo_quad_map[0];
						    quad_vector[label][1] += p_geo_quad_map[1];
						    quad_vector[label][2] += p_geo_quad_map[2];
						    quad_vector[label][3] += p_geo_quad_map[3];
                        } else {
                            quad_vector[label][0] += p_geo_quad_map[0];
						    quad_vector[label][1] += p_geo_quad_map[1];
						    quad_vector[label][6] += p_geo_quad_map[6];
						    quad_vector[label][7] += p_geo_quad_map[7];
                        }
						head_count[label][0] += 1;
                        head_count[label][1] += score;
					}
					if (tail>0)
					{
                        if (isVert) {
						    quad_vector[label][4] += p_geo_quad_map[4];
    						quad_vector[label][5] += p_geo_quad_map[5];
	    					quad_vector[label][6] += p_geo_quad_map[6];
		    				quad_vector[label][7] += p_geo_quad_map[7];
                        } else {
						    quad_vector[label][2] += p_geo_quad_map[2];
    						quad_vector[label][3] += p_geo_quad_map[3];
	    					quad_vector[label][4] += p_geo_quad_map[4];
		    				quad_vector[label][5] += p_geo_quad_map[5];
                        }
						tail_count[label][0] += 1;
                        tail_count[label][1] += score;
					}
				}				
			}

			for(int i=0; i<label_num; i++)
			{				
                int32_t isVert = ptr_is_vert[i];
				if (head_count[i][0])
				{
                    head_count[i][1] /= head_count[i][0];
                    if (isVert) {
					    for(int j=0; j<4; j++)				
						    quad_vector[i][j] /= head_count[i][0];	
                    } else {
                        quad_vector[i][0] /= head_count[i][0];
                        quad_vector[i][1] /= head_count[i][1];
                        quad_vector[i][6] /= head_count[i][6];
                        quad_vector[i][7] /= head_count[i][7];
                    }
				}
				if (tail_count[i][0])
				{					
					tail_count[i][1] /= tail_count[i][0];
                    if (isVert) {
                        for(int j=4; j<8; j++)
						    quad_vector[i][j] /= tail_count[i][0];					
                    } else {
                        for(int j=2; j<6; j++)
						    quad_vector[i][j] /= tail_count[i][0];					
                    }
				}
                quad_vector[i][8] = std::min(head_count[i][1], tail_count[i][1]);
			}
		}
		return quad_vector;	
	}


	std::map<int, std::vector<float>> get_points(
		py::array_t<int32_t, py::array::c_style> label_map, 
		py::array_t<float, py::array::c_style> score_map, 
		int label_num)
	{
		auto pbuf_label_map = label_map.request();
		auto pbuf_score_map = score_map.request();
		auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
		auto ptr_score_map = static_cast<float *>(pbuf_score_map.ptr);
		int h = pbuf_label_map.shape[0];
		int w = pbuf_label_map.shape[1];
		
		std::map<int, std::vector<float>> point_dict;
		std::vector<std::vector<float>> point_vector;
		for(int i=0; i<label_num; i++)
		{
			std::vector<float> point;
			point.push_back(0);
			point.push_back(0);
			point_vector.push_back(point);
		}
		for(int i=0; i<h; i++)
		{
			auto p_label_map = ptr_label_map + i*w;
			auto p_score_map = ptr_score_map + i*w;
			for(int j=0; j<w; j++)
			{
				int32_t label = p_label_map[j];
				if (label==0) continue;
				float score = p_score_map[j];
				point_vector[label][0] += score;
				point_vector[label][1] += 1;
				point_vector[label].push_back(j);
				point_vector[label].push_back(i);
			}
		}
		for(int i=0; i<label_num; i++)
		{
			if (point_vector[i].size()>2)
			{
				point_vector[i][0] /= point_vector[i][1];
				point_dict[i] = point_vector[i];
			}
		}
		return point_dict;
	}
	
	std::vector<int> get_num(py::array_t<int32_t, py::array::c_style> label_map, int label_num)
	{
		auto pbuf_label_map = label_map.request();
		auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
		int h = pbuf_label_map.shape[0];
		int w = pbuf_label_map.shape[1];
		
		//int point_vector[label_num] = {0};
		std::vector<int> point_vector;
		for(int i=0; i<label_num; i++)
			point_vector.push_back(0);
		
		for(int i=0; i<h; i++)
		{
			auto p_label_map = ptr_label_map + i*w;
			for(int j=0; j<w; j++)
			{
				int32_t label = p_label_map[j];
				if (label)
					point_vector[label] += 1;
			}
		}
		return point_vector;
	}
}


PYBIND11_MODULE(pse, m){
	m.def("pse_cpp", &advEast::pse, "re-implementation post process in advEast(cpp)", \
		py::arg("score_map"), py::arg("label_map"), py::arg("label_num"),
        py::arg("head_mask"), py::arg("tail_mask"), py::arg("is_vert"), 
        py::arg("geo_quad_map"), py::arg("xyscale"));

	m.def("get_num", &advEast::get_num, "re-implementation post process in advEast(cpp)", \
		py::arg("label_map"), py::arg("label_num"));

	m.def("get_points", &advEast::get_points, "re-implementation post process in advEast(cpp)", \
		py::arg("label_map"), py::arg("score_map"), py::arg("label_num"));
}


