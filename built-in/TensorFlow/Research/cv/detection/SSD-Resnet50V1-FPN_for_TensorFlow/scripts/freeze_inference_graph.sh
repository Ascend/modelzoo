cur_path=`pwd`
cd ${cur_path}/../models/research/object_detection;
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${cur_path}/../configs/ssd320_full_4gpus.config\
    --trained_checkpoint_prefix /checkpoints/model.ckpt-25000 \
    --output_directory ${cur_path}/savedModel \
    --config_override " \
            model{ \
              ssd { \
                post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"
