#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export num_gpu=1
export use_multi_gpu=False
export use_obj_refine=False
export task='predcls'

export save_result=False

# 定义权重文件名列表
weight_files=(      
            # "model_0015000.pth"
            # "model_0020000.pth"
            # "model_0025000.pth" 
            "model_0030000.pth"
            "model_0035000.pth"
            "model_0040000.pth" 
            "model_0045000.pth" 
            "model_0050000.pth"
            "model_0055000.pth"
            "model_0060000.pth"
             )

export model_config="tmpn_vg" # 
export output_dir="/home/yj/zgw/tmpn/ckpt/${task}-tmpn_vg"
mkdir ${output_dir}
export path_faster_rcnn='/home/yj/zgw/tmpn/Datasets/VG/vg_faster_det.pth' # Put faster r-cnn path
cp /home/yj/zgw/tmpn/hetsgg/modeling/roi_heads/relation_head/roi_relation_predictors.py ${output_dir}/

python tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
    SOLVER.IMS_PER_BATCH 2 \
    TEST.IMS_PER_BATCH 4     \
    OUTPUT_DIR ${output_dir} \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
    MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}

for weight_file in "${weight_files[@]}"
do
    echo "Running with weight file: ${weight_file}"
    
    python tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 8 \
            TEST.SAVE_RESULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/${weight_file}"
done