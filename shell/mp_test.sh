#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export num_gpu=1
export use_multi_gpu=false
export task='predcls'

# 定义权重文件名列表
weight_files=(      
                    # "model_0035000.pth"
                    # "model_0015000.pth"
                    # "model_0020000.pth"
                    # "model_0025000.pth" 
                    # "model_0030000.pth"
                    # "model_0040000.pth"
                    # "model_0045000.pth" 
                    "model_0050000.pth"
                    "model_0055000.pth"
                    "model_0060000.pth" 
                    "model_0065000.pth" 
                    "model_0070000.pth" 
                    "model_0075000.pth" 
                    )

export save_result=False
export output_dir="/home/yj/zgw/het2/ckpt/predcls-het_mp_struct_prompt_proto_MoE_final8"  # 检查点目录

# 遍历权重文件列表
for weight_file in "${weight_files[@]}"
do
    echo "Running with weight file: ${weight_file}"
    
    python tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 8 \
            TEST.SAVE_RESULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/${weight_file}"
done