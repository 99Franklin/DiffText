该项目主要基于diffusers==0.15.0.dev0框架，请依照requirements.txt进行环境的搭建

准备好文本图像训练集后，通过./my_inpainting/new_paradigm_train.sh脚本进行生成模型的训练

完成模型训练后，通过./my_inpainting/my_build_synth_data_baseline.py脚本，制作合成数据集
