all:
	python3 main_single_gpu.py ./datasets/hmdb51_frames/ -m rhythm --dataset hmdb51 -a rgb_resnet152 -s=1 --new_length=1 --epochs 350 --lr 0.001 --lr_steps 75 150 -pf 25> experiment_rhythm_s1.txt
