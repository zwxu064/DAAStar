# CUDA_VISIBLE_DEVICES="0" python train.py --mode="f" --batch=256 --seed=1234 # > logs/train.txt 2>&1 &
CUDA_VISIBLE_DEVICES="0" python train.py --mode="dem" --batch=100 --seed=1234 # > logs/train_dem.txt 2>&1 &

# CUDA_VISIBLE_DEVICES="0" python eval.py --mode="f" --batch=256 --seed=1234 > logs/eval.txt 2>&1 &
