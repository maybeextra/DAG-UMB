echo "$(pwd)"

export CUDA_VISIBLE_DEVICES=0
python examples/cluster_contrast_train_usl.py --kind FSB
python examples/cluster_contrast_train_usl.py --kind FS
python examples/cluster_contrast_train_usl.py --kind FB
python examples/cluster_contrast_train_usl.py --kind SB
