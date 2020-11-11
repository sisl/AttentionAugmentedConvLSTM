echo "Train..."

echo "T+1..."

python train_t_1.py ../configs/PredNet_with_TAAConvLSTM_Kitti.yaml

echo "T+5..."

python train_t_5.py ../configs/PredNet_with_TAAConvLSTM_Kitti.yaml

echo "Finished"
