echo "Train..."

echo "T+1..."

python3 train_t_1.py configs/V3_4layers_ATT4_4heads_250_250_waymo_2.yaml

echo "T+5..."

python3 train_t_5.py configs/V3_4layers_ATT4_4heads_250_250_waymo_2.yaml

echo "Finished"
