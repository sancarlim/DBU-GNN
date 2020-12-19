# !/bin/bash

#python main_pylightning.py --name 'Ablation GAT 5/5' --model 'gat' --history_frames 5 --future_frames 5

#python main_pylightning.py --name 'Ablation GAT 3/3' --model 'gat' --history_frames 3 --future_frames 3

#python main_pylightning.py --name 'Ablation GATED 5/5' --model 'gated' --history_frames 5 --future_frames 5 && echo "Sweep launched" & 
#P0=$!
#"$!" is the PID of the last program your shell ran in the background

#python main_pylightning.py --name 'Ablation GATED 3/3' --model 'gated' --history_frames 3 --future_frames 3 && echo "Sweep launched" & 
#P1=$!

python main_pylightning.py --name 'Ablation GCN 5/5' --model 'gcn' --history_frames 5 --future_frames 5 && echo "Sweep launched" & 
P2=$!

#python main_pylightning.py --name 'Ablation GCN 3/3' --model 'gcn' --history_frames 3 --future_frames 3 && echo "Sweep launched" & 
#P3=$!

wait $P2