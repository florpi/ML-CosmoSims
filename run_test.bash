# Parameters for Dataset
snapshot_nr=45
voxle_nr=512
simpath='/cosma5/data/dp004/dc-beck3/Dark2Light/data/full_physics/'

# Parameters for ML-algorithm
vel=0
lossweight=160
normalize=0
modelidx=2
target_class=0
name="${modelidx}_${lossweight}_v${vel}_r0"
save_name='test_0'

python3 main.py --simulation_path $simpath --snapshot_nr $snapshot_nr --voxle_nr $voxle_nr --lr 0.001 --loss_weight $lossweight --model_idx $modelidx --epochs 22 --label_type 'count' --target_class ${target_class} --load_model 0 --save_name $save_name --normalize $normalize  --record_results 0 --vel $vel > result_full_${name}.txt
