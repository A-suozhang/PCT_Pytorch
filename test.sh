CUDA_VISIBLE_DEVICES=1 python main.py --exp_name=test_scanobj --dataset scanobjnn --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/scanobjnn/models/model.t7 --test_batch_size 8 
