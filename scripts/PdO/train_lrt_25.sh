python src/tasks/train.py \
 trainer.min_epochs=10000 \
 experiment=bnn_lrt \
 seed=5 \
 trainer.deterministic=False \
 task_name=train_lrt \
 datamodule=PdO \
 datamodule.test_split=0.65 \
 datamodule.valid_split=0.1 \
 ckpt_path=/home/riccardo/bin/repos/aenet-bnn/src/logs/train_lrt/runs/2024-05-03_14-45-16/checkpoints/epoch_305-step_11628.ckpt