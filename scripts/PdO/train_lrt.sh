python src/tasks/train.py \
    trainer.min_epochs=10000 \
    experiment=bnn_lrt \
    seed=5 \
    trainer.deterministic=False \
    task_name=train_lrt \
    datamodule=PdO \
    #ckpt_path=/home/riccardo/bin/repos/aenet-bnn/src/results/PdO/LRT/training/checkpoints/last.ckpt
