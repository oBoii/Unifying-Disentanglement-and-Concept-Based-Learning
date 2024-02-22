```shell
./load_datasets.sh

python train.py --dataset 3 --epochs 10 --batch_size 64 --num_workers 4 --lr 0.001 --limit_train_batches 0.1

python eval.py --dataset 3 --epochs 10 --batch_size 64 --num_workers 4 --lr 0.001 --limit_train_batches 0.1
```

```shell
cd /project_ghent/Unifying-Disentanglement-and-Concept-Based-Learning/;
git reset --hard;
git fetch; git pull;
python main.py --dataset 3 --epochs 10 --batch_size 64 --num_workers 4 --lr 0.001 --limit_train_batches 0.1
```