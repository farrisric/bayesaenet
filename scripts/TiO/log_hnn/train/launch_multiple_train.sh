#!/bin/bash


#python ~/bin/bayesaenet/scripts/TiO/create_seeds.py
#mv seeds.txt seeds_20.txt
#seeds=$(cat seeds_20.txt)
#
#for s in ${seeds[@]}
#do
#	qsub 20_train.sh ${s}
#done
#
#python ~/bin/bayesaenet/scripts/TiO/create_seeds.py
#mv seeds.txt seeds_100.txt
seeds=$(cat seeds_100.txt)

for s in ${seeds[@]}
do
	qsub log_80_train.sh ${s}
done
