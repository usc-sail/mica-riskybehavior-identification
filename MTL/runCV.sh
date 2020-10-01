#!/bin/bash
set -euxo pipefail

echo "$0 $*"

PYTHON=$(which python)
echo "Using Python: $PYTHON"

script=$1
shift

device=$1
shift

fold_dir=$1
shift

out_dir="../results/$1/fold_results"
log_dir="../results/$1/logs/"
shift

run1 () {
	fold=$1
	shift

	if [ ! -f $out_dir/fold$fold.res.npz ]; then
		PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$device THEANO_FLAGS="device=cuda$device,floatX=float32" $PYTHON -u $script $fold_dir/$fold $out_dir/fold$fold.all.res --modelname $out_dir/best_model_$fold.hdf5 $* 1>$log_dir/fold$fold.log 2>&1
	else
		echo "skipping $fold_dir/$fold $out_dir/fold$fold.res"
	fi
}


if [ ! -d "$out_dir" ]; then
	mkdir -p "$out_dir" || true
else
	rm $out_dir/* || true
fi

if [ ! -d "$log_dir" ]; then
	mkdir -p $log_dir || true
fi

# Count how many folds are there
nfolds=$(find $fold_dir/* -maxdepth 0 -type d | wc -l)
nfolds=$(($nfolds-1))

for i in $(seq 0 $nfolds); do
	run1 $i $*
done
