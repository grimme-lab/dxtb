#!/usr/bin/env bash

trap "echo Exited!; exit;" SIGINT SIGTERM

pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

if [ -z "$1" ]; then
	FUNC="gfn1"
else
	FUNC="$1"
fi

for i in ACONF SCONF PCONF21 Amino20x4 MCONF; do
	echo $i
	pushd $i

	# run calculations via tblite
	../scripts/run.sh $FUNC

	# create folder structure, i.e. folder with name "$FUNC" for every molecule containing file "energy" in TM format 
	python3 ../scripts/build-energy.py $FUNC

	# get statistics
	../scripts/.res $FUNC

	md=$(tail -1 benchmark/.${FUNC}.out | gawk '{print $4}')
	mae=$(tail -1 benchmark/.${FUNC}.out | gawk '{print $6}')
	rmsd=$(tail -1 benchmark/.${FUNC}.out | gawk '{print $8}')
	echo "MD = $md , MAE = $mae , RMSD = $rmsd"

	popd
done

for i in ACONF SCONF PCONF21 Amino20x4 MCONF; do
	echo $i
	pushd $i

	python3 ../scripts/grad.py

	popd
done