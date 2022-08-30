#!/usr/bin/env bash

trap "echo Exited!; exit;" SIGINT SIGTERM

if [ -z "$1" ]; then
	METHOD="gfn1"
else
	METHOD="$1"
fi

if [ "$1" == "gfn1.fit" ]; then
	METHOD="$1"
	ARGS="--grad --param $HOME/Dokumente/xtbML/src/gfn1-xtb_tmp.toml"
else
	ARGS="--grad --method gfn1"
fi


MAIN=$PWD
#ARGS="--param $HOME/Dokumente/xtbML/src/gfn1-xtb_ORIGINAL.toml"
#


function run_tblite {
	tblite run --guess eeq $ARGS --json $METHOD.json coord > $METHOD.out

	if [ $? -ne 0 ]; then
		tblite run --guess sad $ARGS --json $METHOD.json coord > $METHOD.out
	fi
}

cd benchmark	

for record in $(ls -d */); do
	cd $record
	#echo "Running $record"
	
	if [[ "$subset" == "G21IP/" && "$METHOD" == "gfn2" ]]; then
		if [ "$record" = "b+/" ] || [ "$record" = "b/" ] || [ "$record" = "be+/" ] || [ "$record" = "be/" ] || [ "$record" = "c/" ] || [ "$record" = "c+/" ]; then
			tblite run --guess eeq --etemp 900 $ARGS --json $METHOD.json coord > $METHOD.out
		else
			run_tblite
		fi
	else
		run_tblite
	fi

	cd ..
done
	
