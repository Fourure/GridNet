#!/bin/bash

folder="$1"

tmp="/tmp/plot.gn"
tmp_train="/tmp/train.log"
tmp_valid="/tmp/valid.log"

rm -f $tmp_train $tmp_valid $tmp

function parse_log {
	awk -v train="$tmp_train" -v valid="$tmp_valid" '$5 == "training" && $6 == "data:"  { training = 1 }
		$5 == "validation" && $6 == "data:" { training = 0 }
		$2 == "epoch" { sub("#","",$3); epoch = $3 }
		$1 == "Time" && training==0 {
			sub("ms","*0+",$3)
			sub("s","*1+",$3)
			sub("m","*60+",$3)
	       		sub("h","*3600+",$3)	
			$3=$3 "0"
			system("echo " epoch " $(("$3")) >> " valid)
		}
		$1 == "Time" && training==1 {
			sub("ms","*0+",$3)
			sub("s","*1+",$3)
			sub("m","*60+",$3)
	       		sub("h","*3600+",$3)	
			$3=$3 "0"
			system("echo " epoch " $(("$3")) >> " train)
		}' ${1}
}

function get_logs {
	if [ -e ${1}/log.txt ]
	then
		model=$(cat ${1}/log.txt | grep "^model" | cut -d " " -f 2)
		if [ -e $model ]
		then
			get_logs $(dirname $model)
		fi

		echo "Get logs from ${1}"
		parse_log "${1}/log.txt"
	else
		echo "Error no log file in ${1}"
	fi
}

get_logs $folder

type="lines"

#echo "set multiplot layout $num, 1" > $tmp
echo "set tmargin 2" >> $tmp
echo "set grid xtics ytics" >> $tmp

echo "set ydata time" >> $tmp
echo "set timefmt \"%s\"" >> $tmp
#echo "set format y \"%H/%M\"" >> $tmp 

echo 'set title "Time"' >> $tmp
echo "plot '${tmp_train}' using 1:2 title 'Train' with $type, \\" >> $tmp 
echo "     '${tmp_valid}' using 1:2 title 'Validation' with $type" >> $tmp

gnuplot -persist $tmp
