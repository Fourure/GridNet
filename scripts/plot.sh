#!/bin/bash

num=$(($#-1))
all=$*
folder=${@: -1}
tmp="/tmp/plot.gn"
tmp_train="/tmp/train.log"
tmp_valid="/tmp/valid.log"

#Create tmp log files
rm -f $tmp_train $tmp_valid

function get_logs {
	if [ -e ${1}/log.txt ]
	then
		model=$(cat ${1}/log.txt | grep "^model" | cut -d " " -f 2)
		if [ -e $model ]
		then
			get_logs $(dirname $model)
		fi
		echo "Get logs from ${1}"
		cat ${1}/train.log >> $tmp_train
		cat ${1}/valid.log >> $tmp_valid
	else
		echo "Error no log file in ${1}"
	fi
}

get_logs $folder


type="lines"

#echo "set terminal png size 400,250" >> $tmp
#echo "set ouput 'output.png'" >> $tmp
echo "set multiplot layout $num, 1" > $tmp
echo "set tmargin 2" >> $tmp
echo "set grid xtics ytics" >> $tmp
echo "set key bottom right" >> $tmp

echo "stats '${tmp_valid}' using 4 nooutput name 'I_'" >> $tmp

echo "stats '${tmp_valid}' using 1 every ::I_index_max::I_index_max nooutput" >> $tmp
echo "X_max = STATS_max" >>$tmp

echo "stats '${tmp_valid}' using 2 every ::I_index_max::I_index_max nooutput" >> $tmp
echo "P_max = STATS_max" >>$tmp

echo "stats '${tmp_valid}' using 3 every ::I_index_max::I_index_max nooutput" >> $tmp
echo "C_max = STATS_max" >>$tmp

for var in "$@"
do
    #echo "$var"
    if [ "$var" = "pixels" ]
    then
        echo 'set title "Pixels accuracy"' >> $tmp
        #echo 'unset key' >> $tmp
	echo "set label 2 sprintf(\"%.2f\", P_max) center at first X_max,P_max point pt 7 ps 1 offset 0,-1.5" >> $tmp
        echo "plot '${tmp_train}' using 1:2 title 'Train' with $type, \\" >> $tmp 
        echo "     '${tmp_valid}' using 1:2 title 'Validation' with $type" >> $tmp
	echo "unset label" >> $tmp
    elif [ "$var" = "class" ]
    then
        echo 'set title "Class accuracy"' >> $tmp
        #echo 'unset key' >> $tmp
	echo "set label 2 sprintf(\"%.2f\", C_max) center at first X_max,C_max point pt 7 ps 1 offset 0,-1.5" >> $tmp
        echo "plot '${tmp_train}' using 1:3 title 'Train' with $type, \\" >> $tmp 
        echo "     '${tmp_valid}' using 1:3 title 'Validation' with $type" >> $tmp
	echo "unset label" >> $tmp
    elif [ "$var" = "iou" ]
    then
        echo 'set title "IoU accuracy"' >> $tmp
        #echo 'unset key' >> $tmp
	echo "set label 2 sprintf(\"%.2f\", I_max) center at first X_max,I_max point pt 7 ps 1 offset 0,-1.5" >> $tmp
        echo "plot '${tmp_train}' using 1:4 title 'Train' with $type, \\" >> $tmp 
        echo "     '${tmp_valid}' using 1:4 title 'Validation' with $type" >> $tmp
	echo "unset label" >> $tmp
    fi
done

echo "unset multiplot" >> $tmp

gnuplot -persist $tmp
