#!/bin/bash

dataset=$1;
output=$2;

n=`wc -l $dataset | awk '{print $1}'`;

rm -rf ${output}/ResKNNLOO;

for i in `seq 1 $n`;
do
  echo "idx=$i [$(($i - 1)) $(($n - $i))]";

  head -n $(($i - 1))  $dataset   >  $output/train.dat;
  tail -n $(($n - $i)) $dataset   >> $output/train.dat;
  head -n $i $dataset | tail -n 1 >  $output/test.dat;

  ./tcpp -d ${output}/train.dat -t ${output}/test.dat -R -D l2 -m knn -k 1 >> ${output}/ResKNNLOO;

done

