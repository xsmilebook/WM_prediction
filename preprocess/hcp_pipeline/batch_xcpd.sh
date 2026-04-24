#!/bin/bash

for subj in `cat PNC_sublist_QCpassed.txt`
do  
    #subj=$(echo $file | awk -F'_' '{print $1}')
    echo "perform xcpd of subject: $subj"
    sbatch -J ${subj} -o log/out.${subj}.txt -e log/error.${subj}.txt xcpd_24p_csf_global.sh $subj
done
