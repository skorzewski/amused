#!/bin/bash

echo -e '                    m   dl ll  e   dim do rdo'

grep $1 new_experiment_results/* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -n | sed 's/mean/AVG/g ; s/maxabs/MXA/g ; s/max/MAX/g ; s/zero/ZRO/g ; s/manners/MNR/g ; s/reporting_clauses/RCL/g ; s/sentences/SEN/g ; s/neighbors/NGH/g ; s/[^0-9.A-Z]\+/\t/g ; s/^\t//g ; s/[.]\t$//g' | awk 'NF==8{print}{}' | column -nts$'\t'
