awk -F'|' -v OFS='|' 'length($0)>1{print substr($1,1,6) "xsj" substr($1,7),substr($2,1,4) "xsj" substr($2,5),substr($3,1,7) "xsj" substr($3,8),$4,$5,$6,"xsj.npy"; next} {print}' train.txt > train2.txt
ls | xargs -I{} bash -c 'str={}; mv {}  ${str:0:6}xsj${str:6}'
ls | xargs -I{} bash -c 'str={}; mv {}  ${str:0:4}xsj${str:4}'
 ls | xargs -I{} bash -c 'str={}; mv {}  ${str:0:7}xsj${str:7}'
