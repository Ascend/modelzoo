dir=`pwd`
rm -rf ge_*
rm -rf aicpu_*
rm -rf model_ckpt
rm -rf kernel_meta
 
cd ${dir}/D0
${dir}/D0/clean.sh

cd ${dir}/D1
${dir}/D1/clean.sh

cd ${dir}/D2
${dir}/D2/clean.sh

cd ${dir}/D3
${dir}/D3/clean.sh

cd ${dir}/D4
${dir}/D4/clean.sh

cd ${dir}/D5
${dir}/D5/clean.sh

cd ${dir}/D6
${dir}/D6/clean.sh

cd ${dir}/D7
${dir}/D7/clean.sh
