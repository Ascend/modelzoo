start_str=$1
end_str=$2
log_file=$3


sed -n "/$start_str/, /$end_str/p"  log.log   >./logs/$log_file
cat ./logs/$log_file |grep ERROR
