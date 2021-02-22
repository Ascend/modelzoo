pip3.7 install -r requirements.txt

start_time=`date +%s`
python3.7 evalution.py 2>&1 | tee eval.log
end_time=`date +%s`

echo execution time was `expr $end_time - $start_time` s.