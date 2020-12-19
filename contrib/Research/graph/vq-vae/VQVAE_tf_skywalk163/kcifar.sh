ps -aux |grep cifar |awk '{print $2}'|xargs kill -9

