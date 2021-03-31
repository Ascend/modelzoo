#!/bash/bin

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mpirun --allow-run-as-root \
       --mca btl_tcp_if_include ens12f0 \
       -bind-to socket -map-by slot    \
       -x LD_LIBRARY_PATH \
       -x NCCL_DEBUG=INFO \
       -x PATH  \
       -np 8 \
       -H 192.168.0.12:8    \
       -x NCCL_IB_DISABLE=0 \
       -x NCCL_SOCKET_IFNAME=ens12f0 \
       -x NCCL_MAX_NRINGS=4 \
       -mca pml ob1 -x NCCL_SOCKET_IFNAME=ens12f0 \
       --oversubscribe python3 main.py
