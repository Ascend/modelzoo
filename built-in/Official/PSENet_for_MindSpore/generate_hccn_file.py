import os
import socket
from argparse import ArgumentParser

RANK_TABLE_SAVE_PATH = './rank_table_4p.json'


def main():
    nproc_per_node = 4

    visible_devices = ['0', '1', '2', '3']

    server_id = socket.gethostbyname(socket.gethostname())

    hccn_configs = open('/etc/hccn.conf', 'r').readlines()
    device_ips = {}
    for hccn_item in hccn_configs:
        hccn_item = hccn_item.strip()
        if hccn_item.startswith('address_'):
            device_id, device_ip = hccn_item.split('=')
            device_id = device_id.split('_')[1]
            device_ips[device_id] = device_ip
            print('device_id:{}, device_ip:{}'.format(device_id, device_ip))

    hccn_table = {}
    hccn_table['board_id'] = '0x002f'  # A+K
    # hccn_table['board_id'] = '0x0000' # A+X

    hccn_table['chip_info'] = '910'
    hccn_table['deploy_mode'] = 'lab'
    hccn_table['group_count'] = '1'
    hccn_table['group_list'] = []
    instance_list = []
    for instance_id in range(nproc_per_node):
        instance = {}
        instance['devices'] = []
        device_id = visible_devices[instance_id]
        device_ip = device_ips[device_id]
        instance['devices'].append({
            'device_id': device_id,
            'device_ip': device_ip,
        })
        instance['rank_id'] = str(instance_id)
        instance['server_id'] = server_id
        instance_list.append(instance)
    hccn_table['group_list'].append({
        'device_num': str(nproc_per_node),
        'server_num': '1',
        'group_name': '',
        'instance_count': str(nproc_per_node),
        'instance_list': instance_list,
    })
    hccn_table['para_plane_nic_location'] = 'device'
    hccn_table['para_plane_nic_name'] = []
    for instance_id in range(nproc_per_node):
        eth_id = visible_devices[instance_id]
        hccn_table['para_plane_nic_name'].append('eth{}'.format(eth_id))
    hccn_table['para_plane_nic_num'] = str(nproc_per_node)
    hccn_table['status'] = 'completed'
    import json
    with open(RANK_TABLE_SAVE_PATH, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)


if __name__ == '__main__':
    if os.path.exists(RANK_TABLE_SAVE_PATH):
        print('Rank table file exists.')
    else:
        print('Generating rank table file.')
        main()
        print('Rank table file generated')
