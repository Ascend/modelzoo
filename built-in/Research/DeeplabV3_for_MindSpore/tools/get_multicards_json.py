import json
import os
import sys

def get_multicards_json(server_id):

    hccn_configs = open('/etc/hccn.conf', 'r').readlines()
    device_ips = {}
    for hccn_item in hccn_configs:
        hccn_item = hccn_item.strip()
        if hccn_item.startswith('address_'):
            device_id, device_ip = hccn_item.split('=')
            device_id = device_id.split('_')[1]
            device_ips[device_id] = device_ip
            print('device_id:{}, device_ip:{}'.format(device_id, 
                device_ip))
    hccn_table = {}
    hccn_table['board_id'] = '0x0000'
    hccn_table['chip_info'] = '910'
    hccn_table['deploy_mode'] = 'lab'
    hccn_table['group_count'] = '1'
    hccn_table['group_list'] = []
    instance_list = []
    usable_dev = ''
    for instance_id in range(8):
        instance = {}
        instance['devices'] = []
        device_id = str(instance_id)
        device_ip = device_ips[device_id]
        usable_dev += str(device_id)
        instance['devices'].append({
            'device_id': device_id,
            'device_ip': device_ip,
        })
        instance['rank_id'] = str(instance_id)
        instance['server_id'] = server_id
        instance_list.append(instance)
    hccn_table['group_list'].append({
        'device_num': '8',
        'server_num': '1',
        'group_name': '',
        'instance_count': '8',
        'instance_list': instance_list,
    })
    hccn_table['para_plane_nic_location'] = 'device'
    hccn_table['para_plane_nic_name'] = []
    for instance_id in range(8):
        hccn_table['para_plane_nic_name'].append('eth{}'.format(instance_id))
    hccn_table['para_plane_nic_num'] = '8'
    hccn_table['status'] = 'completed'
    import json
    table_fn = os.path.join(os.getcwd(), 'rank_table_8p.json')
    print(table_fn)
    with open(table_fn, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)


server_id = sys.argv[1]
get_multicards_json(server_id)
