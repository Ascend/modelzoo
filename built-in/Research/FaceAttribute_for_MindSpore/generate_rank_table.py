# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Generate rank table."""
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="mindspore distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for D training, this is recommended to be set "
                             "to the number of D in your system so that "
                             "each process can be bound to a single D.")
    parser.add_argument("--visible_devices", type=str, default="0,1,2,3,4,5,6,7",
                        help="will use the visible devices sequentially")
    parser.add_argument("--server_id", type=str, default="",
                        help="server ip")
    parser.add_argument("--device_mode", type=str, default="A+K",
                        help="A+K or A+X")

    return parser.parse_args()


def main():
    args = parse_args()
    print('args:{}'.format(args))
    visible_devices = args.visible_devices.split(',')
    assert len(visible_devices) >= args.nproc_per_node
    print('visible_devices:{}'.format(visible_devices))
    if args.server_id == '':
        print('pleaser input server ip!!!')
        exit(0)
    print('server_id:{}'.format(args.server_id))

    hccn_configs = open('/etc/hccn.conf', 'r').readlines()
    device_ips = {}
    for hccn_item in hccn_configs:
        hccn_item = hccn_item.strip()
        if hccn_item.startswith('address_'):
            device_id, device_ip = hccn_item.split('=')
            device_id = device_id.split('_')[1]
            device_ips[device_id] = device_ip
            print('device_id:{}, device_ip:{}'.format(device_id, device_ip))

    if args.nproc_per_node != 1:
        hccn_table = {}
        if args.device_mode == 'A+K':
            hccn_table['board_id'] = '0x002f'
        else:
            hccn_table['board_id'] = '0x0000'
        hccn_table['chip_info'] = '910'
        hccn_table['deploy_mode'] = 'lab'
        hccn_table['group_count'] = '1'
        hccn_table['group_list'] = []
        instance_list = []
        usable_dev = ''
        for instance_id in range(args.nproc_per_node):
            instance = {}
            instance['devices'] = []
            device_id = visible_devices[instance_id]
            device_ip = device_ips[device_id]
            usable_dev += str(device_id)
            instance['devices'].append({
                'device_id': device_id,
                'device_ip': device_ip,
            })
            instance['rank_id'] = str(instance_id)
            instance['server_id'] = args.server_id
            instance_list.append(instance)
        hccn_table['group_list'].append({
            'device_num': str(args.nproc_per_node),
            'server_num': '1',
            'group_name': '',
            'instance_count': str(args.nproc_per_node),
            'instance_list': instance_list,
        })
        hccn_table['para_plane_nic_location'] = 'device'
        hccn_table['para_plane_nic_name'] = []
        for instance_id in range(args.nproc_per_node):
            eth_id = visible_devices[instance_id]
            hccn_table['para_plane_nic_name'].append('eth{}'.format(eth_id))
        hccn_table['para_plane_nic_num'] = str(args.nproc_per_node)
        hccn_table['status'] = 'completed'
        import json
        table_fn = os.path.join('scripts/rank_table_{}p.json'.format(args.nproc_per_node))
        with open(table_fn, 'w') as table_fp:
            json.dump(hccn_table, table_fp, indent=4)

    if args.nproc_per_node == 1:
        for instance_id in range(8):
            hccn_table = {}
            if args.device_mode == 'A+K':
                hccn_table['board_id'] = '0x002f'
            else:
                hccn_table['board_id'] = '0x0000'
            hccn_table['chip_info'] = '910'
            hccn_table['deploy_mode'] = 'lab'
            hccn_table['group_count'] = '1'
            hccn_table['group_list'] = []
            instance_list = []
            usable_dev = ''

            instance = {}
            instance['devices'] = []
            device_id = visible_devices[instance_id]
            device_ip = device_ips[device_id]
            usable_dev += str(device_id)
            instance['devices'].append({
                'device_id': device_id,
                'device_ip': device_ip,
            })
            instance['rank_id'] = '0'
            instance['server_id'] = args.server_id
            instance_list.append(instance)

            hccn_table['group_list'].append({
                'device_num': '1',
                'server_num': '1',
                'group_name': '',
                'instance_count': '1',
                'instance_list': instance_list,
            })
            hccn_table['para_plane_nic_location'] = 'device'
            hccn_table['para_plane_nic_name'] = []

            eth_id = visible_devices[instance_id]
            hccn_table['para_plane_nic_name'].append('eth{}'.format(eth_id))

            hccn_table['para_plane_nic_num'] = '1'
            hccn_table['status'] = 'completed'
            import json
            table_fn = os.path.join('scripts/rank_table_1p_{}.json'.format(instance_id))
            with open(table_fn, 'w') as table_fp:
                json.dump(hccn_table, table_fp, indent=4)


if __name__ == "__main__":
    main()
