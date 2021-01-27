# -*- coding:utf-8 -*-

import paramiko
import logging
import time
import sys
import os
sys.dont_write_bytecode = True

class SSHConnection(object):
    '''
    classdocs
    '''
    '''
     #!!=====================================================================
        # 过 程 名： __init__
        # 函数说明：SSHConnection的构造函数, 获取UIHost和mini的环境信息
        # 参数说明：
        #         mini：mini板的环境信息，基于字典的格式，样例格式：
        #          mini={'host':'192.168.4.38','username':'username','password':'passwd'}
        #         UIHost:UIHost的环境信息，格式同mini板
        #         tunnelflag: 是否通过UIHost跳板机，建立ssh隧道在mini板上执行命令；主要基于用例在本地调试的情况
        # 返 回 值：transport的对象
        # 注意事项：无
        # 使用实例：conn = SSHConnection(mini, UIHost,tunnelflag)
        # 对应命令：无
        #!!=====================================================================
    '''
    def __init__(self,mini,UIHost=None,tunnelflag=False):
        '''

        '''
        self._port = 22
        self._transport = None
        self._client = None
        self._tunnelflag=tunnelflag
        self._mini=mini
        self._tunnel=None
        self._UIHost=UIHost
        self._sftp=None
        self._connect()
    
    def _connect(self):
        if self._tunnelflag and self._UIHost:
            pass
        else:
            transport = paramiko.Transport((self._mini['host'], self._port))
            transport.connect(username=self._mini['username'], password=self._mini['password'])
            self._transport = transport
    
    def exec_command(self, command,input=None):
        '''
     #!!=====================================================================
        # 过 程 名： exec_command
        # 函数说明：通过ssh连接，远程执行shell命令
        # 参数说明：
        #         command：执行的shell命令格式，注意，这个函数只适用于执行无交互式的命令
        #         input:交互时shell命令，输入对应的响应消息
        # 返 回 值：执行命令后的返回值
        # 注意事项：无
        # 使用实例：conn.exec_command('df -h')
        # 对应命令：无
        #!!=====================================================================
    '''
        if self._tunnelflag and self._UIHost:
            with SSHTunnelForwarder(
               ssh_address_or_host=(self._UIHost['host'], self._port),
               ssh_username=self._UIHost['username'],
               ssh_password=self._UIHost['password'],
               remote_bind_address=(self._mini['host'], self._port),
               local_bind_address=('0.0.0.0', 10022)
               ) as self._tunnel:
                    self._tunnel.start()
                    self._client = paramiko.SSHClient()
                    self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    self._client.connect('127.0.0.1',10022,self._mini['username'],self._mini['password'])
                    stdin,stdout,stderr = self._client.exec_command(command)
                    stdin.write('root\n')
                    data = stdout.read()
                    if len(data) > 0:
                        logging.debug(data.strip())
                        return data
                    err = stderr.read()
                    if len(err) > 0:
                        logging.debug(err.strip())
                        return err
        else:
            if self._client is None:
                self._client = paramiko.SSHClient()
                self._client._transport = self._transport
            stdin, stdout, stderr = self._client.exec_command(command)
            if input:
                 stdin.write(input+"\n")
            data = stdout.read()
            if len(data) > 0:
                logging.debug(data.strip())
                return data
            err = stderr.read()
            if len(err) > 0:
                logging.debug(err.strip())
                return err
        
    def close(self):
        '''
     #!!=====================================================================
        # 过 程 名： close
        # 函数说明：断开ssh连接，所有的shell命令执行结束后都要执行这个命令
        # 参数说明：无
        # 返 回 值：无
        # 注意事项：无
        # 使用实例：conn.close()
        #!!=====================================================================
    '''
        if self._transport:
            self._transport.close()
        if self._client:
            self._client.close()
        if self._tunnel:
            self._tunnel.close()
    
    def download(self, remotepath, localpath):
        '''
     #!!=====================================================================
        # 过 程 名： download
        # 函数说明：下载对应的文件，注意，remotepath和localpath传入的为文件路径+文件名的格式
        # 参数说明：remotepath   远端路径
        #          localpath  本地下载路径
        # 返 回 值：无
        # 注意事项：无
        # 使用实例：conn.download(remotepath，localpath)
        #!!=====================================================================
    '''
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        if  os.path.exists(os.path.dirname(localpath))==False:
            os.makedirs(os.path.dirname(localpath))
        self._sftp.get(remotepath, localpath)
 
    def put(self, localpath, remotepath):
        '''
     #!!=====================================================================
        # 过 程 名： put
        # 函数说明：上传对应的文件，注意，remotepath和localpath传入的为文件路径+文件名的格式
        # 参数说明：remotepath   远端路径
        #          localpath  本地上传路径
        # 返 回 值：无
        # 注意事项：无
        # 使用实例：conn.put(remotepath，localpath)
        #!!=====================================================================
    '''
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.put(localpath, remotepath)
