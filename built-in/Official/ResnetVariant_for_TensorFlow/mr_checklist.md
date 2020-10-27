# 代码提交自检项

## 1. MR要求

1. 必须：提交的文件完备性和正确性。
   1. 不能包含：.git、.gitignore、.pycharm、.vscode、build、dist、\_\_pycache__、*.pyc。
   1. 不能包含临时测试代码。
   1. 不能包含临时生成的数据文件。
1. 必须：提交没有冲突。
1. 必须：勾选"Delete source branch when merge request is accepted."和"Squash commits when merge request is accepted. "
1. 必须：文件格式（UNIX）和编码（UTF-8）要统一。
1. 必须：~~MR关联issue单（当前CI关联的是JIRA单，待调整为issue后再要求关联）~~。
1. 必须：~~确认是否通过了本地UT（正在协调UT环境，暂时不要求UT）。~~

## 2. 规范性

### 2.1 空行

1. 必须：多个import之间不要有空行。
1. 必须：import段和代码段之间有且仅有两个空行。
1. 必须：普通函数之间有且仅有两个空行。
1. 必须：类内函数有且仅有一个空行。
1. 建议：函数内不要有空行，一个函数只做一件事。
1. 必须：文件末尾有且仅有一个空行。

### 2.2 空格

1. 必须：空行中不包含空格。
2. 必须：行尾部不包含空格。
3. 必须：使用空格替换所有tab。
4. 必须：缩进为四个空格。
5. 必须：函数参数中包含的运算和赋值符两边不要空格。
6. 必须：普通运算符和赋值符两边有且仅有一个空格，单元运算符不要加空格。
7. 必须：括号靠内一侧不加空格，靠外一侧有且仅有一个空格。

### 2.3 单行

1. 必须：一行只能包含一个语句。
2. 必须：单行不能超过120字符，最好80字符内。
3. 必须：删除所有被注释掉的代码行。

### 2.4 import

1. 必须：import 的顺序是：基础库 -> 三方库 -> vega库。
2. 建议：多个 import 不要写在一行。
3. 建议：避免 from pkg import * 。

### 2.5 注释

1. 必须：文件、类、函数必须有注释。
2. 必须：文件头部注释，格式如下：

    ```python
    # -*- coding: utf-8 -*-
    """brief description.

    detail decription.
    """
    ```

3. 必须：类和函数的注释采用 sphinx 风格，参考文件末尾附录。
4. 必须：尽量避免单行注释，在被注释的行上面注释，以#开头，#后留一空格。格式如下：

    ```python
    # Compensate for border
    x = x + 1
    ```

### 2.6 命名

1. 必须：使用小写加下划线(lower_with_under)的风格的有：包（Package）、模块（Module）、函数（Function）、方法（Method）、（Function Parameters）、变量（variable）、文件名。
2. 必须：大写字母开头的单词（CapWords）风格的有：类（Class）
3. 必须：使用大写字母加下划线风格的有：常量（constant）
4. 必须：类或对象的私有成员用单下划线_开头
5. 建议：对于需要被继承的基类成员，如果想要防止与派生类成员重名，可用双下划线__开头。

### 2.7 类型判断

1. 必须：使用is None判断是否为空。
2. 必须：使用isinstance函数替代type做类型检查。

    ```python
    def sample_sort_list(sample_inst):
        if sample_inst is []:
            return
        sample_inst.sort()

    # 替换为：
    def sample_sort_list(sample_inst):
        if not isinstance(sample_inst, list):
            raise TypeError(r"sample_sort_list in para type error %s" % type(sample_inst))
        sample_inst.sort()
    ```

### 2.8 异常处理

1. 必须：若open了某个对象，比如文件，必须在使用try…except…ﬁnally…结构保证操作对象的释放。部分场景可以使用with open来替代try结构。
2. 建议：不要使用“except:”语句来捕获所有异常，应该明确期望处理的异常，不能处理的交由上层处理。
3. 建议：尽量用异常来表示特殊情况，而不要返回None。
4. 必须：使用except X as x替代except X, x。
5. 必须：使用try替代assert。

### 2.9 安全

1. 建议：使用dict.get(key)替代dict[key]
1. 建议：不要在一个函数体中连续使用同一变量名。
1. 建议：使用iterable替代range。

    ```python
    for i in range(len(my_list)):
        print(my_list[i])

    # 替换为：
    for x in my_list:
        print(x)

    # 需要序号，替换为：
    for x in enumerate(my_list):
        print(x)
    ```

1. 建议：函数参数中的可变参数不要使用默认值，在定义时使用 None。
1. 建议：使用subprocess模块代替os.system模块。
1. 建议：使用with语句操作文件。

### 2.10 ~~复杂度 （具体数值待确定，暂不要求）~~

1. 函数：
   1. 必须：代码行数：不超过了100行
   2. 必须：循环：不高于4层循环
2. 类：
   1. 必须：继承层数：不多于5层
3. 文件：
   1. 必须：代码行数：不超过200行

## 3. 性能

1. 建议：多线程适用于阻塞式IO场景，不适用于并行计算场景。可考虑使用multiprocessing，或者concurrent.futures。
2. 建议：在list成员个数可以预知的情况下，创建list时需预留空间正好容纳所有成员的空间。

    ```python
    # 不建议：
    members = [] for i in range(1, 1000000):
        members.append(i)

    # 替换为：
    members = [None] * 1000000
    for i in range(1, 1000000):
        members[i] = i
    ```

3. 建议：在成员个数及内容皆不变的场景下尽量使用tuple替代 list
4. 建议：对于频繁使用的外界对象，尽量使用局部变量来引用

    ```python
    # 不建议：
    import math
    def afunc():
        for x in xrange(100000):
            math.tan(x)
    # 不建议：
    from math import tan
    def afunc():
        for x in xrange(100000):
            tan(x)
    # 建议：
    import math
    def afunc(tan=math.tan):
        for x in xrange(100000):
            tan(x)
    ```

5. 建议：尽量使用generator comprehension代替list comprehension。

    ```python
    # 不建议：
    even_cnt = len([x for x in range(10) if x % 2 == 0])
    # 建议：
    even_cnt = sum(1 for x in range(10) if x % 2 == 0)
    ```

6. 建议：使用format方法、"%"操作符和join方法代替"+"和"+="操作符

7. 建议：使用推导式替换简单循环操作，代码清晰精炼。

    ```python
    odd_num_list = []
    for i in range(100):
        if i % 2 == 1:
            odd_num_list.append(i)

    # 替换为：
    odd_num_list = [i for i in range(100) if i % 2 == 1]
    ```

## Vega开发要求（待补充）

### 系统调用

1. 必须：不允许使用 fuser -k /dev/nvidia*。

### 配置（待补充）

1. 必须：使用缺省的全局配置文件，在此文件上添加各个算法的配置信息。
2. 建议：一个pipeline使用一份配置文件。

### 目录操作

1. 必须：使用 self.make_dir() 替代 os.makedirs()。
2. 必须：使用 self.join_path() 替代 os.path.join()。
3. 必须：不要在代码中设置logging的级别
4. 必须：通过 self.local_output_path 来获取本地输出路径，不要直接绝对路径。
5. 必须：通过 self.local_worker_path 来获取worker路径，不要直接指定绝对路径。
6. 必须：通过 self.remote_output_path 来获取S3存取路径。
7. 必须：通过 self.remote_worker_path 来获取worker的S3存取路径。
8. 必须：使用 self.upload_task_folder()、self.upload_worker_folder()、self.download_worker_folder() 来上传下载整个目录。

## 参考

### 注释格式

https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
http://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain

```python
class SimpleBleDevice(Peripheral):
    """This is a conceptual class representation of a simple BLE device (GATT Server). It is essentially an extended combination of the :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    :param client: A handle to the :class:`simpleble.SimpleBleClient` client object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    :param addr: Device MAC address, defaults to None
    :type addr: str, optional
    :param addrType: Device address type - one of ADDR_TYPE_PUBLIC or ADDR_TYPE_RANDOM, defaults to ADDR_TYPE_PUBLIC
    :type addrType: str, optional
    :param iface: Bluetooth interface number (0 = /dev/hci0) used for the connection, defaults to 0
    :type iface: int, optional
    :param data: A list of tuples (adtype, description, value) containing the AD type code, human-readable description and value for all available advertising data items, defaults to None
    :type data: list, optional
    :param rssi: Received Signal Strength Indication for the last received broadcast from the device. This is an integer value measured in dB, where 0 dB is the maximum (theoretical) signal strength, and more negative numbers indicate a weaker signal, defaults to 0
    :type rssi: int, optional
    :param connectable: `True` if the device supports connections, and `False` otherwise (typically used for advertising ‘beacons’)., defaults to `False`
    :type connectable: bool, optional
    :param updateCount: Integer count of the number of advertising packets received from the device so far, defaults to 0
    :type updateCount: int, optional
    """

    def __init__(self, client, addr=None, addrType=ADDR_TYPE_PUBLIC, iface=0, data=None, rssi=0, connectable=False, updateCount=0):
        """Constructor method
        """
        super().__init__(deviceAddr=None, addrType=addrType, iface=iface)
        self.addr = addr
        self.addrType = addrType
        self.iface = iface
        self.rssi = rssi
        self.connectable = connectable
        self.updateCount = updateCount
        self.data = data
        self._connected = False
        self._services = []
        self._characteristics = []
        self._client = client

    def getServices(self, uuids=None):
        """Returns a list of :class:`bluepy.blte.Service` objects representing the services offered by the device. This will perform Bluetooth service discovery if this has not already been done; otherwise it will return a cached list of services immediately..

        :param uuids: A list of string service UUIDs to be discovered, defaults to None
        :type uuids: list, optional
        :return: A list of the discovered :class:`bluepy.blte.Service` objects, which match the provided ``uuids``
        :rtype: list On Python 3.x, this returns a dictionary view object, not a list
        """
        self._services = []
        if(uuids is not None):
            for uuid in uuids:
                try:
                    service = self.getServiceByUUID(uuid)
                    self.services.append(service)
                except BTLEException:
                    pass
        else:
            self._services = super().getServices()
        return self._services

    def setNotificationCallback(self, callback):
        """Set the callback function to be executed when the device sends a notification to the client.

        :param callback: A function handle of the form ``callback(client, characteristic, data)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback, ``characteristic`` is the notified :class:`bluepy.blte.Characteristic` object and data is a `bytearray` containing the updated value. Defaults to None
        :type callback: function, optional
        """
        self.withDelegate(
            SimpleBleNotificationDelegate(
                callback,
                client=self._client
            )
        )

    def getCharacteristics(self, startHnd=1, endHnd=0xFFFF, uuids=None):
        """Returns a list containing :class:`bluepy.btle.Characteristic` objects for the peripheral. If no arguments are given, will return all characteristics. If startHnd and/or endHnd are given, the list is restricted to characteristics whose handles are within the given range.

        :param startHnd: Start index, defaults to 1
        :type startHnd: int, optional
        :param endHnd: End index, defaults to 0xFFFF
        :type endHnd: int, optional
        :param uuids: a list of UUID strings, defaults to None
        :type uuids: list, optional
        :return: List of returned :class:`bluepy.btle.Characteristic` objects
        :rtype: list
        """
        self._characteristics = []
        if(uuids is not None):
            for uuid in uuids:
                try:
                    characteristic = super().getCharacteristics(
                        startHnd, endHnd, uuid)[0]
                    self._characteristics.append(characteristic)
                except BTLEException:
                    pass
        else:
            self._characteristics = super().getCharacteristics(startHnd, endHnd)
        return self._characteristics

    def connect(self):
        """Attempts to initiate a connection with the device.

        :return: `True` if connection was successful, `False` otherwise
        :rtype: bool
        """
        try:
            super().connect(self.addr, addrType=self.addrType, iface=self.iface)
        except BTLEException as ex:
            self._connected = False
            return (False, ex)
        self._connected = True
        return True

    def disconnect(self):
        """Drops existing connection to device
        """
        super().disconnect()
        self._connected = False

    def isConnected(self):
        """Checks to see if device is connected

        :return: `True` if connected, `False` otherwise
        :rtype: bool
        """
        return self._connected

    def printInfo(self):
        """Print info about device
        """
        print("Device %s (%s), RSSI=%d dB" %
            (self.addr, self.addrType, self.rssi))
        for (adtype, desc, value) in self.data:
            print("  %s = %s" % (desc, value))

```
