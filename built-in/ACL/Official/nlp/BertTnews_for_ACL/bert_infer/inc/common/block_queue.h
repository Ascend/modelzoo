/* *
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _BLOCK_QUEUE_H
#define _BLOCK_QUEUE_H

#include "utility.h"
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <iostream>

template <typename T> class BlockingQueue {
    std::string _name;
    std::queue<T> dataQueue;
    int capacity;
    std::mutex _mutex;
    std::condition_variable _not_full;
    std::condition_variable _not_empty;
    int cnt;

public:
    BlockingQueue<T> &operator = (const BlockingQueue<T> &other) = delete;

    BlockingQueue(std::string name, int size = 10, int ct = 0) : _name(name), capacity(size), cnt(ct) {}

    void put(T data)
    {
        std::unique_lock<std::mutex> lock(_mutex);

        while (isFull()) {
            _not_full.wait(lock);
        }

        dataQueue.push(data);
        _not_empty.notify_one();
    }

    T take()
    {
        std::unique_lock<std::mutex> lock(_mutex);

        while (isEmpty()) {
            _not_empty.wait(lock);
        }

        T val = dataQueue.front();
        cnt++;
        dataQueue.pop();
        _not_full.notify_one();
        return val;
    }

    size_t size()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return dataQueue.size();
    }

private:
    bool isEmpty()
    {
        return dataQueue.empty();
    }

    bool isFull()
    {
        return dataQueue.size() >= capacity;
    }
};

#endif
