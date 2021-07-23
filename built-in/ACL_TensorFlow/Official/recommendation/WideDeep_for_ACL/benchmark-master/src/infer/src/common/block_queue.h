/* 
 *
 * Copyright (C) Huawei Technologies Co., Ltd. 2020-2099. All Rights Reserved.
 * Description: 定义数据队列 
 * Author: Atlas
 * Create: 2020-02-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 */

#ifndef M_BLOCKING_QUEUE_USE_MATRIX_H
#define M_BLOCKING_QUEUE_USE_MATRIX_H

#include <list>
#include <stdint.h>
#include <mutex>
#include <condition_variable>

static const int DEFAULT_MAX_QUEUE_SIZE = 2048;

template <typename T> class BlockingQueue {
public:
    BlockingQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : max_size_(maxSize), is_stoped_(false) {}

    ~BlockingQueue() {}

    bool Pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.empty() && !is_stoped_) {
            empty_cond_.wait(lock);
        }

        if (queue_.empty()) {
            return false;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        if (is_stoped_) {
            return false;
        }

        full_cond_.notify_one();

        return true;
    }

    bool Push(const T &item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.size() >= max_size_ && isWait && !is_stoped_) {
            full_cond_.wait(lock);
        }

        if (is_stoped_) {
            return false;
        }

        queue_.push_back(item);

        empty_cond_.notify_one();

        return true;
    }


    bool Push_Front(const T &item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.size() >= max_size_ && isWait && !is_stoped_) {
            full_cond_.wait(lock);
        }

        if (is_stoped_) {
            return false;
        }

        queue_.push_front(item);

        empty_cond_.notify_one();

        return true;
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_stoped_ = true;
        }

        full_cond_.notify_all();
        empty_cond_.notify_all();
    }

    void Restart()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            is_stoped_ = false;
        }
    }

    // if the queue is stoped ,need call this function to release the unprocessed items
    std::list<T> GetRemainItems()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!is_stoped_) {
            return std::list<T>();
        }

        return queue_;
    }

    bool GetBackItem(T &item)
    {
        if (is_stoped_) {
            return false;
        }

        if (queue_.empty()) {
            return false;
        }

        item = queue_.back();
        return true;
    }

    std::mutex *GetLock()
    {
        return &mutex_;
    }

    bool IsFull()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size() >= max_size_;
    }
    int GetSize()
    {
        return queue_.size();
    }
    bool IsEmpty()
    {
        return queue_.empty();
    }

    void Clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.clear();
    }

private:
    std::list<T> queue_;
    std::mutex mutex_;
    std::condition_variable empty_cond_;
    std::condition_variable full_cond_;
    uint32_t max_size_;

    bool is_stoped_;
};
#endif // MATRIX_GRAPH_BLOCKING_QUEUE_H
