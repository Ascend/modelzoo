import os
import mindspore.dataset as ds

def create_dataset(base_path, filename, batch_size, num_epochs, columns_list, num_consumer):
    """Create dataset"""

    path = os.path.join(base_path, filename)
    dtrain = ds.MindDataset(path, columns_list, num_consumer)
    dtrain = dtrain.shuffle(buffer_size=dtrain.get_dataset_size())
    dtrain = dtrain.batch(batch_size, drop_remainder=True)
    dtrain = dtrain.repeat(count=num_epochs)

    return dtrain

