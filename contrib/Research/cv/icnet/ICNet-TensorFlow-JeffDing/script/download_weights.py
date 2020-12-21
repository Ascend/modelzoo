from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='15S_vZoZZwBsORxtRAMcbdsI99o6Cvo5x',
                                    dest_path='./model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
                                    unzip=False)
gdd.download_file_from_google_drive(file_id='17ZILbQ7Qazg7teb567CIPJ30FD57bVVg',
                                    dest_path='./model/cityscapes/icnet_cityscapes_train_30k.npy',
                                    unzip=False)
gdd.download_file_from_google_drive(file_id='1Z-slNrKYJpfpELeuh2UlueQG1krF9I4a',
                                    dest_path='./model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy',
                                    unzip=False)
gdd.download_file_from_google_drive(file_id='1tZIHpppPcleamBlXKSzjOqL93gNjWGec',
                                    dest_path='./model/cityscapes/icnet_cityscapes_trainval_90k.npy',
                                    unzip=False)
