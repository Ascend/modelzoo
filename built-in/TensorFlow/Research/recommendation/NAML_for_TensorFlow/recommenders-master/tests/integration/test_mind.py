# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.dataset.mind import download_mind, extract_mind


@pytest.mark.integration
def test_download_mind(tmp_path):
    train_path, valid_path = download_mind(size="large", dest_path=tmp_path)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 530196631
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 103456245


@pytest.mark.integration
def test_extract_mind(tmp):
    train_zip, valid_zip = download_mind(size="large", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 1373844151
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 40305151
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 84881998
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588

    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 230662527
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 31958202
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 59055351
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
