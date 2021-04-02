# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import papermill as pm
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.models.lstur import LSTURModel
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator


@pytest.mark.smoke
@pytest.mark.gpu
def test_model_nrms(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"nrms.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = NRMSModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.smoke
@pytest.mark.gpu
def test_model_naml(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding_all.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(tmp, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(tmp, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"naml.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        epochs=1,
    )

    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator)
    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.smoke
@pytest.mark.gpu
def test_model_lstur(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"lstur.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = LSTURModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )


@pytest.mark.smoke
@pytest.mark.gpu
def test_model_npa(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"lstur.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.blob.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hparams is not None

    iterator = MINDIterator
    model = LSTURModel(hparams, iterator)

    assert model.run_eval(valid_news_file, valid_behaviors_file) is not None
    assert isinstance(
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        ),
        BaseModel,
    )
