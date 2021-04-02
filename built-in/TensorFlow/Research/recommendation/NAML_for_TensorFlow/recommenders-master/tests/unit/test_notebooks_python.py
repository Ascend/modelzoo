# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.notebooks
def test_template_runs(notebooks):
    notebook_path = notebooks["template"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )
    nb = sb.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.papermill_dataframe
    assert df.shape[0] == 2
    check_version = df.loc[df["name"] == "checked_version", "value"].values[0]
    assert check_version is True


@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
def test_sar_deep_dive_runs(notebooks):
    notebook_path = notebooks["sar_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
def test_baseline_deep_dive_runs(notebooks):
    notebook_path = notebooks["baseline_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
def test_surprise_deep_dive_runs(notebooks):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.vw
@pytest.mark.notebooks
def test_vw_deep_dive_runs(notebooks):
    notebook_path = notebooks["vowpal_wabbit_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
def test_lightgbm(notebooks):
    notebook_path = notebooks["lightgbm_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            MAX_LEAF=32,
            MIN_DATA=20,
            NUM_OF_TREES=10,
            TREE_LEARNING_RATE=0.15,
            EARLY_STOPPING_ROUNDS=20,
            METRIC="auc",
        ),
    )


@pytest.mark.notebooks
def test_wikidata_runs(notebooks, tmp):
    notebook_path = notebooks["wikidata_knowledge_graph"]
    MOVIELENS_SAMPLE_SIZE = 5
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k",
            MOVIELENS_SAMPLE=True,
            MOVIELENS_SAMPLE_SIZE=MOVIELENS_SAMPLE_SIZE,
        ),
    )


@pytest.mark.notebooks
def test_rlrmc_quickstart_runs(notebooks):
    notebook_path = notebooks["rlrmc_quickstart"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
def test_cornac_deep_dive_runs(notebooks):
    notebook_path = notebooks["cornac_bpr_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
