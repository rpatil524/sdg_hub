"""Tests for the MeltColumnsBlock functionality.

This module contains tests that verify the correct behavior of the MeltColumnsBlock,
including column melting, value mapping, and edge case handling.
"""

# Third Party
# First Party
from sdg_hub.core.blocks.transform import MeltColumnsBlock
from sdg_hub.core.utils.error_handling import MissingColumnError
import pandas as pd
import pytest


def test_flatten_columns_basic():
    """Test basic flattening functionality with simple data."""
    # Create sample data
    data = {
        "id": [1, 2],
        "summary_detailed": ["detailed1", "detailed2"],
        "summary_extractive": ["extractive1", "extractive2"],
        "other_col": ["other1", "other2"],
    }
    dataset = pd.DataFrame(data)

    # Initialize block
    block = MeltColumnsBlock(
        block_name="test_flatten",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    # Generate flattened dataset
    result = block.generate(dataset)

    # Verify results
    assert len(result) == 4  # 2 rows * 2 columns to flatten
    assert "id" in result.columns.tolist()
    assert "other_col" in result.columns.tolist()
    assert "summary" in result.columns.tolist()
    assert "dataset_type" in result.columns.tolist()

    # Check specific values
    assert "detailed1" in result["summary"].tolist()
    assert "extractive1" in result["summary"].tolist()
    assert "summary_detailed" in result["dataset_type"].tolist()
    assert "summary_extractive" in result["dataset_type"].tolist()


def test_flatten_columns_with_empty_dataset():
    """Test flattening with an empty dataset."""
    data = {"id": [], "summary_detailed": [], "summary_extractive": [], "other_col": []}
    dataset = pd.DataFrame(data)

    block = MeltColumnsBlock(
        block_name="test_flatten_empty",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    result = block.generate(dataset)
    assert len(result) == 0
    assert all(
        col in result.columns.tolist()
        for col in ["id", "other_col", "summary", "dataset_type"]
    )


def test_flatten_columns_with_missing_values():
    """Test flattening with missing values in the data."""
    data = {
        "id": [1, 2, 3],
        "summary_detailed": ["detailed1", None, "detailed3"],
        "summary_extractive": ["extractive1", "extractive2", None],
        "other_col": ["other1", "other2", "other3"],
    }
    dataset = pd.DataFrame(data)

    block = MeltColumnsBlock(
        block_name="test_flatten_missing",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    result = block.generate(dataset)
    assert len(result) == 6  # 3 rows * 2 columns to flatten
    assert None in result["summary"].tolist()


def test_flatten_columns_with_all_columns():
    """Test flattening all columns except id."""
    data = {
        "id": [1, 2],
        "summary_detailed": ["detailed1", "detailed2"],
        "summary_extractive": ["extractive1", "extractive2"],
        "summary_atomic_facts": ["atomic1", "atomic2"],
        "base_document": ["base1", "base2"],
    }
    dataset = pd.DataFrame(data)

    block = MeltColumnsBlock(
        block_name="test_flatten_all",
        input_cols=[
            "summary_detailed",
            "summary_extractive",
            "summary_atomic_facts",
            "base_document",
        ],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    result = block.generate(dataset)
    assert len(result) == 8  # 2 rows * 4 columns to flatten
    assert len(result.columns.tolist()) == 3  # id, dataset_type, and summary columns
    assert "id" in result.columns.tolist()
    assert "summary" in result.columns.tolist()
    assert "dataset_type" in result.columns.tolist()


def test_flatten_columns_with_invalid_input():
    """Test flattening with invalid input data."""
    data = {"id": [1, 2], "summary_detailed": ["detailed1", "detailed2"]}
    dataset = pd.DataFrame(data)

    # Test with non-existent column
    block = MeltColumnsBlock(
        block_name="test_flatten_invalid",
        input_cols=["non_existent_column"],  # Only the columns to be melted
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    # Should raise MissingColumnError for missing columns during validation
    with pytest.raises(MissingColumnError):
        block(dataset)


def test_flatten_columns_with_empty_columns():
    """Test flattening with columns containing all None values."""
    data = {
        "id": [1, 2],
        "summary_detailed": [None, None],
        "summary_extractive": ["extractive1", "extractive2"],
        "other_col": ["other1", "other2"],
    }
    dataset = pd.DataFrame(data)

    block = MeltColumnsBlock(
        block_name="test_flatten_empty_cols",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    result = block.generate(dataset)

    # Verify that None values are preserved
    assert None in result["summary"].tolist()
    # Verify that the column with None values still appears in dataset_type
    assert "summary_detailed" in result["dataset_type"].tolist()


def test_flatten_columns_with_different_lengths():
    """Test flattening with columns of different lengths."""
    # Create data with different lengths
    data = {
        "id": [1, 2, 3],
        "summary_detailed": ["detailed1", "detailed2"],  # Shorter than id
        "summary_extractive": ["extractive1", "extractive2", "extractive3"],
        "other_col": ["other1", "other2", "other3"],
    }

    # Dataset creation will fail due to different lengths
    with pytest.raises(Exception) as exc_info:
        pd.DataFrame(data)
    assert (
        "expected length" in str(exc_info.value).lower()
        or "same length" in str(exc_info.value).lower()
    )


def test_flatten_columns_verify_column_names():
    """Test that value_name and var_name are used correctly in the output."""
    data = {
        "id": [1],
        "summary_detailed": ["detailed1"],
        "summary_extractive": ["extractive1"],
    }
    dataset = pd.DataFrame(data)

    # Test with custom names
    custom_value_name = "custom_value"
    custom_var_name = "custom_var"

    block = MeltColumnsBlock(
        block_name="test_flatten_custom_names",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=[
            custom_value_name,
            custom_var_name,
        ],  # [value_column, variable_column]
    )

    result = block.generate(dataset)

    # Verify custom column names are used
    assert custom_value_name in result.columns.tolist()
    assert custom_var_name in result.columns.tolist()
    assert "summary" not in result.columns.tolist()
    assert "dataset_type" not in result.columns.tolist()

    # Verify content is correctly mapped to custom names
    assert "detailed1" in result[custom_value_name].tolist()
    assert "extractive1" in result[custom_value_name].tolist()
    assert "summary_detailed" in result[custom_var_name].tolist()
    assert "summary_extractive" in result[custom_var_name].tolist()


def test_flatten_columns_validation_errors():
    """Test Pydantic validation errors in MeltColumnsBlock."""
    # Test empty input_cols
    with pytest.raises(ValueError, match="input_cols cannot be empty"):
        MeltColumnsBlock(
            block_name="test_flatten_empty_vars",
            input_cols=[],  # Empty columns to melt
            output_cols=["value", "variable"],  # [value_column, variable_column]
        )

    # Test wrong number of output columns (not exactly 2)
    with pytest.raises(
        ValueError, match="MeltColumnsBlock expects exactly two output columns"
    ):
        MeltColumnsBlock(
            block_name="test_flatten_wrong_outputs",
            input_cols=["col1"],  # Columns to melt
            output_cols=["value"],  # Only 1 output column
        )

    # Test too many output columns
    with pytest.raises(
        ValueError, match="MeltColumnsBlock expects exactly two output columns"
    ):
        MeltColumnsBlock(
            block_name="test_flatten_too_many_outputs",
            input_cols=["col1"],  # Columns to melt
            output_cols=["value", "variable", "extra"],  # 3 output columns
        )


def test_flatten_columns_with_input_output_cols():
    """Test MeltColumnsBlock with explicit input_cols and output_cols."""
    data = {
        "id": [1, 2],
        "summary_detailed": ["detailed1", "detailed2"],
        "summary_extractive": ["extractive1", "extractive2"],
        "other_col": ["other1", "other2"],
    }
    dataset = pd.DataFrame(data)

    # Test with explicit input_cols and output_cols
    block = MeltColumnsBlock(
        block_name="test_flatten_explicit_cols",
        input_cols=["summary_detailed", "summary_extractive"],  # Columns to melt
        output_cols=["summary", "dataset_type"],  # [value_column, variable_column]
    )

    result = block.generate(dataset)

    # Verify results
    assert len(result) == 4  # 2 rows * 2 columns to flatten
    assert "id" in result.columns.tolist()
    assert "other_col" in result.columns.tolist()
    assert "summary" in result.columns.tolist()
    assert "dataset_type" in result.columns.tolist()
