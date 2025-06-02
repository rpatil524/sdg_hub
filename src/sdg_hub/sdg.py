# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import traceback

# Third Party
from datasets import Dataset
from tqdm import tqdm

# Local
from .logger_config import setup_logger
from .pipeline import Pipeline
from .utils.datautils import safe_concatenate_datasets
from .checkpointer import Checkpointer

logger = setup_logger(__name__)


class SDG:
    def __init__(
        self, pipelines: List[Pipeline], num_workers=1, batch_size=None, save_freq=None
    ) -> None:
        self.pipelines = pipelines
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_freq = save_freq

    def _split_dataset(self, dataset: Dataset, batch_size: int) -> List[Dataset]:
        """Split the dataset into smaller batches."""
        total_size = len(dataset)
        num_batches = (total_size + batch_size - 1) // batch_size

        batches = [
            (i * batch_size, min((i + 1) * batch_size, total_size))
            for i in tqdm(range(num_batches))
        ]

        return batches

    @staticmethod
    def _generate_data(pipelines, input_split, ds, i=None):
        logger.info(f"Processing split {i}")
        input_split = ds.select(range(input_split[0], input_split[1]))
        try:
            for pipeline in pipelines:
                input_split = pipeline.generate(input_split)
            return input_split
        except Exception as e:
            logger.error(f"Error processing split {i}: {e}")
            traceback.print_exc()
            return None

    def generate(self, dataset: Dataset, checkpoint_dir=None) -> Dataset:
        # Initialize checkpointer
        checkpointer = Checkpointer(checkpoint_dir, self.save_freq)
        
        # Load existing checkpoints and determine missing data
        seed_data, pre_generated_data = checkpointer.load_existing_data(dataset)
        
        # If all data has been generated, return the pre-generated data
        if seed_data.num_rows == 0 and pre_generated_data is not None:
            return pre_generated_data

        if not self.batch_size:
            # If batch size is not provided, generate the dataset in a single pass
            generated_dataset = seed_data
            # generated_data is initialized with seed_data, and it gets updated with each pipeline
            for pipeline in self.pipelines:
                generated_dataset = pipeline.generate(seed_data)
            return generated_dataset
        
        logger.info("Splitting the dataset into smaller batches")
        input_splits = self._split_dataset(seed_data, self.batch_size)
        logger.info(
            f"Generating dataset with {len(input_splits)} splits, "
            f"batch size {self.batch_size}, and {self.num_workers} workers"
        )

        generated_data = [pre_generated_data] if pre_generated_data else []
        last_saved_split_index = 0  # To track the last saved split

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_data, self.pipelines, input_split, seed_data, i
                )
                for i, input_split in enumerate(input_splits)
            ]

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                generated_data_split = future.result()  # Ensure each future completes

                if generated_data_split:
                    generated_data.append(generated_data_split)
                    logger.info(f"Finished future processing split {i} \n\n")
                    
                    # Use checkpointer to handle intermediate saves
                    if checkpointer.should_save_checkpoint(i):
                        # Save only the new splits since the last checkpoint
                        new_splits = generated_data[last_saved_split_index : i + 1]
                        checkpoint_dataset = safe_concatenate_datasets(new_splits)
                        # check if checkpoint_dataset is not None
                        if checkpoint_dataset:
                            checkpointer.save_intermediate_checkpoint(checkpoint_dataset)
                            last_saved_split_index = i + 1

        generated_dataset = safe_concatenate_datasets(generated_data)

        return generated_dataset