from enum import Enum
from typing import NamedTuple


class Dataset(NamedTuple):
    dataset_name: str
    dataset_url: str


class HuggingFaceDataset(Dataset, Enum):
    RESD = Dataset('Russian Emotional Speech Dialoges', 'aniemore/resd')
    RESD_ANNOTATED = Dataset('Russian Emotional Speech Dialoges [Annotated]', 'aniemore/resd-annotated')
    RESD_ANNOTATED_MULTI = Dataset(
        'Russian Emotional Speech Dialoges [Annotated with multi labels]', 'aniemore/resd-annotated-multi')
    CEDR_M7 = Dataset('Corpus for Emotions Detecting moods 7', 'aniemore/cedr-m7')
