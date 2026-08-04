from __future__ import annotations

from typing import List,Tuple
from dataclasses import dataclass
from functools import cached_property

import numpy as np


# Parse annotation 
## segmentation = SegmentationData.from_annotation(annotation_dict)
## multiclass_mask = segmentation.multiclass_mask
## binary_mask = segmentation.binary_mask_any_ich
## subarachnoid_mask = segmentation.binary_mask_for_class(2)
## binary_masks = segmentation.binary_masks_by_class


@dataclass(frozen=True)
class RLEMask:
    shape: tuple[int, int]
    counts: list[int]

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError(f"Expected 2D shape, got {len(self.shape)}D")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"Shape dimensions must be positive, got {self.shape}")
        if not self.counts:
            raise ValueError("RLE counts cannot be empty")
        if len(self.counts) % 2 != 0:
            raise ValueError(f"RLE counts must have even length (class,count pairs), got {len(self.counts)}")
        if any(count < 0 for count in self.counts):
            raise ValueError(f"RLE counts must be non-negative, got {self.counts}")

    @property
    def total_pixels(self) -> int:
        return self.shape[0] * self.shape[1]
    
    @property
    def encoded_length(self) -> int:
        return len(self.counts)
    
    def get_run_pairs(self) -> List[Tuple[int, int]]:
        """Convert flat counts list to pairs of (class_value, count)."""
        pairs = []
        for i in range(0, len(self.counts), 2):
            class_value = self.counts[i]
            class_count = self.counts[i + 1]
            pairs.append((class_value, class_count))
        return pairs


@dataclass(frozen=True)
class SegmentationClass:
    value: int
    name: str
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Class value must be non-negative, got {self.value}")
        if not self.name:
            raise ValueError("Class name cannot be empty")


@dataclass(frozen=True)
class SegmentationData:
    rle_mask: RLEMask
    class_map: list[SegmentationClass]

    @classmethod
    def from_annotation(
        cls,
        annotation: dict,
    ) -> "SegmentationData":

        segmentation = annotation["segmentation_rle"]

        return cls(
            rle_mask=RLEMask(
                shape=tuple(segmentation["shape"]),
                counts=segmentation["counts"],
            ),
            class_map=[
                SegmentationClass(
                    value=item["value"],
                    name=item["name"],
                )
                for item in annotation["class_map"]
            ],
        )

    @cached_property
    def multiclass_mask(self) -> np.ndarray:
        return RLECodec.decode_multiclass(
            self.rle_mask
        )
    
    def multiclass_mask_for_classes(
        self,
        classes: list[int],
    ) -> np.ndarray:
        mask = self.multiclass_mask
        return np.where(np.isin(mask, classes), mask, 0)

    @cached_property
    def binary_mask_any_ich(self) -> np.ndarray:
        return (
            self.multiclass_mask > 0
        ).astype(np.uint8)

    @cached_property
    def binary_masks_by_class(self) -> dict[int, np.ndarray]:
        multiclass = self.multiclass_mask

        return {
            cls.value: (
                multiclass == cls.value
            ).astype(np.uint8)
            for cls in self.foreground_classes
        }

    @property
    def foreground_classes(
        self,
    ) -> list[SegmentationClass]:

        return [
            cls
            for cls in self.class_map
            if cls.value != 0
        ]
    
    def binary_mask_for_class(
        self,
        class_value: int,
    ) -> np.ndarray:

        return (
            self.multiclass_mask == class_value
        ).astype(np.uint8)

    def get_class_name(
        self,
        value: int,
    ) -> str | None:

        for cls in self.class_map:
            if cls.value == value:
                return cls.name

        return None

    def get_class_value(
        self,
        name: str,
    ) -> int | None:

        name = name.lower()

        for cls in self.class_map:
            if cls.name.lower() == name:
                return cls.value

        return None


class RLECodec:

    @staticmethod
    def decode_multiclass(rle_mask: RLEMask) -> np.ndarray:

        total_pixels = rle_mask.total_pixels
        mask = np.empty(total_pixels, dtype=np.uint8)
        
        position = 0
        run_pairs = rle_mask.get_run_pairs()
        
        for class_value, count in run_pairs:
            end_position = position + count
            
            if end_position > total_pixels:
                raise ValueError(
                    f"RLE overflow: position {end_position} exceeds total pixels {total_pixels}"
                )
            
            mask[position:end_position] = class_value
            position = end_position
        
        if position != total_pixels:
            raise ValueError(
                f"RLE underflow: decoded {position} pixels, expected {total_pixels}"
            )
        
        return mask.reshape(rle_mask.shape)


    @staticmethod
    def encode_multiclass(mask: np.ndarray) -> RLEMask:

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.ndim}D")
    
        
        flattened = mask.flatten()
        counts = []
        
        current_class = flattened[0]
        current_count = 1
        
        for pixel in flattened[1:]:
            if pixel == current_class:
                current_count += 1
            else:
                counts.extend([current_class, current_count])
                current_class = pixel
                current_count = 1
        
        counts.extend([current_class, current_count])
        
        return RLEMask(shape=mask.shape, counts=counts)