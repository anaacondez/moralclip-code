"""
MoralCLIP model implementations and training utilities.

This module contains the main MoralCLIP trainer class that extends CLIP with moral supervision,
along with data collators and utility functions for training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .loss import compute_moral_loss


@dataclass
class CLIPDataCollator:
    """
    Data collator for CLIP model that handles batching of images, text, and moral labels.
    
    This collator stacks individual samples into batches suitable for CLIP training
    while preserving moral label information.
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dictionaries, each containing:
                - 'input_ids': tokenized text input
                - 'attention_mask': attention mask for text
                - 'pixel_values': processed image tensor
                - 'moral_labels': set or list of moral labels
                
        Returns:
            Dict containing batched inputs ready for model forward pass
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        pixel_values = torch.stack([example['pixel_values'] for example in batch])

        #moral labels (these remain as list of sets/lists)
        moral_labels = [example["moral_labels"] for example in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'moral_labels': moral_labels,
            'return_loss': True,
        }


class CLIPTrainer(Trainer):
    """
    Custom trainer for CLIP model
    """
    def __init__(self, moral_lambda: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moral_lambda = moral_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        moral_labels = inputs.pop("moral_labels")

        outputs = model(**inputs, return_dict=True)
        clip_loss = outputs.loss

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        moral_loss = compute_moral_loss(image_embeds, 
                                        text_embeds,
                                        moral_labels)
        
        total_loss = clip_loss + self.moral_lambda * moral_loss

        return (total_loss, outputs) if return_outputs else total_loss


class MoralCLIP:
    """
    High-level MoralCLIP interface for easy model loading and inference.
    
    This class provides a simple interface for loading trained MoralCLIP models
    and performing moral-aware embedding computation and retrieval.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model identifier
        device: Device to load model on ('cpu', 'cuda', etc.)
    """
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode_image(self, images: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode images into moral-aware embeddings.
        
        Args:
            images: Single image path/PIL Image or list of images
            
        Returns:
            Normalized image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        image_features = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(image_features, p=2, dim=1)
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text into moral-aware embeddings.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Normalized text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(text_features, p=2, dim=1)
    
    @torch.no_grad()
    def compute_similarity(
        self, 
        images: Union[str, List[str]], 
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Compute moral-aware similarity between images and texts.
        
        Args:
            images: Image(s) to encode
            texts: Text(s) to encode
            
        Returns:
            Similarity matrix of shape (num_images, num_texts)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        return image_features @ text_features.t()
    
    def save_pretrained(self, save_directory: str):
        """Save the model and processor to a directory."""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        device: Optional[str] = None
    ) -> "MoralCLIP":
        """
        Load a pre-trained MoralCLIP model.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model identifier
            device: Device to load model on
            
        Returns:
            MoralCLIP instance
        """
        return cls(model_name_or_path=model_name_or_path, device=device)


def create_moral_trainer(
    model: CLIPModel,
    train_dataset,
    eval_dataset,
    data_collator: CLIPDataCollator,
    training_args: TrainingArguments,
    moral_lambda: float = 0.1,
    **kwargs
) -> MoralCLIPTrainer:
    """
    Factory function to create a MoralCLIP trainer with proper configuration.
    
    Args:
        model: Pre-trained CLIP model to fine-tune
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset  
        data_collator: Data collator for batching
        training_args: Hugging Face training arguments
        moral_lambda: Weight for moral loss component
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured MoralCLIPTrainer instance
    """
    return MoralCLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        moral_lambda=moral_lambda,
        **kwargs
    )
