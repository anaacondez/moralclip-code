from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, List, Dict
from pathlib import Path


class CLIPImageMultiClassification(nn.Module):
    """Visual Moral Compass: Multi-label moral classifier based on CLIP"""
    def __init__(self, model, num_pairs=5):
        super(CLIPImageMultiClassification, self).__init__()
        self.clip_model = model
        dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        with torch.no_grad():
            visual_features = self.clip_model.get_image_features(pixel_values=dummy_input)

        self.classifiers = nn.ModuleList([
            nn.Linear(visual_features.shape[-1], 3) for _ in range(num_pairs)
        ])
    
    def forward(self, images):
        visual_features = self.clip_model.get_image_features(pixel_values=images)
        outputs = [classifier(visual_features) for classifier in self.classifiers]
        return outputs


class VisualMoralCompass:
    """
    Visual Moral Compass for moral classification of images based on Moral Foundations Theory.
    
    Attributes:
        MORAL_FOUNDATIONS: The five moral foundation pairs used for classification
        LABEL_MAPPING: Mapping from classifier outputs to moral labels
    """
    
    MORAL_FOUNDATIONS = [
        ('Care', 'Harm'),
        ('Fairness', 'Cheating'),
        ('Loyalty', 'Betrayal'),
        ('Respect', 'Subversion'),
        ('Sanctity', 'Degradation')
    ]
    
    LABEL_MAPPING = {
        0: [1, 0],  #First label present
        1: [0, 1],  #Second label present
        2: [2, 2],  #Neither present (neutral)
    }
    
    def __init__(self, hf_model_name: str = None, device: str = None):
        """
        Initialize Visual Moral Compass model.
        
        Args:
            hf_model_name: HuggingFace model name (e.g., "username/visual-moral-compass")
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if hf_model_name:
            self.model, self.processor = self._load_from_hub(hf_model_name)
        else:
            raise ValueError("Must provide hf_model_name")
    
    def _load_from_hub(self, hf_model_name: str):
        """Load model from HuggingFace Hub"""
        from huggingface_hub import hf_hub_download
        
        print(f"Loading Visual Moral Compass from HuggingFace: {hf_model_name}...")
        model_id = "openai/clip-vit-base-patch16"
        processor = CLIPProcessor.from_pretrained(model_id)
        base_model = CLIPModel.from_pretrained(model_id)
        model = CLIPImageMultiClassification(base_model, num_pairs=5).to(self.device)
        
        #Download model weights
        model_file = hf_hub_download(repo_id=hf_model_name, filename="pytorch_model.bin")
        model.load_state_dict(torch.load(model_file, map_location=self.device))
        model.eval()
        print(f"✓ Model loaded successfully from HuggingFace on {self.device}")
        return model, processor
    
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = None):
        """
        Load model from HuggingFace Hub.
        
        Args:
            model_name: HuggingFace model name (e.g., "username/visual-moral-compass")
            device: Device to run model on
            
        Returns:
            VisualMoralCompass instance
        """
        return cls(hf_model_name=model_name, device=device)
    
    def _map_predictions(self, predictions: List[int]) -> Dict[str, int]:
        """Map classifier predictions to moral foundation labels"""
        mapped_labels = {}
        for i, predicted_class in enumerate(predictions):
            first_label, second_label = self.MORAL_FOUNDATIONS[i]
            first_value, second_value = self.LABEL_MAPPING[predicted_class]
            mapped_labels[first_label] = first_value
            mapped_labels[second_label] = second_value
        return mapped_labels
    
    def classify_image(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        return_scores: bool = False,
        return_embeddings: bool = False
    ) -> Dict:
        """
        Classify a single image according to Moral Foundations Theory.
        
        Args:
            image: Image to classify (file path, PIL Image, or numpy array)
            return_scores: Whether to include softmax scores for each foundation
            return_embeddings: Whether to include CLIP embeddings
            
        Returns:
            Dictionary with classification results:
            - 'classifications': Dict mapping each moral foundation to 0/1/2
                0 = not present, 1 = present, 2 = neutral
            - 'scores': (optional) Softmax probabilities for each classifier
            - 'embedding': (optional) Normalized CLIP embedding vector
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path, PIL Image, or numpy array")
        
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(self.device)
        
        #Predictions
        with torch.no_grad():
            outputs = self.model(pixel_values)
            
            predictions = []
            scores = []
            for output in outputs:
                probs = nn.functional.softmax(output, dim=-1)
                predicted_class = torch.argmax(probs[0], dim=-1)
                predictions.append(predicted_class.item())
                scores.append(probs[0].cpu().numpy())
            
            #Classifications
            classifications = self._map_predictions(predictions)
            
            result = {"classifications": classifications}
            
            if return_scores:
                result["scores"] = {
                    foundation: scores[i].tolist()
                    for i, foundation in enumerate([f"{pair[0]}/{pair[1]}" for pair in self.MORAL_FOUNDATIONS])
                }
            
            if return_embeddings:
                embeddings = self.model.clip_model.get_image_features(pixel_values)
                normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
                result["embedding"] = normalized_embeddings[0].cpu().numpy().tolist()
        
        return result
    
    def classify_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 16,
        return_scores: bool = False,
        return_embeddings: bool = False
    ) -> List[Dict]:
        """
        Classify multiple images in batches for efficiency.
        
        Args:
            images: List of images (file paths or PIL Images)
            batch_size: Number of images to process at once
            return_scores: Whether to include softmax scores
            return_embeddings: Whether to include CLIP embeddings
            
        Returns:
            List of dictionaries, one per image, with classification results
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Load batch of images
            batch_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    batch_images.append(Image.open(img).convert('RGB'))
                elif isinstance(img, Image.Image):
                    batch_images.append(img.convert('RGB'))
                else:
                    raise ValueError("Images must be file paths or PIL Images")
            
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values)
                
                if return_embeddings:
                    embeddings = self.model.clip_model.get_image_features(pixel_values)
                    normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
                
                for j in range(len(batch_images)):
                    predictions = []
                    scores = []
                    for output in outputs:
                        probs = nn.functional.softmax(output, dim=-1)
                        predicted_class = torch.argmax(probs[j], dim=-1)
                        predictions.append(predicted_class.item())
                        scores.append(probs[j].cpu().numpy())
                    
                    classifications = self._map_predictions(predictions)
                    
                    result = {"classifications": classifications}
                    
                    if return_scores:
                        result["scores"] = {
                            foundation: scores[idx].tolist()
                            for idx, foundation in enumerate([f"{pair[0]}/{pair[1]}" for pair in self.MORAL_FOUNDATIONS])
                        }
                    
                    if return_embeddings:
                        result["embedding"] = normalized_embeddings[j].cpu().numpy().tolist()
                    
                    all_results.append(result)
        
        return all_results
    
    def get_moral_profile(self, image: Union[str, Path, Image.Image]) -> Dict[str, str]:
        """
        Get a human-readable moral profile of an image.
        
        Args:
            image: Image to analyze
            
        Returns:
            Dictionary with moral dimensions and their interpretations
        """
        result = self.classify_image(image, return_scores=False)
        classifications = result['classifications']
        
        profile = {}
        for pair in self.MORAL_FOUNDATIONS:
            first, second = pair
            first_val = classifications[first]
            second_val = classifications[second]
            
            if first_val == 1:
                profile[f"{first}/{second}"] = f"{first}"
            elif second_val == 1:
                profile[f"{first}/{second}"] = f"{second}"
            else:
                profile[f"{first}/{second}"] = "Neutral"
        
        return profile


def main():
    """Example usage of Visual Moral Compass"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Moral Compass - Moral classification of images")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--hf-model", type=str, help="HuggingFace model name")
    parser.add_argument("--scores", action="store_true", help="Show confidence scores")
    parser.add_argument("--embeddings", action="store_true", help="Include CLIP embeddings")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    if args.hf_model:
        compass = VisualMoralCompass.from_pretrained(args.hf_model, device=args.device)
    else:
        print("Error: Must provide --hf-model")
        return
    
    # Classify image
    print(f"\nAnalyzing image: {args.image}")
    result = compass.classify_image(
        args.image, 
        return_scores=args.scores,
        return_embeddings=args.embeddings
    )
    
    # Display results
    print("\n" + "="*50)
    print("MORAL CLASSIFICATION RESULTS")
    print("="*50)
    
    profile = compass.get_moral_profile(args.image)
    for foundation, interpretation in profile.items():
        print(f"{foundation:25s}: {interpretation}")
    
    if args.scores:
        print("\n" + "-"*50)
        print("CONFIDENCE SCORES")
        print("-"*50)
        for foundation, score in result['scores'].items():
            print(f"{foundation:25s}: {score}")
    
    print("\n")


if __name__ == "__main__":
    main()
