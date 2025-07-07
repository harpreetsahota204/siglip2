import logging
import os
from packaging.version import Version
import warnings
import math
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SigLIP2Config(fout.TorchImageModelConfig):
    """
    This config class extends TorchImageModelConfig to provide specific parameters
    needed for the used for text-image similarity search.
    
    Args:
        model_path (str): Path to the model's weights on disk or HuggingFace model ID.

        text_prompt (str): Optional baseline text prompt to use for classification.
            Defaults to "".

    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        super().__init__(d)
        
        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="")
        
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")

 
class SigLIP2(fout.TorchImageModel, fom.PromptMixin):
    """
    This model leverages a vision-language model to create embeddings for
    both images and text in a shared vector space, enabling text-image similarity search.
    
    The model can:
    1. Embed images into a vector space
    2. Embed text queries into the same vector space
    3. Calculate similarity between images and text
    4. Support zero-shot classification by comparing image embeddings to class name embeddings
    
    It extends TorchImageModel for image processing capabilities and PromptMixin to
    enable text embedding capabilities.
    """
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A Config instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Store config parameters as instance variables for easier access
        self._text_features = None  # Cached text features for classification

        # Storage for the last computed embeddings (needed for FiftyOne API)
        self._last_computed_embeddings = None

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.
        
        Returns:
            bool: Always True for this model as embedding generation is supported
        """
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts.
        
        Returns:
            bool: Always True for this model as text embedding is supported
        """
        return True

    def _load_model(self, config):
        """Load the model and processor from disk or HuggingFace.
        
        This method initializes both the processor (for tokenization and image
        preprocessing) and the model itself, configuring them for inference.

        Args:
            config: Config instance containing model parameters

        Returns:
            The loaded PyTorch model ready for inference
        """

        # Load the model and processor from HuggingFace
        model_path = config.model_path

        model_kwargs = {
            "device_map":self.device,
            }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Check if Flash Attention 2 is available
            if is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Check if device supports bfloat16 (Ampere or newer GPUs have capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
    
        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True
            )
        
        self.model = AutoModel.from_pretrained(
            model_path,
            **model_kwargs
            )

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        return self.model

    def _get_text_features(self):
        """Get or compute text features for the model's classification.
        
        This method caches the result for efficiency in repeated calls.
        
        Returns:
            numpy array: Text features as a numpy array for classification
        """
        # Check if text features are already computed and cached
        if self._text_features is None:
            prompt = self.config.text_prompt
            # Compute and cache the text features
            self._text_features = self._embed_prompts([prompt])
        
        # Return the cached features
        return self._text_features
    
    def _embed_prompts(self, prompts):
        """Embed text prompts for similarity search.
        
        Follows the approach used in the native implementation of the model,
        using a dummy image as required by the multimodal architecture.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Embeddings for the prompts as numpy arrays
        """
        # Process text inputs
        text_inputs = self.processor.tokenizer(
            prompts, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        
        # Normalize features
        normalized = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Return as CPU numpy array for FiftyOne compatibility
        return normalized.detach().cpu().numpy()

    def embed_prompt(self, prompt):
        """Embed a single text prompt.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: Embedding for the prompt
        """
        # Format prompt with the template (done inside _embed_prompts)
        # Embed the single prompt by calling _embed_prompts with a list
        embeddings = self._embed_prompts([prompt])
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_prompts(self, prompts):
        """Embed multiple text prompts.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Embeddings for the prompts
        """
        # Directly call _embed_prompts which handles batch processing
        return self._embed_prompts(prompts)

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)
            
        Returns:
            numpy array: Embeddings for the images
        """

        # Process images
        image_inputs = self.processor(
            images=imgs, 
            return_tensors="pt").to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
        
        # Normalize features
        normalized = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Cache the embeddings for get_embeddings() method
        self._last_computed_embeddings = normalized
        
        # Return as CPU numpy array for FiftyOne compatibility
        return normalized.detach().cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Implementation of TorchEmbeddingsMixin.embed() method.
        
        Args:
            img: PIL image or PyTorch tensor to embed
            
        Returns:
            numpy array: Embedding for the image
        """
        # Convert single image to a list for batch processing
        if isinstance(img, torch.Tensor):
            imgs = [img]
        else:
            imgs = [img]
        
        # Embed the single image using the batch method
        embeddings = self.embed_images(imgs)
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Implementation of TorchEmbeddingsMixin.embed_all() method.
        
        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)
            
        Returns:
            numpy array: Embeddings for the images
        """
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.
        
        Returns:
            numpy array: The last computed embeddings
            
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
            
        # Return the stored embeddings as a CPU numpy array
        return self._last_computed_embeddings.detach().cpu().numpy()

    def _get_class_logits(self, text_features, image_features):
        """Calculate scaled similarity scores between text and image features.
        
        Args:
            text_features: Text embeddings (numpy array or tensor)
            image_features: Image embeddings (numpy array or tensor)
            
        Returns:
            tuple: (logits_per_image, logits_per_text) scaled similarity matrices
        """
        with torch.no_grad():
            # Convert numpy arrays to torch tensors and move to device
            if isinstance(text_features, np.ndarray):
                text_features = torch.from_numpy(text_features).to(self.device)
            if isinstance(image_features, np.ndarray):
                image_features = torch.from_numpy(image_features).to(self.device)
            
            # Ensure correct dtype
            text_features = text_features.float()
            image_features = image_features.float()
            
            # Normalize features (following CLIP approach)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Get the logit scale from the model
            logit_scale = self.model.logit_scale.exp()
            
            # Compute scaled dot product similarity (using @ like CLIP)
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.
        
        Used for zero-shot classification by comparing image embeddings
        to text embeddings of class names.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            numpy array: Raw logits (or processed by output_processor if available)
        """
        # Get image embeddings (returns numpy array)
        image_embeddings = self.embed_images(imgs)
        
        # Get text embeddings for classes (returns numpy array)
        text_features = self._get_text_features()
        
        # Calculate similarity between images and text (returns torch tensors)
        logits_per_image, logits_per_text = self._get_class_logits(text_features, image_embeddings)
        
        # If you have an output processor (like CLIP), use it
        if hasattr(self, '_output_processor') and self._output_processor is not None:
            # Get frame size for output processor
            if isinstance(imgs[0], torch.Tensor):
                height, width = imgs[0].size()[-2:]
            elif hasattr(imgs[0], 'size'):  # PIL Image
                width, height = imgs[0].size
            else:
                height, width = imgs[0].shape[:2]  # numpy array
            
            frame_size = (width, height)
            
            if self.has_logits:
                self._output_processor.store_logits = self.store_logits
            
            return self._output_processor(
                logits_per_image, 
                frame_size, 
                confidence_thresh=self.config.confidence_thresh
            )
        
        # Return raw logits as CPU numpy array
        return logits_per_image.detach().cpu().numpy()