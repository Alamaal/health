from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        current_batch.append(element)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

class TeamClassifier:
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIG_LIP_MODEL_PATH if 'SIG_LIP_MODEL_PATH' in globals() else SIGLIP_MODEL_PATH).to(device)
        
        if "cuda" in device:
            self.features_model = self.features_model.half()
            
        self.processor = AutoProcessor.from_pretrained(SIG_LIP_MODEL_PATH if 'SIG_LIP_MODEL_PATH' in globals() else SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init=10)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([])
            
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = list(create_batches(crops, self.batch_size))
        data = []

        with torch.inference_mode():
            for batch in tqdm(batches, desc='Turbo Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

                if "cuda" in self.device:
                    inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}

                # التصحيح هنا: نستخدم الموديل مباشرة ثم نأخذ الـ pooler_output
                outputs = self.features_model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)