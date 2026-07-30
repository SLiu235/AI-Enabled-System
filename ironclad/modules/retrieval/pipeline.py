import os
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import numpy as np
import faiss

from modules.retrieval.indexing import FaissIndex
from modules.retrieval.search import FaissSearch
from modules.extraction.embedding import Embedding
from modules.extraction.preprocess import Preprocessing

class Pipeline:
    def __init__(self, pretrained='casia-webface', device='cpu', embedding_dim = 512, image_size=160, index_type='brute_force'):
        self.embedding = Embedding(pretrained=pretrained, device=device, embedding_dim=embedding_dim)
        self.preprocessing = Preprocessing(image_size=image_size)
        self.indexing = FaissIndex(index_type=index_type)

        self.faiss_index = None
        self.metadata = []
        self.image_size = image_size
        self.embedding_dim = embedding_dim

    def __encode(self, image):
        image = self.preprocessing.process(image)
        return self.embedding.encode(image)        


    def __precompute(self, gallery_directory):
        embeddings = []
        self.indexing.metadata = []
        valid_extensions = ('.png', '.jpg', '.jpeg')

        if not os.path.exists(gallery_directory):
            raise FileNotFoundError(f"Gallery directory not found: {gallery_directory}")

        for person_name in os.listdir(gallery_directory):
            person_folder = os.path.join(gallery_directory, person_name)
            
            if os.path.isdir(person_folder):  
                for filename in os.listdir(person_folder):
                    if filename.lower().endswith(valid_extensions):
                        try:
                            image_path = os.path.join(person_folder, filename)
                            image = Image.open(image_path).convert('RGB')
                            embedding = self.__encode(image)
                            embeddings.append(np.array(embedding))
                            # Switch to Indexing.metadata if required
                            self.metadata.append({
                                'name': person_name,  
                                'filename': filename
                            })
                        except Exception as e:
                            print(f"Error processing {filename} in {person_name}'s folder: {str(e)}")
            
        if not embeddings:
            raise ValueError(f"No valid images found in {gallery_directory}. ")
        
        embeddings = np.array(embeddings)
        # self.indexing.add_embeddings(new_vector=embeddings)
        self.faiss_index = self.indexing.create_index(vector_dimension=embeddings.shape[1])

        # self.faiss_index  = faiss.IndexFlatIP(embeddings.shape[1]) # Consine
        # self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1]) # Euclid
        if isinstance(self.faiss_index, (faiss.IndexIVF, faiss.IndexPQ, faiss.IndexIVFScalarQuantizer)):
            self.faiss_index.train(embeddings)
        self.faiss_index.add(np.array(embeddings))

    def __save_embeddings(self, save_file):
        if not os.path.exists(f'../storage/catalog/{save_file}'):
            os.makedirs(f'../storage/catalog/{save_file}')
        
        # self.indexing.save(faiss_path=f'../storage/catalog/{save_file}/faiss_index.bin', metadata_path=f'../storage/catalog/{save_file}/metadata.npy')
        faiss.write_index(self.faiss_index, f'../storage/catalog/{save_file}/faiss_index.bin')
        np.save(f'../storage/catalog/{save_file}/metadata.npy', self.metadata)


    def search_gallery(self, probe, k):
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized. ")
        
        self.search = FaissSearch(faiss_index=self.indexing, metric='manhattan', p=1)
        # self.search = FaissSearch(faiss_index=self.indexing, metric='cosine')
        # self.search = FaissSearch(faiss_index=self.indexing, metric='euclidean')
        probe_embedding = self.__encode(probe)
        # results = self.search.search(query_vector=probe_embedding, k=k)
        
        results = []
        distances, indices = self.faiss_index.search(probe_embedding.reshape(1, -1), k)

        for idx in indices[0]:
            meta = self.metadata[idx]
            results.append({
                'name': meta['name'],
                'filename': meta['filename'],
                'embedding': self.faiss_index.reconstruct(int(idx))
            })
        return results

    def process_gallery(self, gallery_directory, save_file='default'):
        self.__precompute(gallery_directory)
        self.__save_embeddings(save_file)

    def load_embeddings(self, load_file='default'):
        if not os.path.exists(f'../storage/catalog/{load_file}/faiss_index.bin') or not os.path.exists(f'../storage/catalog/{load_file}/metadata.npy'):
            raise FileNotFoundError("FAISS index or metadata file not found. ")
        
        # self.indexing.load(faiss_path=f'../storage/catalog/{load_file}/faiss_index.bin', metadata_path=f'../storage/catalog/{load_file}/metadata.npy')
        self.faiss_index = faiss.read_index(f'../storage/catalog/{load_file}/faiss_index.bin')
        self.metadata = np.load(f'../storage/catalog/{load_file}/metadata.npy', allow_pickle=True).tolist()


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.process_gallery('../storage/multi_image_gallery')
    pipeline.load_embeddings()

    probe_image = Image.open('../simclr_resources/probe/Alan_Ball/Alan_Ball_0002.jpg')
    results = pipeline.search_gallery(probe_image, k=5)
    print(results)