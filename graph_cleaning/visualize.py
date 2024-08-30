from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from pykeen.models import TuckER
from pykeen.triples import TriplesFactory
import torch

from sklearn.cluster import DBSCAN
import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--load_path', type=str, default="results/")
parser.add_argument('--epoch_num', type=int, default=0)

args=parser.parse_args()


triplets=TriplesFactory.from_path(f'{args.load_path}/epoch_{args.epoch_num}/triplets.txt')
training_triplets,validating_triplets,testing_triplets=triplets.split([0.96,0.02,0.02])
model = TuckER(triples_factory=triplets,random_seed=1234)
model.load_state_dict(torch.load(f'{args.load_path}/epoch_{args.epoch_num}/epoch_{args.epoch_num}/model.pkl'))

entity_dict=triplets.entity_to_id
relation_dict=triplets.relation_to_id

entity_embedding = model.relation_representations[0]
relation_embedding= model.relation_representations[0]
# Assuming `numpy_array` is your high-dimensional data
numpy_array = list(entity_embedding.parameters())[0].detach().numpy()

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=0)
numpy_array_2d = tsne.fit_transform(numpy_array)

# Plot the nodes
plt.figure(figsize=(12, 10))
plt.scatter(x=numpy_array_2d[:,0],y=numpy_array_2d[:,1], color='blue')
plt.show()