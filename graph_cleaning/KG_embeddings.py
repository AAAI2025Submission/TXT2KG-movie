
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--KG_train_epoch_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--model', type=str, default='TuckER')

args=parser.parse_args()


triplets=TriplesFactory.from_path(f'{args.load_path}/epoch_{args.epoch_num}/triplets.txt')
training_triplets,validating_triplets,testing_triplets=triplets.split([0.8,0.1,0.1],random_state=args.seed)
result = pipeline(
    model=args.model,
    training=training_triplets,
    testing=testing_triplets,
    validation=validating_triplets,
    random_seed=args.seed,
    device='cuda',
    training_kwargs=dict(
        num_epochs=args.KG_train_epoch_num,
    ),
)
# print(result)
# Get the trained model
model = result.model
model.save_state(f'{args.save_path}/epoch_{args.epoch_num}/model.pkl')


