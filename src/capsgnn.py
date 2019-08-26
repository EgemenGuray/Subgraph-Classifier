import glob
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from utils import create_numeric_mapping
from layers import ListModule, PrimaryCapsuleLayer, Attention, SecondaryCapsuleLayer, margin_loss

class CapsGNN(torch.nn.Module):
    """
    gets args: Arguments object.
    gets number_of_features: Number of vertex features.
    gets number_of_targets: Number of classes.
    """
    def __init__(self, args, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()
        
        
    """
    Creating GCN layers.
    """
    def _setup_base_layers(self):
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for layer in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv( self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)

    """
    Creating primary capsules.
    """
    def _setup_primary_capsules(self):
        
        self.first_capsule = PrimaryCapsuleLayer(in_units = self.args.gcn_filters, in_channels = self.args.gcn_layers, num_units = self.args.gcn_layers, capsule_dimensions = self.args.capsule_dimensions)
    
    """
    Creating attention layer.
    """
    def _setup_attention(self):
        self.attention = Attention(self.args.gcn_layers* self.args.capsule_dimensions, self.args.inner_attention_dimension)

    """
    Creating graph capsules.
    """
    def _setup_graph_capsules(self):
        self.graph_capsule = SecondaryCapsuleLayer(self.args.gcn_layers, self.args.capsule_dimensions, self.args.number_of_capsules, self.args.capsule_dimensions)

    """
    Creating class capsules.
    """
    def _setup_class_capsule(self):
        self.class_capsule =  SecondaryCapsuleLayer(self.args.capsule_dimensions,self.args.number_of_capsules, self.number_of_targets, self.args.capsule_dimensions)

    """
    Creating histogram reconstruction layers.
    """
    def _setup_reconstruction_layers(self):
        self.reconstruction_layer_1 = torch.nn.Linear(self.number_of_targets*self.args.capsule_dimensions, int((self.number_of_features * 2) / 3))
        self.reconstruction_layer_2 = torch.nn.Linear(int((self.number_of_features * 2) / 3), int((self.number_of_features * 3) / 2))
        self.reconstruction_layer_3 = torch.nn.Linear(int((self.number_of_features * 3) / 2), self.number_of_features)

    """
    Creating layers of model.
    1. GCN layers.
    2. Primary capsules.
    3. Attention
    4. Graph capsules.
    5. Class capsules.
    6. Reconstruction layers.
    """
    def _setup_layers(self):
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()

    """
    Calculating the reconstruction loss of the model.
    gets capsule_input: Output of class capsule.
    gets features: Feature matrix.
    returns reconstrcution_loss: Loss of reconstruction.
    """
    def calculate_reconstruction_loss(self, capsule_input, features):
        v_mag = torch.sqrt((capsule_input**2).sum(dim=1))
        _, v_max_index = v_mag.max(dim=0)
        v_max_index = v_max_index.data

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
        capsule_masked[v_max_index,:] = capsule_input[v_max_index,:]
        capsule_masked = capsule_masked.view(1, -1)

        feature_counts = features.sum(dim=0)
        feature_counts = feature_counts/feature_counts.sum()

        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(capsule_masked))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.softmax(self.reconstruction_layer_3(reconstruction_output),dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)

        reconstruction_loss = torch.sum((features-reconstruction_output)**2)
        
        return reconstruction_loss
        
    """
    Forward propagation pass.
    gets data: Dictionary of tensors with features and edges.
    returns class_capsule_output: Class capsule outputs.
    """
    def forward(self, data):
        features = data["features"]
        edges = data["edges"]
        hidden_representations = []
        
        for layer in self.base_layers:
            features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters,-1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1,self.args.gcn_layers* self.args.capsule_dimensions)
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers, self.args.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions, self.args.number_of_capsules ) 
        class_capsule_output = self.class_capsule(reshaped_graph_capsule_output)
        class_capsule_output =  class_capsule_output.view(-1, self.number_of_targets*self.args.capsule_dimensions )
        class_capsule_output = torch.mean(class_capsule_output,dim=0).view(1,self.number_of_targets,self.args.capsule_dimensions)
        reconstruction_loss = self.calculate_reconstruction_loss(class_capsule_output.view(self.number_of_targets,self.args.capsule_dimensions), data["features"])
        return class_capsule_output, reconstruction_loss
        
"""
CapsGNN training and scoring.
"""
class CapsGNNTrainer(object):
    
    """
    gets args: Arguments object.
    """
    def __init__(self,args):
        self.args = args
        self.setup_model()

    """
    Enumerating the features and targets in order to setup weights later.
    """
    def enumerate_unique_labels_and_targets(self):
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)
    
        graph_paths = self.train_graph_paths + self.test_graph_paths

        targets = set()
        features = set()
        for path in tqdm(graph_paths):
            data = json.load(open(path))
            targets = targets.union(set([data["target"]]))
            features = features.union(set(data["labels"]))

        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = len(self.feature_map)
        self.number_of_targets = len(self.target_map)
    
    """
    Enumerating labels and initializing a CapsGNN.
    """
    def setup_model(self):
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets)

    """
    Batching the graphs for training.
    """
    def create_batches(self):
        self.batches = [self.train_graph_paths[i:i + self.args.batch_size] for i in range(0,len(self.train_graph_paths), self.args.batch_size)]

    """
    Creating a data dictionary.
    gets target: Target vector.
    gets edges: Edge list tensor.
    gets features: Feature tensor.
    """
    def create_data_dictionary(self, target, edges, features):
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward
    
    """
    Target createn based on data dicionary.
    gets Data dictionary.
    returns Target vector.
    """
    def create_target(self, data):
       return  torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(self.number_of_targets)])
    
    """
    Create an edge matrix.
    gets Data dictionary.
    returns Edge matrix.
    """
    def create_edges(self,data):
        edges = [[edge[0],edge[1]] for edge in data["edges"]] + [[edge[1],edge[0]] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    """
    Creates feature matrix
    
    gets Data dictionary
    
    returns matrix of features
    
    """
    def create_features(self,data):
       
        features = np.zeros((len(data["labels"]), self.number_of_features))
        
        node_indices = [node for node in range(len(data["labels"]))]
        #print(self.feature_map)
        #print(data["labels"])
        
        feature_indices = [self.feature_map[label] for label in data["labels"].keys()] 
        features[node_indices,feature_indices] = 1.0
        features = torch.FloatTensor(features)
        return features

    """
    
    Creates tensors and a data dictionary with Torch tensors
    
    gets path to the json data
    
    returns to_pass_forward which is the data dictionary
    
    """
    def create_input_data(self, path):
        
        data = json.load(open(path))
        target = self.create_target(data)
        edges = self.create_edges(data)
        features = self.create_features(data)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    """
    Training a model on the training set.
    """
    def fit(self):
        
        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        for epoch in tqdm(range(self.args.epochs), desc = "Epochs: ", leave = True):
            random.shuffle(self.train_graph_paths)
            self.create_batches()
            losses = 0       
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    data = self.create_input_data(path)
                    prediction, reconstruction_loss = self.model(data)
                    loss = margin_loss(prediction, data["target"], self.args.lambd)+self.args.theta*reconstruction_loss
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                average_loss = losses/(step + 1)
                self.steps.set_description("CapsGNN (Loss=%g)" % round(average_loss,4))

    """
    Predicts all the test set calculates the accrucay
    """
    def score(self):
        
        print("\n\nScoring.\n")
        self.model.eval()
        self.predictions = []
        self.hits = []
        for path in tqdm(self.test_graph_paths):
            data = self.create_input_data(path)
            prediction, reconstruction_loss = self.model(data)
            prediction_mag = torch.sqrt((prediction**2).sum(dim=2))
            _, prediction_max_index = prediction_mag.max(dim=1)
            prediction = prediction_max_index.data.view(-1).item()
            self.predictions.append(prediction)
            self.hits.append(data["target"][prediction]==1.0)

        print("\nAccuracy: " + str(round(np.mean(self.hits),4)))
        return round(np.mean(self.hits),4)
        
    """
    Predicts the given json file in path
    """
    def predict(self,path):
        self.model.eval()
        self.predictions = []
        self.hits = []
    
        data = self.create_input_data(path)
        prediction, reconstruction_loss = self.model(data)
        prediction_mag = torch.sqrt((prediction**2).sum(dim=2))
        _, prediction_max_index = prediction_mag.max(dim=1)
        prediction = prediction_max_index.data.view(-1).item()
        return [prediction, reconstruction_loss.item()]
        
    """
    Saves model to given path if no path given saves into saved_models folder 
    in given name if no game given saves with the timestamp
    """
    def save_model(self, path=None):
        if path == None:
            import os
            cwd = os.getcwd()[:-4]
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            path = cwd + '/saved_models/' + timestr + '.pth'
        elif '/' not in path:
            import os
            cwd = os.getcwd()[:-4]
            import time
            name = path
            path = cwd + '/saved_models/' + name + '.pth'
        
        torch.save(self.model, path)
        print('Model has been succesfully saved to:', path)
        
    """
    Load models from the saved_models folder
    """        
    def load_model(self, path):
        import os
        cwd = os.getcwd()[:-4]
        #import time
        PATH = cwd + '/saved_models/' + path
        self.model = torch.load(PATH)
        self.model.eval()
        print('Model has been succesfully loaded')
        
    """
    Saves the test set predictions
    """        
    def save_predictions(self):
        
        identifiers = [path.split("/")[-1].strip(".json") for path in self.test_graph_paths]
        out = pd.DataFrame()
        out["id"] = identifiers
        out["predictions"] = self.predictions
        out.to_csv(self.args.prediction_path, index = None)
