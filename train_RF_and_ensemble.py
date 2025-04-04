import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import chromadb
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent
from agents.ensemble_agent import EnsembleAgent

# Constants
QUESTION = "How much does this cost to the nearest dollar?\n\n"
DB = "products_vectorstore"

def setup_environment():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

def load_test_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def load_chroma_data(db_path):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection('products')
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    prices = [metadata['price'] for metadata in result['metadatas']]
    return vectors, prices, collection

def train_random_forest(vectors, prices, model_path):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    rf_model.fit(vectors, prices)
    joblib.dump(rf_model, model_path)
    return rf_model

def load_random_forest(model_path):
    return joblib.load(model_path)

def train_ensemble_model(X, y, model_path):
    lr = LinearRegression()
    lr.fit(X, y)
    joblib.dump(lr, model_path)
    return lr

def description(item):
    return item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]

def collect_predictions(test_data, specialist, frontier, random_forest):
    specialists, frontiers, random_forests, prices = [], [], [], []
    for item in tqdm(test_data):
        text = description(item)
        specialists.append(specialist.price(text))
        frontiers.append(frontier.price(text))
        random_forests.append(random_forest.price(text))
        prices.append(item.price)
    return specialists, frontiers, random_forests, prices

def prepare_ensemble_features(specialists, frontiers, random_forests):
    mins = [min(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]
    maxes = [max(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]
    return pd.DataFrame({
        'Specialist': specialists,
        'Frontier': frontiers,
        'RandomForest': random_forests,
        'Min': mins,
        'Max': maxes,
    })

def main():
    setup_environment()
    test_data = load_test_data('test.pkl')
    vectors, prices, collection = load_chroma_data(DB)

    # Train or load Random Forest model
    rf_model_path = 'random_forest_model.pkl'
    try:
        rf_model = load_random_forest(rf_model_path)
    except FileNotFoundError:
        rf_model = train_random_forest(vectors, prices, rf_model_path)

    # Initialize agents
    specialist = SpecialistAgent()
    frontier = FrontierAgent(collection)
    random_forest = RandomForestAgent()

    # Collect predictions
    specialists, frontiers, random_forests, prices = collect_predictions(test_data[1000:1250], specialist, frontier, random_forest)

    # Prepare ensemble features and train ensemble model
    X = prepare_ensemble_features(specialists, frontiers, random_forests)
    y = pd.Series(prices)
    ensemble_model_path = 'ensemble_model.pkl'
    train_ensemble_model(X, y, ensemble_model_path)

    # # Ensemble agent testing
    # ensemble = EnsembleAgent(collection)
    # product = "Quadcast HyperX condenser mic for high quality audio for podcasting"
    # print(ensemble.price(product))

if __name__ == "__main__":
    main()
