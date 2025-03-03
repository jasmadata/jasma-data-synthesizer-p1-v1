from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd

# input dataset
input_data = "./data/titanic.csv"

# location of two output files
mode = "correlated_attribute_mode"
description_file = f"./out/description.json"
synthetic_data = f"./out/sythetic_data.csv"

categorical_attributes = {
    "Name": True,
    "Sex": True,
    "Ticket": True,
    "Cabin": True,
    "Embarked": True,
}
candidate_keys = {"PassengerId": True}
# An attribute is categorical if its domain size is less than this threshold.
threshold_value = 20
# A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
# change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
# Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
epsilon = 1
# The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
degree_of_bayesian_network = 2
# Number of tuples generated in synthetic dataset.
num_tuples_to_generate = 1000

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(
    dataset_file=input_data,
    epsilon=epsilon,
    k=degree_of_bayesian_network,
    attribute_to_is_categorical=categorical_attributes,
    attribute_to_is_candidate_key=candidate_keys,
)
describer.save_dataset_description_to_file(description_file)

display_bayesian_network(describer.bayesian_network)
generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(
    num_tuples_to_generate, description_file
)
generator.save_synthetic_data(synthetic_data)
# Read both datasets using Pandas.
synthetic_df = pd.read_csv(synthetic_data)
# Read attribute description from the dataset description file.
attribute_description = read_json_file(description_file)["attribute_description"]
inspector = ModelInspector(titanic_df, synthetic_df, attribute_description)
inspector.mutual_information_heatmap()
