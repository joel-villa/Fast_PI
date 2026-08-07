"""For reading and writing to json files"""

import json

def load_json(save_path: str) -> dict:
    """Load the data in the json file as a dictionary

    Args:
        save_path (str): path to the json file

    Returns:
        dict: dictionary of the json file
    """
    with open(save_path, 'r') as file:
        data = json.load(file)

    return data

def save_json(data: dict, save_path:str) -> None:
    """Add this data to the dictionary that is currently stored at save_path

    Args:
        data (dict):  dictionary to be added to the json file
        save_path (str): path to the json file
    """
    # Saving to a file
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=2)