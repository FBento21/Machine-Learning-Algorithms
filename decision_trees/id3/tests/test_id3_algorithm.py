import pandas as pd

from id3_classifier import ID3Classifier

test_tree = ID3Classifier()

def create_dataset():
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    data_test = {
        "Outlook": ["Sunny", "Rain"],
        "Temperature": ["Hot", "Mild"],
        "Humidity": ["High", "High"], # "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": ["Weak", "Strong"], #, "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "Play": ["No", "Yes"]
    }

    return pd.DataFrame(data), pd.DataFrame(data_test)



def test_compute_best_feature():
    df_train, df_test = create_dataset()
    X_train, y_train = df_train.drop(['Play'], axis=1), df_train['Play']

    test_output = test_tree._compute_best_feature(X_train, y_train)
    expected_output = 'Outlook'

    assert test_output == expected_output
