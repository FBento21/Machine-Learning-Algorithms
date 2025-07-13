import pandas as pd

from decision_trees.id3.id3_classifier import ID3Classifier


class TestID3Classifier:

    INPUT_OUTPUT_ERROR_DFS = {'GENERAL_DFS':
        [
            (
                pd.DataFrame({
                    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain",
                                "Sunny", "Overcast", "Overcast", "Rain"],
                    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild",
                                    "Mild", "Hot", "Mild"],
                    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal",
                                 "Normal", "High", "Normal", "High"],
                    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong",
                             "Strong", "Weak", "Strong"],
                    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
                }),
                pd.DataFrame({
                    "Outlook": ["Sunny", "Rain"],
                    "Temperature": ["Hot", "Mild"],
                    "Humidity": ["High", "High"],
                    "Wind": ["Weak", "Strong"],
                    "Play": ["No", "No"]
                })
            ),
            (
                pd.DataFrame({
                    "Outlook": ["Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain"],
                    "Temperature": ["Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild"],
                    "Humidity": ["High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal"],
                    "Wind": ["Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak"],
                    "Play": ["Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
                }),
                pd.DataFrame({
                    "Outlook": ["Sunny", "Rain"],
                    "Temperature": ["Hot", "Mild"],
                    "Humidity": ["High", "High"],
                    "Wind": ["Weak", "Strong"],
                    "Play": ["No", "No"]
                })
            )
        ],
        
        'BEST_FEATURES_DFS':
        [
            (
                pd.DataFrame({
                    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain",
                                "Sunny", "Overcast", "Overcast", "Rain"],
                    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild",
                                    "Mild", "Hot", "Mild"],
                    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal",
                                 "Normal", "High", "Normal", "High"],
                    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong",
                             "Strong", "Weak", "Strong"],
                    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
                }),
                ('Outlook', None)
            ),
            (
                pd.DataFrame({
                    "Outlook": ["Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain"],
                    "Temperature": ["Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild"],
                    "Humidity": ["High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal"],
                    "Wind": ["Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak"],
                    "Play": ["Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
                }),
                ('Outlook', None)
            )
        ],

        'CONTINUOUS_FEATURES_DFS':
        [
            (
                (
                    ['Temperature'],
                    pd.DataFrame({
                    "Outlook": ["Sunny", "Rain", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain",
                                "Sunny", "Overcast", "Overcast", "Rain"],
                    "Temperature": [30, 35, 3, 23, 15, 14, 14, 21, 10, 22, 23, 20, 28, 19],
                    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal",
                                 "Normal", "High", "Normal", "High"],
                    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong",
                             "Strong", "Weak", "Strong"],
                    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
                })
                ),
                pd.DataFrame({
                    "Outlook": ["Sunny", "Rain"],
                    "Temperature": [32, 24],
                    "Humidity": ["High", "High"],
                    "Wind": ["Weak", "Strong"],
                    "Play": ["No", "No"]
                })
            ),
        ]
    }

    def test_id3_classifier(self):
        test_tree = ID3Classifier()
        for train, test in self.INPUT_OUTPUT_ERROR_DFS['GENERAL_DFS']:
            X_train, y_train = train.drop(['Play'], axis=1), train['Play']
            X_test, y_test = test.drop(['Play'], axis=1), test['Play'].to_list()

            test_tree.fit(X_train, y_train)
            predicted_output = test_tree.predict(X_test)

            assert predicted_output == y_test

    def test_compute_best_feature(self):
        test_tree = ID3Classifier()
        for train, test in self.INPUT_OUTPUT_ERROR_DFS['BEST_FEATURES_DFS']:
            X_train, y_train = train.drop(['Play'], axis=1), train['Play']

            test_output = test_tree._compute_best_feature(X_train, y_train)

            assert test_output == test

    def test_id3_continuous_features(self):
        for (numerical_features, train), test in self.INPUT_OUTPUT_ERROR_DFS['CONTINUOUS_FEATURES_DFS']:
            test_tree = ID3Classifier(numerical_features=numerical_features)

            X_train, y_train = train.drop(['Play'], axis=1), train['Play']
            X_test, y_test = test.drop(['Play'], axis=1), test['Play'].to_list()

            test_tree.fit(X_train, y_train)
            predicted_output = test_tree.predict(X_test)

            assert predicted_output == y_test