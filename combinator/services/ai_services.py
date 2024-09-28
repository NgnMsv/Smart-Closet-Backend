#LogisticRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from combinator.models import Combination
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import logging


# LR
# class AIServices:
#     def __init__(self, closet_user):
#         self.model = None
#         self.closet_user = closet_user
#         self.model_file = f'model_{closet_user.id}.pkl'
    
#     def hex_to_rgb(self, hex_color):
#         """Convert hex color to RGB tuple."""
#         hex_color = hex_color.lstrip('#')
#         return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
#     def fetch_data(self):
#         """Fetch data from Django models and combine them into a DataFrame."""
#         valid_combinations = list(Combination.objects.filter(shirt__closet__user=self.closet_user, 
#                                                              label__isnull=False).values('shirt__color', 'shirt__usage_1', 'shirt__usage_2', 
#                                                                                          'pants__color', 'pants__usage_1', 'pants__usage_2', 
#                                                                                          'footwear__color', 'footwear__usage_1', 'footwear__usage_2', 
#                                                                                          'label'))
        
#         # Combine all the data into a single DataFrame
#         df = pd.DataFrame(valid_combinations)
        
#         return df

#     def preprocess_data(self, df):
#         """Preprocess data by converting categorical variables to numeric."""
#         df[['shirt_R', 'shirt_G', 'shirt_B']] = df['shirt__color'].apply(self.hex_to_rgb).apply(pd.Series)
#         df[['pants_R', 'pants_G', 'pants_B']] = df['pants__color'].apply(self.hex_to_rgb).apply(pd.Series)
#         df[['footwear_R', 'footwear_G', 'footwear_B']] = df['footwear__color'].apply(self.hex_to_rgb).apply(pd.Series)
#         df.drop(['shirt__color', 'pants__color', 'footwear__color'], axis=1, inplace=True)
#         df = pd.get_dummies(df, columns=['shirt__usage_1', 'shirt__usage_2', 
#                                          'pants__usage_1', 'pants__usage_2', 
#                                          'footwear__usage_1', 'footwear__usage_2'])
        
#         expected_columns = [
#             'shirt_R', 'shirt_G', 'shirt_B', 
#             'pants_R', 'pants_G', 'pants_B', 
#             'footwear_R', 'footwear_G', 'footwear_B',
#             'shirt__usage_1_f', 'shirt__usage_1_c', 'shirt__usage_1_s', 'shirt__usage_1_g', 
#             'shirt__usage_2_f', 'shirt__usage_2_c', 'shirt__usage_2_s', 'shirt__usage_2_g',
#             'pants__usage_1_f', 'pants__usage_1_c', 'pants__usage_1_s', 'pants__usage_1_g',
#             'pants__usage_2_f', 'pants__usage_2_c', 'pants__usage_2_s', 'pants__usage_2_g',
#             'footwear__usage_1_f', 'footwear__usage_1_c', 'footwear__usage_1_s', 'footwear__usage_1_g',
#             'footwear__usage_2_f', 'footwear__usage_2_c', 'footwear__usage_2_s', 'footwear__usage_2_g',
#             'label',
#         ]

#         # Add missing columns that are not present in the dataset
#         for col in expected_columns:
#             if col not in df:
#                 df[col] = 0  # Add missing columns and fill with 0

#         # Ensure the columns are ordered correctly to match the expected structure
#         df = df[expected_columns]
#         print(df)

#         # Separate features (X) and target (y)

#         numerical_cols = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
#                         'footwear_R', 'footwear_G', 'footwear_B']
        
#         # Scale only the numerical columns
#         scaler = StandardScaler()
#         df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
#         joblib.dump(scaler, f'scaler_{self.closet_user.id}.pkl')

#         X = df.drop(columns=['label'])
#         y = df['label']

#         return X, y

#     def train_model(self):
#         """Train the logistic regression model, save it, and output the evaluation metrics."""
#         df = self.fetch_data()
#         X, y = self.preprocess_data(df)
        
#         # Split data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Initialize and train the logistic regression model
#         self.model = LogisticRegression(max_iter=1000)
#         self.model.fit(X_train, y_train)
        
#         # Save the trained model
#         joblib.dump(self.model, self.model_file)
        
#         # Evaluate the model
#         y_pred = self.model.predict(X_test)
        
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         report = classification_report(y_test, y_pred)
        
#         print("Accuracy:", accuracy)
#         print("F1 Score:", f1)
#         print("Classification Report:\n", report)

#     def load_model(self):
#         """Load the saved model from disk."""
#         self.model = joblib.load(self.model_file)

#     def preprocess_input(self, combination):
#         """Preprocess input data for prediction."""
#         input_data = {
#             'shirt__color': combination.shirt.color,
#             'shirt__usage_1': combination.shirt.usage_1,
#             'shirt__usage_2': combination.shirt.usage_2,
#             'pants__color': combination.pants.color,
#             'pants__usage_1': combination.pants.usage_1,
#             'pants__usage_2': combination.pants.usage_2,
#             'footwear__color': combination.footwear.color,
#             'footwear__usage_1': combination.footwear.usage_1,
#             'footwear__usage_2': combination.footwear.usage_2,
#         }
        
#         # Convert the hex color to RGB
#         input_data['shirt_R'], input_data['shirt_G'], input_data['shirt_B'] = self.hex_to_rgb(input_data.pop('shirt__color'))
#         input_data['pants_R'], input_data['pants_G'], input_data['pants_B'] = self.hex_to_rgb(input_data.pop('pants__color'))
#         input_data['footwear_R'], input_data['footwear_G'], input_data['footwear_B'] = self.hex_to_rgb(input_data.pop('footwear__color'))

#         # Convert the input_data dictionary into a DataFrame
#         input_df = pd.DataFrame([input_data])

#         # Encode categorical variables (usage fields) as they were during training
#         input_df = pd.get_dummies(input_df, columns=['shirt__usage_1', 'shirt__usage_2', 
#                                                      'pants__usage_1', 'pants__usage_2', 
#                                                      'footwear__usage_1', 'footwear__usage_2'])

#         # Recreate model columns used during training to match the format
#         model_columns = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
#                          'footwear_R', 'footwear_G', 'footwear_B', 
#                          'shirt__usage_1_f', 'shirt__usage_1_c', 'shirt__usage_1_s', 'shirt__usage_1_g', 
#                          'shirt__usage_2_f', 'shirt__usage_2_c', 'shirt__usage_2_s', 'shirt__usage_2_g',
#                          'pants__usage_1_f', 'pants__usage_1_c', 'pants__usage_1_s', 'pants__usage_1_g',
#                          'pants__usage_2_f', 'pants__usage_2_c', 'pants__usage_2_s', 'pants__usage_2_g',
#                          'footwear__usage_1_f', 'footwear__usage_1_c', 'footwear__usage_1_s', 'footwear__usage_1_g',
#                          'footwear__usage_2_f', 'footwear__usage_2_c', 'footwear__usage_2_s', 'footwear__usage_2_g', ]

#         # Ensure all model columns are present in input data
#         for col in model_columns:
#             if col not in input_df.columns:
#                 input_df[col] = False  # Add missing columns with default value 0

#         # Reorder columns to match the training data structure
#         input_df = input_df[model_columns]

#         # Standardize the input features using StandardScaler (if used during training)
#         # Separate the columns that need scaling (RGB values)
#         numerical_cols = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
#                         'footwear_R', 'footwear_G', 'footwear_B']
        
#         # Scale only the numerical columns
#         scaler = joblib.load(f'scaler_{self.closet_user.id}.pkl')
#         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

#         return input_df

#     def predict_item(self, combination):
#         """Make predictions using the loaded model."""
#         if self.model is None:
#             self.load_model()  # Load model if not already loaded
        
#         print(combination)
#         input_data = self.preprocess_input(combination)
#         print(input_data.values)
#         prediction = self.model.predict_proba(input_data)
#         return prediction[0][1]


# # Decision Trees
class AIServices:
    def __init__(self, closet_user):
        self.model = None
        self.closet_user = closet_user
        self.model_file = f'model_{closet_user.id}.pkl'
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def fetch_data(self):
        """Fetch data from Django models and combine them into a DataFrame."""
        valid_combinations = list(Combination.objects.filter(shirt__closet__user=self.closet_user, 
                                                             label__isnull=False).values('shirt__color', 'shirt__usage_1', 'shirt__usage_2', 
                                                                                         'pants__color', 'pants__usage_1', 'pants__usage_2', 
                                                                                         'footwear__color', 'footwear__usage_1', 'footwear__usage_2', 
                                                                                         'label'))
        
        # Combine all the data into a single DataFrame
        df = pd.DataFrame(valid_combinations)
        
        return df

    def preprocess_data(self, df):
        df[['shirt_R', 'shirt_G', 'shirt_B']] = df['shirt__color'].apply(self.hex_to_rgb).apply(pd.Series)
        df[['pants_R', 'pants_G', 'pants_B']] = df['pants__color'].apply(self.hex_to_rgb).apply(pd.Series)
        df[['footwear_R', 'footwear_G', 'footwear_B']] = df['footwear__color'].apply(self.hex_to_rgb).apply(pd.Series)
        df.drop(['shirt__color', 'pants__color', 'footwear__color'], axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['shirt__usage_1', 'shirt__usage_2', 
                                         'pants__usage_1', 'pants__usage_2', 
                                         'footwear__usage_1', 'footwear__usage_2'])
        
        expected_columns = [
            'shirt_R', 'shirt_G', 'shirt_B', 
            'pants_R', 'pants_G', 'pants_B', 
            'footwear_R', 'footwear_G', 'footwear_B',
            'shirt__usage_1_f', 'shirt__usage_1_c', 'shirt__usage_1_s', 'shirt__usage_1_g', 
            'shirt__usage_2_f', 'shirt__usage_2_c', 'shirt__usage_2_s', 'shirt__usage_2_g',
            'pants__usage_1_f', 'pants__usage_1_c', 'pants__usage_1_s', 'pants__usage_1_g',
            'pants__usage_2_f', 'pants__usage_2_c', 'pants__usage_2_s', 'pants__usage_2_g',
            'footwear__usage_1_f', 'footwear__usage_1_c', 'footwear__usage_1_s', 'footwear__usage_1_g',
            'footwear__usage_2_f', 'footwear__usage_2_c', 'footwear__usage_2_s', 'footwear__usage_2_g',
            'label',
        ]

        for col in expected_columns:
            if col not in df:
                df[col] = 0 

        df = df[expected_columns]

        numerical_cols = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
                        'footwear_R', 'footwear_G', 'footwear_B']
        
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        joblib.dump(scaler, f'scaler_{self.closet_user.id}.pkl')

        X = df.drop(columns=['label'])
        y = df['label']

        return X, y

    def train_model(self):
        df = self.fetch_data()
        X, y = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = DecisionTreeClassifier(random_state=42)  
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, self.model_file)
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Classification Report:\n", report)
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"F1: {f1}")

    def load_model(self):
        """Load the saved model from disk."""
        self.model = joblib.load(self.model_file)

    def preprocess_input(self, combination):
        """Preprocess input data for prediction."""
        input_data = {
            'shirt__color': combination.shirt.color,
            'shirt__usage_1': combination.shirt.usage_1,
            'shirt__usage_2': combination.shirt.usage_2,
            'pants__color': combination.pants.color,
            'pants__usage_1': combination.pants.usage_1,
            'pants__usage_2': combination.pants.usage_2,
            'footwear__color': combination.footwear.color,
            'footwear__usage_1': combination.footwear.usage_1,
            'footwear__usage_2': combination.footwear.usage_2,
        }
        
        # Convert the hex color to RGB
        input_data['shirt_R'], input_data['shirt_G'], input_data['shirt_B'] = self.hex_to_rgb(input_data.pop('shirt__color'))
        input_data['pants_R'], input_data['pants_G'], input_data['pants_B'] = self.hex_to_rgb(input_data.pop('pants__color'))
        input_data['footwear_R'], input_data['footwear_G'], input_data['footwear_B'] = self.hex_to_rgb(input_data.pop('footwear__color'))

        # Convert the input_data dictionary into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables (usage fields) as they were during training
        input_df = pd.get_dummies(input_df, columns=['shirt__usage_1', 'shirt__usage_2', 
                                                     'pants__usage_1', 'pants__usage_2', 
                                                     'footwear__usage_1', 'footwear__usage_2'])

        # Recreate model columns used during training to match the format
        model_columns = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
                         'footwear_R', 'footwear_G', 'footwear_B', 
                         'shirt__usage_1_f', 'shirt__usage_1_c', 'shirt__usage_1_s', 'shirt__usage_1_g', 
                         'shirt__usage_2_f', 'shirt__usage_2_c', 'shirt__usage_2_s', 'shirt__usage_2_g',
                         'pants__usage_1_f', 'pants__usage_1_c', 'pants__usage_1_s', 'pants__usage_1_g',
                         'pants__usage_2_f', 'pants__usage_2_c', 'pants__usage_2_s', 'pants__usage_2_g',
                         'footwear__usage_1_f', 'footwear__usage_1_c', 'footwear__usage_1_s', 'footwear__usage_1_g',
                         'footwear__usage_2_f', 'footwear__usage_2_c', 'footwear__usage_2_s', 'footwear__usage_2_g', ]

        # Ensure all model columns are present in input data
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = False  # Add missing columns with default value 0

        # Reorder columns to match the training data structure
        input_df = input_df[model_columns]

        # Standardize the input features using StandardScaler (if used during training)
        # Separate the columns that need scaling (RGB values)
        numerical_cols = ['shirt_R', 'shirt_G', 'shirt_B', 'pants_R', 'pants_G', 'pants_B', 
                        'footwear_R', 'footwear_G', 'footwear_B']
        
        # Scale only the numerical columns
        scaler = joblib.load(f'scaler_{self.closet_user.id}.pkl')
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        return input_df

    def predict_item(self, combination):
        """Make predictions using the loaded model."""
        if self.model is None:
            self.load_model()  # Load model if not already loaded
        
        input_data = self.preprocess_input(combination)
        prediction = self.model.predict_proba(input_data)
        return prediction[0][1]