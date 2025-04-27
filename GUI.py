import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LoanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Approval Predictor")

        self.model = None
        self.scaler = None
        self.columns = None

        # Load Dataset Button
        self.load_button = tk.Button(root, text="Load Loan Dataset", command=self.load_data)
        self.load_button.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=5)

        # Fields to input
        self.entries = {}
        fields = [("Age", 30), ("Income (₹)", 50000), ("Experience", 5),
                  ("Education (Graduate/Not Graduate)", "Graduate"), ("Assets (₹)", 100000),
                  ("Liabilities (₹)", 20000)]

        for label, default in fields:
            frame = tk.Frame(root)
            frame.pack()
            tk.Label(frame, text=label, width=30, anchor='w').pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, str(default))
            entry.pack(side="left")
            self.entries[label] = entry

        # Predict Button
        self.predict_button = tk.Button(root, text="Predict", command=self.make_prediction, state="disabled")
        self.predict_button.pack(pady=20)

    def load_data(self):
        # Load dataset
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "C:\\Users\\lenovo\\OneDrive\\Documents\\loan_approval_dataset.csv")])
        if not file_path:
            return

        try:
            # Load and preprocess data
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()

            # Process and clean data
            if 'loan_id' in data.columns:
                data.drop(columns=['loan_id'], inplace=True)

            data['Assets'] = data['residential_assets_value'] + data['commercial_assets_value'] + \
                             data['luxury_assets_value'] + data['bank_asset_value']
            data.drop(columns=['residential_assets_value', 'commercial_assets_value',
                               'luxury_assets_value', 'bank_asset_value'], inplace=True)

            data['education'] = data['education'].apply(lambda x: x.strip())
            data['income_annum'] = pd.to_numeric(data['income_annum'], errors='coerce')
            data = data.dropna()

            # Split data into features (X) and target (y)
            X = data.drop(columns=['loan_status'])
            y = data['loan_status']
            X = pd.get_dummies(X)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, model.predict(X_test))

            # Store trained model and scaler for predictions
            self.model = model
            self.scaler = scaler
            self.columns = X.columns

            # Update status
            self.status_label.config(text=f"Model trained successfully! Accuracy: {accuracy:.2%}")
            self.predict_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")

    def make_prediction(self):
        # Ensure model is loaded
        if not self.model:
            messagebox.showwarning("Model Not Ready", "Please load a dataset first.")
            return

        try:
            # Prepare the input data
            input_data = {
                'age': int(self.entries["Age"].get()),
                'income_annum': float(self.entries["Income (₹)"].get()),
                'experience': int(self.entries["Experience"].get()),
                'education': self.entries["Education (Graduate/Not Graduate)"].get().strip(),
                'loan_amount': 500000,  # Sample loan amount
                'Assets': float(self.entries["Assets (₹)"].get()),
                'liabilities': float(self.entries["Liabilities (₹)"].get()),
                'cibil_score': 750  # Sample CIBIL score
            }

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)

            # Match columns to model's features
            for col in self.columns:
                if col not in input_df:
                    input_df[col] = 0
            input_df = input_df[self.columns]

            # Scale the input data
            input_scaled = self.scaler.transform(input_df)

            # Predict the outcome
            result = self.model.predict(input_scaled)[0]
            confidence = np.max(self.model.predict_proba(input_scaled)[0])
            outcome = "APPROVED ✅" if result == 1 else "REJECTED ❌"

            # Display result
            messagebox.showinfo("Prediction", f"Loan Status: {outcome}\nConfidence: {confidence:.2%}")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApp(root)
    root.mainloop()
