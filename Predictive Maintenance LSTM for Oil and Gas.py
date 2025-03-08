import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# 1. Load Data Pelatihan dan Pelacakan
train_file = 'C:/Users/ASUS/Documents/Pengembangan Diri/LKTI Pertamina/Codingan/Train.csv'
test_file = 'C:/Users/ASUS/Documents/Pengembangan Diri/LKTI Pertamina/Codingan/Test.csv'

train_df = pd.read_csv(train_file, parse_dates=['Time'], index_col='Time')
test_df = pd.read_csv(test_file, parse_dates=['Time'], index_col='Time')

# 2. Ubah Interval Time Series ke 1 Menit
def resample_to_1_min(df):
    df_resampled = df.resample('1T').mean()
    df_resampled = df_resampled.interpolate(method='linear')
    return df_resampled

train_df = resample_to_1_min(train_df)
test_df = resample_to_1_min(test_df)

train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. Preprocess Data
def preprocess_data(df, features):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    return scaled_data, scaler

features = train_df.columns.tolist()
scaled_train, train_scaler = preprocess_data(train_df, features)
scaled_test, test_scaler = preprocess_data(test_df, features)

# 4. Autoencoder untuk Deteksi Anomali
def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(inputs)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder(len(features))
autoencoder.fit(scaled_train, scaled_train, epochs=50, batch_size=32, verbose=1)

reconstructed = autoencoder.predict(scaled_test)

reconstruction_loss = np.mean(np.abs(reconstructed - scaled_test), axis=1)

anomaly_threshold = 0.5

test_df['Anomaly'] = reconstruction_loss > anomaly_threshold

# 5. Membuat Model LSTM
def build_lstm_model(input_shape, num_features):
    model = Sequential([ 
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.3),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

n_input = 30
batch_size = 16
input_shape = (n_input, len(features))

def create_generator(data, n_input, batch_size):
    return TimeseriesGenerator(data, data, length=n_input, batch_size=batch_size)

train_generator = create_generator(scaled_train, n_input, batch_size)
test_generator = create_generator(scaled_test, n_input, batch_size)

lstm_model = build_lstm_model(input_shape, len(features))
lstm_model.fit(train_generator, epochs=100, batch_size=batch_size, verbose=1)

model_path = 'lstm_model.h5'
lstm_model.save(model_path)
print(f"Model berhasil disimpan di {model_path}")

predictions = lstm_model.predict(test_generator)
predictions_rescaled = test_scaler.inverse_transform(predictions)
actual_rescaled = test_scaler.inverse_transform(scaled_test[n_input:])

results_df = pd.DataFrame(actual_rescaled, columns=[f'Actual_{col}' for col in features])
for i, col in enumerate(features):
    results_df[f'Pred_{col}'] = predictions_rescaled[:, i]

# 6. GUI untuk Real-Time Visualization
class RealtimeGUI:
    def __init__(self, root, results_df, parameters):
        self.root = root
        self.results_df = results_df
        self.parameters = parameters
        self.current_parameter_index = 0

        self.x_data = []
        self.y_actual = []
        self.y_pred = []

        # Frame utama
        self.root.title("Real-Time Prediction")
        self.root.geometry("1000x700")

        # Grafik matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Tombol parameter
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Tombol parameter (diperbaiki agar tidak lebar dan cukup banyak tampil)
        for i, param in enumerate(self.parameters):
            button = ttk.Button(self.button_frame, text=str(i), command=lambda idx=i: self.update_parameter(idx), width=4)
            button.pack(side=tk.LEFT, padx=5, pady=5)

        # Tampilkan grafik awal
        self.update_graph()
        self.root.after(60000, self.update_realtime)  # Update setiap 1 menit

    def update_parameter(self, index):
        self.current_parameter_index = index

    def update_graph(self):
        self.ax.clear()

        actual_col = f"Actual_{self.parameters[self.current_parameter_index]}"
        pred_col = f"Pred_{self.parameters[self.current_parameter_index]}"

        # Update plot dengan data baru
        self.ax.plot(self.x_data, self.y_actual, label=f"Actual {self.parameters[self.current_parameter_index]}", color='blue')
        self.ax.plot(self.x_data, self.y_pred, label=f"Predicted {self.parameters[self.current_parameter_index]}", color='red', linestyle='dashed')

        self.ax.set_title(f"Real-Time Prediction: {self.parameters[self.current_parameter_index]}")
        self.ax.set_xlabel("Time (Index)")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.ax.grid()

        self.canvas.draw()

    def update_realtime(self):
        if len(self.x_data) < len(self.results_df):
            frame = len(self.x_data)
            actual_col = f"Actual_{self.parameters[self.current_parameter_index]}"
            pred_col = f"Pred_{self.parameters[self.current_parameter_index]}"

            self.x_data.append(frame)
            self.y_actual.append(self.results_df.iloc[frame][actual_col])
            self.y_pred.append(self.results_df.iloc[frame][pred_col])

            self.update_graph()

        self.root.after(60000, self.update_realtime)  # Ulangi setelah 1 menit

# 7. Simpan Hasil ke File CSV
hasil_file = 'C:/Users/ASUS/Documents/Pengembangan Diri/LKTI Pertamina/Codingan'
output_file = os.path.join(os.path.dirname(hasil_file), "Hasil.csv")

# Menyusun DataFrame untuk hasil
comparison_df = pd.DataFrame()

for col in features:
    actual_col = f"Actual_{col}"
    pred_col = f"Pred_{col}"
    comparison_df[actual_col] = results_df[actual_col]
    comparison_df[pred_col] = results_df[pred_col]

# Simpan DataFrame ke file CSV
comparison_df.to_csv(output_file, index=False)
print(f"Hasil prediksi berhasil disimpan di {output_file}")

# 8. Simpan Grafik Hasil Prediksi
grafik_file = 'C:/Users/ASUS/Documents/Pengembangan Diri/LKTI Pertamina/Hasil Coding/New Result'
graph_path = os.path.dirname(grafik_file)  # Path untuk menyimpan grafik

for col in features:
    actual_col = f"Actual_{col}"
    pred_col = f"Pred_{col}"

    # Ambil 30 data pertama untuk plotting
    actual_values = results_df[actual_col].iloc[:30]
    pred_values = results_df[pred_col].iloc[:30]

    # Membuat plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(30), actual_values, label="Actual", color="blue")
    plt.plot(range(30), pred_values, label="Prediction", color="red", linestyle="dashed")
    plt.title(f"Comparison: {col}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    # Simpan gambar
    graph_file = os.path.join(graph_path, f"{col}_Prediction_Graph.png")
    plt.savefig(graph_file, bbox_inches="tight")
    print(f"Grafik prediksi untuk {col} berhasil disimpan di {graph_file}")

    # Tutup figure untuk menghemat memori
    plt.close()

# Jalankan GUI
if __name__ == "__main__":
    parameter_names = [
        "Pompa Tekanan Inlet", "Pompa Tekanan Outlet", "Pompa Suhu Operasi", "Pompa Kecepatan Putaran",
        "Pipa Tekanan Fluida", "Pipa Suhu Fluida", "Pipa Laju Aliran", "Kompresor Tekanan Inlet",
        "Kompresor Tekanan Outlet", "Kompresor Suhu Operasi", "Generator Output Daya", "Generator Suhu Operasi",
        "Katup Posisi Bukaan", "Katup Tekanan Fluida", "Tangki Level Fluida", "Tangki Suhu Fluida", "Tangki Tekanan"
    ]

    root = tk.Tk()
    app = RealtimeGUI(root, results_df, parameter_names)
    root.mainloop()