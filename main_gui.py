from mpi4py import MPI
import pandas as pd
import sys
import platform
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Importamos nodos
from svm_node import train_svm
from bayes_node import train_nb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CLASE 1: VENTANA DE CONFIGURACIÓN (Inicio) ---
class ConfigWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Configuración de Entrenamiento MPI")
        self.geometry("600x500")

        self.filepath = None
        self.params = {}
        self.start_training = False # Bandera para controlar el flujo

        # 1. Selección de Archivo
        ctk.CTkLabel(self, text="1. Selección de Dataset", font=("Arial", 16, "bold")).pack(pady=10)
        self.btn_file = ctk.CTkButton(self, text="Seleccionar CSV (spam.csv)", command=self.select_file)
        self.btn_file.pack(pady=5)
        self.lbl_file = ctk.CTkLabel(self, text="Ningún archivo seleccionado", text_color="gray")
        self.lbl_file.pack()

        # 2. Hiperparámetros SVM
        ctk.CTkLabel(self, text="2. Parámetros SVM", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        frame_svm = ctk.CTkFrame(self)
        frame_svm.pack(pady=5)
        ctk.CTkLabel(frame_svm, text="Valor C (Regularización):").pack(side="left", padx=10)
        self.entry_c = ctk.CTkEntry(frame_svm, width=80)
        self.entry_c.insert(0, "1.0")
        self.entry_c.pack(side="left", padx=10)

        # 3. Hiperparámetros Naive Bayes
        ctk.CTkLabel(self, text="3. Parámetros Naive Bayes", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        frame_nb = ctk.CTkFrame(self)
        frame_nb.pack(pady=5)
        ctk.CTkLabel(frame_nb, text="Valor Alpha (Suavizado):").pack(side="left", padx=10)
        self.entry_alpha = ctk.CTkEntry(frame_nb, width=80)
        self.entry_alpha.insert(0, "1.0")
        self.entry_alpha.pack(side="left", padx=10)

        # 4. Botón Iniciar
        self.btn_start = ctk.CTkButton(self, text="INICIAR ENTRENAMIENTO DISTRIBUIDO",
                                       command=self.on_start, fg_color="#2fa572", height=50)
        self.btn_start.pack(pady=40, fill="x", padx=40)

    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.filepath = filename
            self.lbl_file.configure(text=filename.split("/")[-1], text_color="white")

    def on_start(self):
        if not self.filepath:
            messagebox.showerror("Error", "Debes seleccionar un archivo CSV primero.")
            return

        try:
            # Guardamos los parámetros
            self.params = {
                "svm_C": float(self.entry_c.get()),
                "nb_alpha": float(self.entry_alpha.get())
            }
            self.start_training = True
            self.destroy() # Cerramos ventana para que el script MPI continúe
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser números (use punto para decimales).")

# --- CLASE 2: VENTANA DE RESULTADOS (Final) ---
class ResultWindow(ctk.CTk):
    def __init__(self, data_n1, data_n2, dist_data):
        super().__init__()
        self.title("Resultados del Clúster")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")

        # Layout principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Encabezado
        header = ctk.CTkFrame(self, height=50)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(header, text="DASHBOARD DE RESULTADOS MPI", font=("Arial", 20, "bold")).pack(pady=10)

        # Área de Scroll para gráficas
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=10)

        # 1. Tarjetas de Resumen
        self.frame_cards = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.frame_cards.pack(fill="x", pady=10)
        self.crear_tarjeta(self.frame_cards, "Rank 1 (Naive Bayes)", data_n1, "#1f538d").pack(side="left", fill="x", expand=True, padx=5)
        self.crear_tarjeta(self.frame_cards, "Rank 2 (SVM)", data_n2, "#2fa572").pack(side="left", fill="x", expand=True, padx=5)

        # 2. Gráficas
        self.frame_graphs = ctk.CTkFrame(self.scroll)
        self.frame_graphs.pack(fill="both", expand=True, pady=10)
        self.mostrar_graficas_avanzadas(data_n1, data_n2, dist_data)

    def crear_tarjeta(self, parent, titulo, datos, color):
        frame = ctk.CTkFrame(parent, fg_color=color)
        ctk.CTkLabel(frame, text=titulo, font=("Arial", 14, "bold"), text_color="white").pack(pady=5)
        ctk.CTkLabel(frame, text=f"{datos['model_name']}", text_color="white").pack()
        ctk.CTkLabel(frame, text=f"Acc: {datos['accuracy']:.2%}", font=("Arial", 20, "bold"), text_color="white").pack()
        ctk.CTkLabel(frame, text=f"Tiempo: {datos['training_time']:.4f}s", text_color="white").pack()
        return frame

    def mostrar_graficas_avanzadas(self, d1, d2, dist_data):
        # Creamos una figura con GridSpec para organizar mejor
        fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
        gs = fig.add_gridspec(2, 3) # 2 filas, 3 columnas

        # A. Distribución del Dataset (Pie Chart)
        ax_pie = fig.add_subplot(gs[0, 0])
        labels = ['Ham', 'Spam']
        ax_pie.pie(dist_data, labels=labels, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], textprops={'color':"w"})
        ax_pie.set_title("Distribución Dataset", color='white')

        # B. Comparativa Accuracy (Barras)
        ax_bar = fig.add_subplot(gs[0, 1])
        modelos = [d1['model_name'].split()[0], d2['model_name'].split()[0]] # Nombres cortos
        accs = [d1['accuracy'], d2['accuracy']]
        ax_bar.bar(modelos, accs, color=['#1f538d', '#2fa572'])
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_title("Accuracy Comparativo", color='white')
        ax_bar.tick_params(colors='white')
        ax_bar.set_facecolor('#2b2b2b')

        # C. Comparativa Tiempos
        ax_time = fig.add_subplot(gs[0, 2])
        tiempos = [d1['training_time'], d2['training_time']]
        ax_time.bar(modelos, tiempos, color='orange')
        ax_time.set_title("Tiempo Entrenamiento (s)", color='white')
        ax_time.tick_params(colors='white')
        ax_time.set_facecolor('#2b2b2b')

        # D. Matriz de Confusión NB
        self.plot_confusion_matrix(fig.add_subplot(gs[1, 0]), d1['confusion_matrix'], "Matriz NB")

        # E. Matriz de Confusión SVM
        self.plot_confusion_matrix(fig.add_subplot(gs[1, 1]), d2['confusion_matrix'], "Matriz SVM")

        # Ajustar espacio
        plt.tight_layout()

        # Renderizar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_graphs)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def plot_confusion_matrix(self, ax, cm, title):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, color='white')
        ax.tick_params(colors='white')

        # Etiquetas
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Ham', 'Spam'], color='white')
        ax.set_yticklabels(['Ham', 'Spam'], color='white')

        # Escribir números en los cuadros
        thresh = np.max(cm) / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i][j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i][j] > thresh else "black")


# --- LOGICA DEL MASTER ---
def master_logic(comm, size):
    # 1. Lanzar GUI de Configuración
    print("[Master] Iniciando GUI de Configuración...")
    config_app = ConfigWindow()
    config_app.mainloop()

    # Si el usuario cerró la ventana sin dar "Iniciar", abortamos
    if not config_app.start_training:
        print("[Master] Operación cancelada por el usuario.")
        comm.Abort()
        return

    # 2. Cargar y Procesar Datos
    params = config_app.params
    filepath = config_app.filepath

    print(f"[Master] Cargando {filepath} con parámetros {params}...")
    try:
        df = pd.read_csv(filepath, encoding="cp1252")
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
        df["label"] = df["label"].map({"ham": 0, "spam": 1})

        # Guardar distribución para la gráfica
        dist_data = df["label"].value_counts().sort_index().tolist() # [num_ham, num_spam]

        X = df["message"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_df=0.95, min_df=2)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

    except Exception as e:
        print(f"Error procesando CSV: {e}")
        comm.Abort()
        return

    # 3. Empaquetar todo para los hijos
    # Incluimos los hiperparámetros en el paquete
    payload = {
        "X_train": X_train_tfidf, "y_train": y_train,
        "X_test": X_test_tfidf, "y_test": y_test,
        "svm_C": params["svm_C"],
        "nb_alpha": params["nb_alpha"]
    }

    print("[Master] Distribuyendo datos y configuración a los nodos...")
    payload = comm.bcast(payload, root=0)

    # 4. Esperar Resultados
    print("[Master] Esperando entrenamiento...")
    res_nb = comm.recv(source=1, tag=11)
    res_svm = comm.recv(source=2, tag=22)
    print("[Master] Resultados recibidos.")

    # 5. Mostrar GUI de Resultados
    res_app = ResultWindow(res_nb, res_svm, dist_data)
    res_app.mainloop()


# --- LOGICA DE LOS HIJOS ---
def slave_logic(comm, rank):
    # 1. Esperar datos (Se quedarán aquí hasta que el Master termine su GUI de config)
    print(f"[Rank {rank}] Esperando configuración del Master...")
    data = comm.bcast(None, root=0)

    # 2. Desempaquetar
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # 3. Entrenar según rango
    if rank == 1:
        # Extraemos parámetro alpha específico
        params = {"alpha": data["nb_alpha"]}
        metrics = train_nb(X_train, y_train, X_test, y_test, params)
        comm.send(metrics, dest=0, tag=11)

    elif rank == 2:
        # Extraemos parámetro C específico
        params = {"C": data["svm_C"]}
        metrics = train_svm(X_train, y_train, X_test, y_test, params)
        comm.send(metrics, dest=0, tag=22)

# --- MAIN ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        master_logic(comm, size)
    else:
        slave_logic(comm, rank)

if __name__ == "__main__":
    main()
