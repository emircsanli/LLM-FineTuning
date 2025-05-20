import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.losses import BinaryCrossentropy
import pickle
import os


# TPU konfigürasyonu
def configure_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        return strategy
    except ValueError:
        print('TPU not found, using CPU/GPU')
        return tf.distribute.get_strategy()


# Veri yükleme ve hazırlama
def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Özellik listeleri - projenizin veri yapısına göre düzenlenmiş
    candidate_features = [
        'candidate_degreeType', 'candidate_department', 'candidate_graduationYear', 'candidate_gpa',
        'candidate_employmentType', 'candidate_preferredWorkType', 'candidate_preferredPosition',
        'candidate_minWorkHours', 'candidate_maxWorkHours', 'candidate_canTravel', 'candidate_expectedSalary',
        'candidate_experienceYears', 'candidate_jobExperience',
        'candidate_reading', 'candidate_writing', 'candidate_speaking', 'candidate_listening',
        'candidate_certificationCount', 'candidate_projectCount'
    ]

    job_features = [
        'job_positionName', 'job_industry',
        'job_minSalary', 'job_maxSalary', 'job_travelRest', 'job_licenseRequired',
        'job_workType', 'job_employmentType', 'job_minWorkHours', 'job_maxWorkHours',
        'job_degreeType', 'job_jobExperience', 'job_experienceYears',
        'job_technicalSkillCount', 'job_socialSkillCount',
        'job_reading', 'job_writing', 'job_speaking', 'job_listening',
        'job_employeeCount', 'job_establishedDate'
    ]

    # Label encoding
    le_dict = {}
    for col in candidate_features + job_features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    # Normalizasyon
    scaler = MinMaxScaler()
    df[candidate_features + job_features] = scaler.fit_transform(df[candidate_features + job_features])

    return df, candidate_features, job_features, le_dict, scaler


# Model eğitimi
def train_model(df, X_cols, label="model"):
    X = df[X_cols]
    y = df['match']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TPU için batch size'ı artır
    BATCH_SIZE = 1024  # TPU için optimize edilmiş batch size

    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    strategy = configure_tpu()

    with strategy.scope():
        model = Sequential([
            LayerNormalization(input_shape=(X_train.shape[1],)),
            Dense(256, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(1e-5, 1e-4)),
            Dropout(0.4),
            Dense(128, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(1e-5, 1e-4)),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(1e-5, 1e-4)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(1e-5, 1e-4)),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

    # Callbacks
    callbacks = [
        LearningRateScheduler(scheduler),
        ModelCheckpoint(
            f'best_{label}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    # TPU için optimize edilmiş eğitim
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# Modelleri eğit ve kaydet
def train_and_save_models(data_path, model_dir="models"):
    # Dizin oluştur
    os.makedirs(model_dir, exist_ok=True)

    # Veriyi hazırla
    df, candidate_features, job_features, le_dict, scaler = prepare_data(data_path)

    # 1. Adaya uygun ilan öneren modeli eğit
    user_to_job_model, user_to_job_history = train_model(df, candidate_features + job_features, label="user_to_job")

    # 2. İlana uygun aday öneren modeli eğit
    job_to_user_model, job_to_user_history = train_model(df, job_features + candidate_features, label="job_to_user")

    # Modelleri kaydet
    user_to_job_model.save(os.path.join(model_dir, "user_to_job_model.h5"))
    job_to_user_model.save(os.path.join(model_dir, "job_to_user_model.h5"))

    # Eğitim geçmişini kaydet
    with open(os.path.join(model_dir, "training_history.pkl"), "wb") as f:
        pickle.dump({
            "user_to_job": user_to_job_history.history,
            "job_to_user": job_to_user_history.history
        }, f)

    # Label encoder ve scaler'ı kaydet
    with open(os.path.join(model_dir, "le_dict.pkl"), "wb") as f:
        pickle.dump(le_dict, f)

    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Özellikleri kaydet
    with open(os.path.join(model_dir, "features.pkl"), "wb") as f:
        pickle.dump({
            "candidate_features": candidate_features,
            "job_features": job_features
        }, f)

    print("Modeller başarıyla eğitildi ve kaydedildi.")
    return candidate_features, job_features


# Kaydedilmiş modelleri yükle
def load_saved_models(model_dir="models"):
    user_to_job_model = load_model(os.path.join(model_dir, "user_to_job_model.h5"))
    job_to_user_model = load_model(os.path.join(model_dir, "job_to_user_model.h5"))

    with open(os.path.join(model_dir, "le_dict.pkl"), "rb") as f:
        le_dict = pickle.load(f)

    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    return user_to_job_model, job_to_user_model, le_dict, scaler, features


# İş önerisi fonksiyonu
def recommend_jobs_for_candidate(candidate_profile, job_pool_df, model, le_dict, scaler, features, k=30):
    candidate_features = features["candidate_features"]
    job_features = features["job_features"]

    recs = []
    for _, job in job_pool_df.iterrows():
        # Aday ve iş profili birleştirme
        combined = {**candidate_profile, **job}
        X_input = pd.DataFrame([combined])

        # Label encoding uygulama
        for col in X_input.columns:
            if col in le_dict and X_input[col].dtype == 'object':
                X_input[col] = le_dict[col].transform(X_input[col].astype(str))

        # Normalizasyon
        X_input_scaled = scaler.transform(X_input[candidate_features + job_features])

        # Tahmin
        score = model.predict(X_input_scaled, verbose=0)[0][0]
        recs.append((score, job))

    # En iyi eşleşmeleri sırala
    top_jobs = sorted(recs, key=lambda x: -x[0])[:k]
    return pd.DataFrame([x[1] for x in top_jobs])


# Aday önerisi fonksiyonu
def recommend_candidates_for_job(job_profile, candidate_pool_df, model, le_dict, scaler, features, k=30):
    candidate_features = features["candidate_features"]
    job_features = features["job_features"]

    recs = []
    for _, cand in candidate_pool_df.iterrows():
        # İş ve aday profili birleştirme
        combined = {**job_profile, **cand}
        X_input = pd.DataFrame([combined])

        # Label encoding uygulama
        for col in X_input.columns:
            if col in le_dict and X_input[col].dtype == 'object':
                X_input[col] = le_dict[col].transform(X_input[col].astype(str))

        # Normalizasyon
        X_input_scaled = scaler.transform(X_input[job_features + candidate_features])

        # Tahmin
        score = model.predict(X_input_scaled, verbose=0)[0][0]
        recs.append((score, cand))

    # En iyi eşleşmeleri sırala
    top_candidates = sorted(recs, key=lambda x: -x[0])[:k]
    return pd.DataFrame([x[1] for x in top_candidates])


# Ana fonksiyon
if __name__ == "__main__":
    # Modelleri eğit ve kaydet (sadece bir kez çalıştır)
    data_path = "merged_all_candidate_job_combinations.csv"
    candidate_features, job_features = train_and_save_models(data_path)