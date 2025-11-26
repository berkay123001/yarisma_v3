#!/bin/bash

COLAB_HOST="wear-remain-ant-cuts.trycloudflare.com"
PASS="123456" # Colab'da belirlediğin şifre

echo "🚀 Dosyalar Colab'a atılıyor..."
# ÖNCE train_optuna.py'nin OLDUĞUNDAN EMİN OL!
sshpass -p $PASS scp -o StrictHostKeyChecking=no Processed/clean_train.csv Processed/clean_test.csv train_optuna.py root@$COLAB_HOST:/content/

echo "🔥 Model eğitiliyor..."
sshpass -p $PASS ssh -o StrictHostKeyChecking=no root@$COLAB_HOST "cd /content/ && pip install optuna lightgbm scikit-learn pandas && python3 train_optuna.py"

echo "📥 Sonuçlar çekiliyor..."
sshpass -p $PASS scp -o StrictHostKeyChecking=no root@$COLAB_HOST:/content/submission_optuna.csv ./submission_optuna.csv
mkdir -p Models
sshpass -p $PASS scp -o StrictHostKeyChecking=no root@$COLAB_HOST:/content/Models/best_params.json ./Models/

echo "✅ TAMAMLANDI!"