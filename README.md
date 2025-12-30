# README – Initialisation de l’environnement virtuel (Windows & macOS)

Ce projet utilise un environnement virtuel Python (venv) afin d’isoler les dépendances et éviter les conflits de versions.

## Prérequis
- Python 3.9 ou plus
- pip installé
- Terminal (macOS/Linux) ou PowerShell / CMD (Windows)

Vérification :
python --version
pip --version

## 1. Création de l’environnement virtuel

Se placer dans le dossier du projet.

macOS / Linux :
python3 -m venv venv

Windows (PowerShell ou CMD) :
python -m venv venv

Un dossier venv/ est alors créé à la racine du projet.

## 2. Activation de l’environnement virtuel

macOS / Linux :
source venv/bin/activate

Windows – PowerShell :
venv\Scripts\Activate.ps1

Si l’activation fonctionne, le terminal affiche (venv) au début de la ligne.

## Problème courant sous Windows (droits PowerShell)

Erreur possible :
"l’exécution de scripts est désactivée sur ce système"

Solution recommandée (sécurisée – une seule fois) :
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Confirmer avec O puis Entrée, puis relancer :
venv\Scripts\Activate.ps1

## Alternatives si tu ne veux pas modifier la policy

Option CMD :
venv\Scripts\activate.bat

Option PowerShell temporaire :
powershell -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1

## 3. Installation des dépendances

Une fois l’environnement activé :
pip install --upgrade pip
pip install -r requirements.txt

## 4. Lancer l’application
streamlit run app.py

## 5. Quitter l’environnement virtuel
deactivate

## Structure recommandée
AMF_QCM/
├── app.py
├── AMF_values.xlsx
├── requirements.txt
├── README.md
└── venv/

Remarques :
- Le dossier venv/ ne doit pas être versionné
- Toujours activer l’environnement virtuel avant de lancer le projet
