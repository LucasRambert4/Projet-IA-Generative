# Voice-Controlled Streamlit Dashboard

Dashboard interactif contrôlé à la voix, développé avec **Streamlit**, **Whisper**, **Silero VAD**, **Ollama** et **Kokoro TTS**.

Ce projet permet de naviguer dans un dashboard, poser des questions sur les données affichées, obtenir une réponse locale avec un modèle LLM, puis entendre cette réponse via une synthèse vocale locale.

---

## Objectif du projet

L’objectif est de créer un prototype d’accessibilité permettant à un utilisateur d’interagir avec un dashboard sans utiliser la souris ni le clavier.

Le système permet notamment de :

* contrôler un dashboard Streamlit avec la voix ;
* détecter une vraie parole grâce à un système VAD ;
* utiliser un wake word comme `Ok Jack` ou `Jack` ;
* transcrire les commandes vocales avec Whisper ;
* poser des questions à un assistant local via Ollama ;
* obtenir une réponse vocale avec Kokoro TTS ;
* éviter que le micro réécoute la voix générée par l’assistant.

---

## Fonctionnalités principales

### Commandes vocales

Le dashboard peut être contrôlé avec des commandes comme :

```text
Ok Jack va à la page ventes
Ok Jack affiche les ventes par région
Ok Jack descends
Ok Jack monte
Ok Jack retourne en haut
Ok Jack tout en bas
```

### Mode chatbot vocal

Le chatbot peut être ouvert vocalement :

```text
Ok Jack chatbot
```

Une fois le chatbot activé, les questions suivantes sont envoyées directement à l’assistant :

```text
Quelle est ma vente moyenne ?
Quelle région vend le plus ?
Quelle est la pire région ?
Combien de clients avons-nous ?
```

Pour fermer le chatbot :

```text
ferme chatbot
chatbot désactivé
```

---

## Architecture générale

```text
Microphone
   ↓
Silero VAD
   ↓
Ring buffer audio
   ↓
Wake word detection
   ↓
Whisper STT
   ↓
Command parser ou chatbot
   ↓
Ollama LLM
   ↓
Kokoro TTS
   ↓
Lecture audio locale Python
```

---

## Technologies utilisées

| Technologie    | Rôle                                  |
| -------------- | ------------------------------------- |
| Streamlit      | Interface du dashboard                |
| faster-whisper | Transcription vocale locale           |
| Silero VAD     | Détection de vraie parole             |
| Ollama         | Exécution locale du modèle LLM        |
| llama3.2       | Modèle de langage utilisé avec Ollama |
| Kokoro TTS     | Synthèse vocale locale                |
| sounddevice    | Capture micro et lecture audio locale |
| pandas         | Données du dashboard                  |

---

## Structure du projet

```text
stt_dashboard_voice/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── command_bus.py
│   ├── command_parser.py
│   ├── continuous_listener_live.py
│   ├── dashboard_controller.py
│   ├── kokoro_tts_engine.py
│   ├── ollama_client.py
│   ├── stt_engine.py
│   ├── vad_engine.py
│   └── wake_word.py
│
├── runtime/
│   ├── latest_command.json
│   ├── listener_status.json
│   ├── listener_control.json
│   ├── listener_phrase.wav
│   └── kokoro_response.wav
│
├── .streamlit/
│   └── config.toml
│
└── README.md
```

Le dossier `runtime/` contient les fichiers temporaires utilisés pour la communication entre le dashboard, le listener vocal, le STT et le TTS. Il ne doit pas être versionné.

---

## Prérequis

Avant de lancer le projet, il faut installer :

* Python 3.10 ou 3.11 ;
* Ollama ;
* un microphone fonctionnel ;
* un environnement virtuel Python recommandé.

Ollama doit être disponible dans le terminal avec la commande :

```powershell
ollama --version
```

---

## Installation

### 1. Cloner le projet

```powershell
git clone https://github.com/LucasRambert4/Projet-IA-Generative.git
cd Projet-IA-Generative
```

### 2. Créer un environnement virtuel

```powershell
python -m venv .venv
```

### 3. Activer l’environnement virtuel

```powershell
.\.venv\Scripts\activate
```

### 4. Installer les dépendances

```powershell
pip install -r requirements.txt
```

Si le fichier `requirements.txt` n’est pas encore présent, installer les dépendances principales :

```powershell
pip install streamlit streamlit-autorefresh pandas numpy sounddevice soundfile faster-whisper silero-vad requests kokoro
```

### 5. Installer le modèle Ollama

```powershell
ollama pull llama3.2
```

---

## Configuration Streamlit recommandée

Créer le fichier suivant :

```text
.streamlit/config.toml
```

Avec le contenu :

```toml
[server]
fileWatcherType = "none"
runOnSave = false

[browser]
gatherUsageStats = false
```

Cette configuration évite certains ralentissements et messages liés au watcher Streamlit.

---

## Configuration du micro

Dans le fichier :

```text
src/continuous_listener_live.py
```

la variable suivante définit le périphérique micro utilisé :

```python
FIXED_INPUT_DEVICE = 1
```

Selon votre ordinateur, il peut être nécessaire de changer cette valeur.

Pour afficher la liste des périphériques audio disponibles, vous pouvez exécuter :

```python
import sounddevice as sd
print(sd.query_devices())
```

Puis remplacer `FIXED_INPUT_DEVICE` par l’index correspondant au bon microphone.

---

## Lancement propre du projet

Depuis PowerShell :

```powershell
cd C:\chemin\vers\le\projet
.\.venv\Scripts\activate
```

Nettoyer les anciens processus :

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*continuous_listener_live.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Nettoyer les fichiers temporaires :

```powershell
del runtime\live_listener.pid -ErrorAction SilentlyContinue
del runtime\listener_status.json -ErrorAction SilentlyContinue
del runtime\listener_control.json -ErrorAction SilentlyContinue
del runtime\latest_command.json -ErrorAction SilentlyContinue
del runtime\listener_phrase.wav -ErrorAction SilentlyContinue
del runtime\kokoro_response.wav -ErrorAction SilentlyContinue
```

Lancer le dashboard :

```powershell
streamlit run dashboard/app.py
```

Le dashboard démarre automatiquement le listener vocal.

---

## Utilisation

### Ouvrir le chatbot

```text
Ok Jack chatbot
```

### Poser une question

```text
Quelle est ma vente moyenne ?
```

### Poser une autre question sans répéter le wake word

```text
Quelle région vend le plus ?
```

### Fermer le chatbot

```text
ferme chatbot
```

---

## Fonctionnement du STT

Le système utilise `faster-whisper` pour convertir la voix en texte.

Pour améliorer la fiabilité, le projet utilise :

* un prompt de contexte contenant le vocabulaire du dashboard ;
* un modèle Whisper local ;
* une détection VAD avant transcription ;
* un buffer audio pour éviter de couper le début de phrase.

Cela permet de mieux reconnaître des commandes comme :

```text
Ok Jack quelle est ma vente moyenne ?
Ok Jack affiche les ventes par région
Ok Jack ouvre le chatbot
```

---

## Fonctionnement du VAD

Le projet utilise **Silero VAD** pour détecter une vraie voix humaine.

Avant, le micro s’activait uniquement avec un seuil de volume. Cette approche posait plusieurs problèmes :

* activation trop tardive ;
* perte du début de phrase ;
* activation avec du bruit ;
* coupure avant la fin de la phrase.

Avec Silero VAD, le système détecte une activité vocale réelle avant de lancer l’enregistrement complet.

---

## Fonctionnement du chatbot

Le chatbot utilise Ollama avec le modèle :

```text
llama3.2
```

Le dashboard injecte un contexte à Ollama contenant les données disponibles :

* ventes par mois ;
* clients par mois ;
* ventes par région ;
* clients par région ;
* ventes totales ;
* vente moyenne ;
* meilleure région ;
* meilleur mois ;
* pire mois.

Cela évite que l’assistant réponde hors contexte ou parle du site officiel Streamlit.

---

## Fonctionnement du TTS

Le projet utilise **Kokoro TTS** pour générer la réponse vocale.

La voix est générée dans un fichier :

```text
runtime/kokoro_response.wav
```

Puis le fichier est lu directement par Python avec `sounddevice`.

Cette méthode évite les restrictions d’autoplay du navigateur. Le son ne dépend donc plus de Streamlit, Chrome ou Opera.

---

## Pause automatique du micro pendant la voix Kokoro

Lorsque Kokoro parle, le micro est temporairement mis en pause.

Cela évite que le système entende sa propre voix et relance une commande accidentellement.

Le dashboard écrit une instruction dans :

```text
runtime/listener_control.json
```

Le listener vocal lit ce fichier et se met en pause jusqu’à la fin estimée de la lecture audio.

---

## Exemples de commandes

### Navigation

```text
Ok Jack va à la page résumé
Ok Jack va à la page ventes
Ok Jack va à la page clients
Ok Jack va à la page régions
```

### Graphiques

```text
Ok Jack affiche les ventes par région
Ok Jack affiche les clients par région
Ok Jack affiche les ventes par mois
```

### Scroll

```text
Ok Jack descends
Ok Jack monte
Ok Jack tout en haut
Ok Jack tout en bas
```

### Questions chatbot

```text
Ok Jack chatbot
Quelle est ma vente moyenne ?
Quelle région vend le plus ?
Quel est le meilleur mois ?
Quelle est la pire région ?
Combien de clients avons-nous ?
ferme chatbot
```

---

## Dépannage

### Le micro ne s’active pas

Vérifier l’index du micro dans :

```python
FIXED_INPUT_DEVICE = 1
```

Puis relancer le dashboard.

---

### Le listener ne redémarre pas

Arrêter les anciens processus :

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*continuous_listener_live.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Puis supprimer :

```powershell
del runtime\live_listener.pid -ErrorAction SilentlyContinue
```

---

### Ollama ne répond pas

Vérifier que le modèle est installé :

```powershell
ollama pull llama3.2
```

Lancer Ollama manuellement :

```powershell
ollama serve
```

Puis relancer Streamlit.

---

### Kokoro ne parle pas

Vérifier que le fichier audio est bien généré :

```text
runtime/kokoro_response.wav
```

Vérifier aussi que `sounddevice` et `soundfile` sont installés :

```powershell
pip install sounddevice soundfile
```

---

### Messages `torchvision` dans le terminal

Certains messages peuvent apparaître à cause du watcher Streamlit et de la librairie `transformers`.

La configuration suivante réduit fortement ces messages :

```toml
[server]
fileWatcherType = "none"
runOnSave = false
```

---

### Warning `use_container_width`

Streamlit peut afficher :

```text
Please replace use_container_width with width
```

Cela n’empêche pas le projet de fonctionner. Pour corriger, remplacer progressivement :

```python
use_container_width=True
```

par :

```python
width="stretch"
```

---

## Limites actuelles

Ce projet est un prototype local. Il n’a pas encore le même niveau de robustesse qu’un assistant vocal commercial comme Siri, Alexa ou Google Assistant.

Les limites principales sont :

* wake word basé sur transcription, pas encore sur un vrai modèle dédié ;
* sensibilité dépendante du micro ;
* latence possible au premier chargement de Whisper, Ollama ou Kokoro ;
* données de démonstration intégrées directement dans le dashboard.

---

## Améliorations possibles

Les améliorations futures possibles :

* ajouter un vrai moteur de wake word comme openWakeWord ou Porcupine ;
* créer un wake word personnalisé ;
* connecter le dashboard à une vraie base de données ;
* ajouter plus de pages et de métriques ;
* améliorer la détection des intentions ;
* créer un vrai composant frontend pour le chatbot ;
* ajouter un mode multi-langue ;
* packager le projet avec Docker.

---

## Résumé technique

Le projet fonctionne avec plusieurs processus qui communiquent via des fichiers JSON dans `runtime/`.

```text
continuous_listener_live.py
    écoute le micro
    détecte la voix
    transcrit avec Whisper
    écrit la commande dans latest_command.json

dashboard/app.py
    lit latest_command.json
    modifie l’état Streamlit
    appelle Ollama si nécessaire
    lance Kokoro TTS
    met le listener en pause pendant la lecture audio

kokoro_tts_engine.py
    génère le fichier WAV
    lit la réponse vocalement via Python

command_bus.py
    centralise les échanges entre listener et dashboard
```

Cette architecture permet de garder le dashboard réactif tout en séparant clairement :

* l’écoute micro ;
* la transcription ;
* le parsing ;
* l’analyse LLM ;
* la synthèse vocale ;
* l’interface utilisateur.

---

## Auteurs

Projet réalisé dans le cadre d’un travail autour de l’IA générative, de l’accessibilité et du contrôle vocal d’un dashboard.

Développé par :

```text
Lucas Rambert
```
