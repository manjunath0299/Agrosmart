# AgroSmart – Smart Agriculture AI System

AgroSmart is an AI-assisted smart agriculture system for plant disease detection, disease-severity estimation, treatment recommendation, and MQTT-based spray-control simulation.

## Components
- Disease detection: EfficientNetV2-B0 domain-adapted model
- Severity estimation: EfficientNet-B0 U-Net trained on human annotations
- Backend: FastAPI
- Frontend: React + Vite
- MQTT: Mosquitto
- Hardware simulation: Python simulated ESP32
- Treatment engine: prototype treatment database

## 1. Install on Windows

Install:
1. Python 3.12.x
2. Node.js LTS
3. Mosquitto MQTT
4. NVIDIA driver if an NVIDIA GPU is available (recommended)

Git is optional for a ZIP. Git + Git LFS are needed when cloning from GitHub.

## 2. Extract the project

Example:
```text
E:\smart_agriculture\smart_agriculture
```

Do NOT copy an existing `.venv`. Create a new one.

## 3. Python setup

Open PowerShell in the project root:

```powershell
cd E:\smart_agriculture\smart_agriculture
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check PyTorch:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Check timm:

```powershell
python -c "import timm; print('timm:', timm.__version__)"
```

If an NVIDIA GPU is available, use the appropriate official PyTorch CUDA build for that machine if necessary.

## 4. Start MQTT

Keep a terminal running:

```powershell
mosquitto -v
```

If Mosquitto is not in PATH:

```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

Broker:
```text
localhost:1883
```

## 5. Start FastAPI

New terminal:

```powershell
cd E:\smart_agriculture\smart_agriculture
.venv\Scripts\activate
python -m uvicorn backend.main:app --reload
```

Backend:
```text
http://127.0.0.1:8000
```

Health:
```text
http://127.0.0.1:8000/health
```

## 6. Start simulated ESP32

New terminal:

```powershell
cd E:\smart_agriculture\smart_agriculture
.venv\Scripts\activate
python hardware_sim\simulated_esp32.py
```

MQTT topics:
```text
Command: agrosmart/devices/ESP32-001/command
Status:  agrosmart/devices/ESP32-001/status
```

## 7. Start React frontend

New terminal:

```powershell
cd E:\smart_agriculture\smart_agriculture\frontend
npm install
npm run dev
```

Open:
```text
http://localhost:5173
```

## 8. Normal runtime

Run four terminals:

Terminal 1:
```powershell
mosquitto -v
```

Terminal 2:
```powershell
cd E:\smart_agriculture\smart_agriculture
.venv\Scripts\activate
python -m uvicorn backend.main:app --reload
```

Terminal 3:
```powershell
cd E:\smart_agriculture\smart_agriculture
.venv\Scripts\activate
python hardware_sim\simulated_esp32.py
```

Terminal 4:
```powershell
cd E:\smart_agriculture\smart_agriculture\frontend
npm run dev
```

Then open `http://localhost:5173`.

## 9. Production model files

The configured models are:

```text
ai/disease/models/domain_adaptation/best_domain_adapted_model.pth
ai/severity/results/human_gt_unet/best_human_gt_severity_unet.pth
ai/severity/models/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt
```

For a GitHub clone:

```powershell
git lfs install
git lfs pull
```

Do not move these files without updating the corresponding configuration.

## 10. Environment variables

Copy `.env.example` to `.env` and add required secrets, for example:

```text
TELEGRAM_BOT_TOKEN=your_token_here
```

Never commit real secrets.

## 11. Datasets

PlantDoc and PlantVillage are intentionally NOT included in the ZIP/repository.

They are needed for training/evaluation workflows, not normal inference using the trained models.

## 12. Troubleshooting

Python not found:
- Install Python 3.12.x and add it to PATH.
- Reopen PowerShell.

Uvicorn not found:
```powershell
python -m uvicorn backend.main:app --reload
```

Torch missing:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

timm missing:
```powershell
pip install timm
```

Frontend dependencies:
```powershell
cd frontend
npm install
npm run dev
```

CUDA check:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 13. Safety

The treatment database is a prototype/demo controller configuration. `amount_ml` values are controller volumes and are NOT agricultural pesticide label doses or field application rates.

For real-world spraying, use locally registered product labels and qualified agronomist/extension guidance. Physical spraying must remain disabled until hardware, flow calibration, treatment rules, safety interlocks, and approval are validated.

## Quick setup

```powershell
cd E:\smart_agriculture\smart_agriculture
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then start Mosquitto, FastAPI, the ESP32 simulator, and the React frontend as described above.
