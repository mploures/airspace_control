# 🛩️ airspace_control

**airspace_control** é um pacote ROS (Noetic) para simulação e controle de espaço aéreo urbano em nível Very-Low-Level (VLL).  
O sistema integra geração automática de mapas e mundos Stage, agentes VANT autônomos e ferramentas para criação e análise de grafos logísticos.

---

## 📁 Estrutura do Projeto

```

airspace_control/
├── graph/                  # scripts e artefatos de geração de grafo/bitmap
│   ├── mapafinal.png
│   ├── processar_mapa.py
│   └── selecionar_cluster.py
│
├── launch/                 # arquivos de lançamento ROS
│   ├── airspace_all.launch
│   └── airspace_stage.launch
│
├── scripts/                # scripts Python usados em runtime
│   ├── control_panel.py
│   ├── create_map_and_launch_stage.sh
│   ├── gen_stage_world.py
│   └── uav_agent.py
│
├── srv/                    # serviços ROS customizados
│   ├── GetBattery.srv
│   └── GotoXY.srv
│
├── ultrades_lib/           # biblioteca de suporte (modelagem discreta)
│   ├── **init**.py
│   └── core/
│       ├── AutomatonNode.py
│       ├── **init**.py
│       └── utils.py
│
├── worlds/                 # mundos Stage gerados automaticamente
│   ├── airspace.world
│   ├── grafo_fundo_branco.png
│   ├── grafo_fundo_branco_muro.png
│   └── ...
│
├── CMakeLists.txt
├── package.xml
└── setup.py

```

---

## ⚙️ Funcionalidades

✅ Geração automática de **mapas e mundos Stage** a partir de descrições gráficas  
✅ Identificação de cruzamentos e construções via **processamento de imagem (OpenCV)**  
✅ Construção de **grafos logísticos** com entidades:
- **VANTPORT** – bases de decolagem e pouso  
- **ESTACAO** – pontos de controle e coordenação  
- **FORNECEDOR** – origem de missões/logística  
- **CLIENTE** – destino de entrega/missão  

✅ Simulação multiagente com `stage_ros`  
✅ Controladores autônomos de navegação e coordenação distribuída  

---

## 🧩 Dependências

### ROS
- **ROS Noetic** (Ubuntu 20.04)
- `stage_ros`
- `rospy`
- `geometry_msgs`, `nav_msgs`, `std_srvs`, `std_msgs`

### Python
- Python 3.8+
- Bibliotecas:

```
opencv-python
numpy
matplotlib
scikit-learn

````

Instale-as via:
```bash
pip3 install -r requirements.txt
````

ou crie o arquivo `requirements.txt` com:

```text
opencv-python
numpy
matplotlib
scikit-learn
```

---

## 🏗️ Instalação

Clone este repositório dentro do seu **catkin workspace**:

```bash
cd ~/catkin_ws/src
git clone https://github.com/<seu-usuario>/airspace_control.git
cd ..
catkin_make
source devel/setup.bash
```

---

## 🚀 Execução

### 1️⃣ Gerar o mapa e iniciar o Stage

```bash
roslaunch airspace_control airspace_stage.launch nvants:=3
```

* Gera o bitmap (`worlds/*.png`) e o arquivo `.world`
* Lança o Stage com os VANTs posicionados automaticamente

---

### 2️⃣ Rodar o sistema completo (mapa + agentes + controle)

```bash
roslaunch airspace_control airspace_all.launch nvants:=3
```

---

## 🧠 Scripts principais

| Script                                   | Descrição                                           |
| ---------------------------------------- | --------------------------------------------------- |
| `graph/processar_mapa.py`                | Detecta construções e cruzamentos em mapas (OpenCV) |
| `graph/selecionar_cluster.py`            | Seleciona regiões de interesse (clusters urbanos)   |
| `scripts/gen_stage_world.py`             | Gera o mundo `.world` e configura o Stage           |
| `scripts/uav_agent.py`                   | Define o comportamento autônomo dos VANTs           |
| `scripts/control_panel.py`               | Interface de controle de simulação                  |
| `scripts/create_map_and_launch_stage.sh` | Script de inicialização integrada                   |

---

## 🗺️ Saídas e artefatos gerados

* `worlds/airspace.world` → mundo Stage completo
* `worlds/grafo_fundo_branco.png` → grafo de referência
* `graph/grafo.txt` → definição textual do grafo (nós, posições, conexões)
* `pontos_cruzamentos.txt`, `pontos_construcoes.txt` → arquivos auxiliares de mapeamento

---

## 🧪 Exemplo de uso

```bash
roslaunch airspace_control airspace_stage.launch nvants:=4 sep_px:=5 max_wh:=2048
```

Saída esperada:

```
[INFO] Gerando mapa...
[OK] Gerado worlds/airspace.world
  - bitmap usado: worlds/mapa_base.png
  - size: [1024.00 768.00] m  resolution=1 px/m
  - VANTs: 4/4  sep_px=5
    vant_0: pose [10.00 20.00 0 0]
    vant_1: pose [15.00 20.00 0 0]
    ...
```

---

## 📜 Licença

Distribuído sob a **MIT License**.
Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
