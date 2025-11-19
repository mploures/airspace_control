import os
import sys

# *******************************************************************
# IMPORT OBRIGATÓRIO (SEM TRY/CATCH)
# *******************************************************************
from airspace_core.UTM import * # *******************************************************************

def test_utm_model():
    """
    Função principal para testar a inicialização e síntese da UTMModel.
    """
    
    # --- Configurações de teste ---
    grafo_path = "/home/mploures/catkin_ws/src/airspace_control/graph/sistema_logistico/grafo_recortado.txt"
    
    # Nó inicial (usado apenas na inicialização do grafo, não afeta a lógica UTM)
    init_node = "VERTIPORT_0" 
    
    print(f"--- 🚀 Iniciando Teste Manual da UTMModel ---")
    print(f"Grafo de Entrada: {grafo_path}")

    # 1. Inicialização da Classe UTMModel
    try:
        utm = UTMModel(grafo_txt=grafo_path, init_node=init_node)
        print("\n[SUCESSO] UTMModel inicializada com sucesso.")
        
    except Exception as e:
        print(f"\n[ERRO FATAL] Falha ao inicializar UTMModel ou calcular supervisor: {e}")
        # Se a inicialização falhar, paramos o teste.
        return
    

    # A. Propriedades do Grafo
    print(f"\n--- 🗺️ Propriedades do Grafo ---")
    print(f"Nós no Grafo (V): {len(utm.G.nodes)}")
    print(f"Arestas no Grafo (E): {len(utm.G.edges)}")

    # B. Propriedades do Modelo
    print(f"\n--- ⚙️ Propriedades da Modelagem ---")
    print(f"Número de Eventos Únicos (Alfabeto): {len(utm.eventos)}")
    print(f"Número de Plantas (Recursos): {len(utm.plantas)}")
    print(f"Número de Especificações (Restrições UTM): {len(utm.specs)}")


    # 2. Verificação das Propriedades da Classe e do Supervisor
    supervisor = utm.supervisor_mono
    
    if supervisor is None:
        print("\n[ERRO] O supervisor monolítico 'utm.supervisor_mono' é None.")
        return
    print(f"\n--- 🧠 Propriedades do Supervisor S_UTM ---")
    try:
        num_estados = len(states(supervisor))
        num_transicoes = len(transitions(supervisor))
        eventos_supervisor = events(supervisor)
        
        print(f"Supervisor (S_UTM) calculado.")
        print(f"Total de Estados Acessíveis: {num_estados}")
        print(f"Total de Transições: {num_transicoes}")
        print(f"Eventos no Supervisor (Alfabeto): {len(eventos_supervisor)}")

        # Verificação da controlabilidade
        controlaveis = sum(1 for e in eventos_supervisor if is_controllable(e))
        nao_controlaveis = len(eventos_supervisor) - controlaveis
        print(f"Eventos Controláveis no S_UTM: {controlaveis}")
        print(f"Eventos Não-Controláveis no S_UTM: {nao_controlaveis}")
        
    except Exception as e:
        print(f"[ERRO] Falha ao inspecionar o supervisor: {e}")

    i=0

    proibidos_incial=utm.eventos_proibidos_estado[utm.agent_state[0]]

    for e in proibidos_incial:
        print(str(e))


    print("\n--- ✅ Teste Manual Concluído ---")

if __name__ == "__main__":
    test_utm_model()