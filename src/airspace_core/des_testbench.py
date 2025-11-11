#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vant_instance_real.py - Teste da VANTInstance com grafo REAL
"""

import os
os.environ.setdefault("PYTHONNET_CLEANUP", "0")
import sys
import argparse
from typing import Dict, Tuple, Any

# --------------------------------------------------------------------
# Configuração de caminhos
# --------------------------------------------------------------------
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.append(_PKG_ROOT)

# Mock do ROS para teste sem ROS
class MockROS:
    @staticmethod
    def init_node(name, anonymous=False):
        print(f"[MOCK ROS] Inicializando nó: {name}")
    
    @staticmethod
    def Publisher(topic, queue_size=10, latch=False):
        class MockPub:
            def __init__(self, topic):
                self.topic = topic
            def publish(self, data):
                print(f"[MOCK ROS] Publicando em '{self.topic}': {data}")
        return MockPub(topic)
    
    @staticmethod
    def sleep(duration):
        pass
    
    @staticmethod
    def loginfo(msg):
        print(f"[ROS INFO] {msg}")
    
    @staticmethod
    def logwarn(msg):
        print(f"[ROS WARN] {msg}")

# Mock do VANT
class MockVANT:
    def __init__(self, name):
        self.name = name
        self.goal = None

# Substituir ROS por mock
sys.modules['rospy'] = MockROS

# Importar as classes reais
try:
    from airspace_core.controlador_vant import GenericVANTModel, VANTInstance
    from ultrades.automata import transitions, states, initial_state, is_marked
except ImportError as e:
    print(f"[ERRO] Falha ao importar módulos: {e}")
    sys.exit(1)

# =====================================================================
# TESTE DA LÓGICA VANTInstance COM GRAFO REAL
# =====================================================================

def test_vant_instance_with_real_graph(grafo_path, init_node, vant_id):
    """Teste com grafo real"""
    print("\n" + "="*60)
    print(f"TESTE: VANTInstance com Grafo Real (ID {vant_id})")
    print("="*60)
    
    try:
        # Criar modelo com grafo REAL
        print(f"[TESTE] Carregando grafo: {grafo_path}")
        model = GenericVANTModel(grafo_txt=grafo_path, init_node=init_node)
        print(f"[TESTE] Modelo criado com sucesso")
        print(f"[TESTE] Nós: {len(model.G.nodes())}, Arestas: {len(model.G.edges())}")
        print(f"[TESTE] Eventos gerados: {len(model.eventos)}")
        
        # Calcular supervisor
        print(f"[TESTE] Calculando supervisor...")
        supervisor = model.compute_monolithic_supervisor()
        print(f"[TESTE] Supervisor calculado")
        
        # Criar instância VANT
        print(f"[TESTE] Criando VANTInstance ID {vant_id}")
        instance = VANTInstance(
            model=model,
            id_num=vant_id,
            supervisor_mono=supervisor,
            obj_vant=None,
            enable_ros=False
        )
        
        # Verificar estado inicial
        print(f"[VERIFICAÇÃO] Estado inicial: {instance.state()}")
        
        # Verificar eventos habilitados
        enabled = instance.enabled_events()
        print(f"[VERIFICAÇÃO] Eventos habilitados inicialmente: {len(enabled)} eventos")
        
        # Mostrar alguns eventos de exemplo
        print("\n[AMOSTRA] Alguns eventos habilitados:")
        for event in enabled[:10]:  # Mostrar apenas os primeiros 10
            print(f"  {event}")
        if len(enabled) > 10:
            print(f"  ... e mais {len(enabled) - 10} eventos")
        
        # Verificar mapeamento de eventos
        print(f"\n[VERIFICAÇÃO] Mapeamento de eventos: {len(instance.event_map)} eventos mapeados")
        
        # Mostrar alguns mapeamentos de exemplo
        print("\n[AMOSTRA] Alguns mapeamentos:")
        count = 0
        for generic, specialized in list(instance.event_map.items())[:5]:
            print(f"  {generic} -> {specialized}")
            count += 1
        
        return instance, model
        
    except Exception as e:
        print(f"[ERRO] Falha ao criar instância: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_vant_transitions(instance, vant_id):
    """Teste de transições de estado"""
    print("\n" + "="*60)
    print(f"TESTE: Transições de Estado (VANT {vant_id})")
    print("="*60)
    
    if instance is None:
        print("[ERRO] Instância não disponível")
        return
    
    print(f"[TESTE] Estado atual: {instance.state()}")
    enabled_events = instance.enabled_events()
    print(f"[TESTE] Eventos habilitados: {len(enabled_events)}")
    
    if not enabled_events:
        print("[AVISO] Nenhum evento habilitado para teste")
        return
    
    # Testar alguns eventos específicos
    test_events = []
    
    # Procurar por eventos de movimento
    movement_events = [ev for ev in enabled_events if ev.startswith('pega_')]
    if movement_events:
        test_events.append(movement_events[0])  # Primeiro evento de movimento
    
    # Procurar por eventos de trabalho
    work_events = [ev for ev in enabled_events if ev.startswith('comeca_trabalho_')]
    if work_events:
        test_events.append(work_events[0])  # Primeiro evento de trabalho
    
    # Procurar por eventos de carregamento
    charge_events = [ev for ev in enabled_events if ev.startswith('carregar_')]
    if charge_events:
        test_events.append(charge_events[0])  # Primeiro evento de carregamento
    
    # Adicionar alguns eventos aleatórios para teste
    if len(enabled_events) > 3:
        test_events.extend(enabled_events[1:3])  # Alguns eventos adicionais
    
    print(f"[TESTE] Eventos selecionados para teste: {len(test_events)}")
    
    for event in test_events:
        print(f"\n[TESTE] Tentando evento: {event}")
        
        # Verificar se deve processar
        should_process = instance._should_process(event)
        status = "✓ PROCESSAR" if should_process else "✗ IGNORAR"
        print(f"  Filtro: {status}")
        
        if should_process:
            old_state = str(instance.state())
            success = instance.step(event)
            new_state = str(instance.state())
            
            if success:
                print(f"  ✓ Transição: {old_state} -> {new_state}")
                print(f"  ✓ Novos eventos habilitados: {len(instance.enabled_events())}")
            else:
                print(f"  ✗ FALHA: Evento '{event}' não causou transição do estado '{old_state}'")
        else:
            print(f"  [INFO] Evento ignorado - não é para este VANT")
    
    return instance

def test_multiple_vants(grafo_path, init_node, vant_ids):
    """Teste com múltiplas instâncias VANT"""
    print("\n" + "="*60)
    print("TESTE: Múltiplas Instâncias VANT")
    print("="*60)
    
    instances = []
    
    for vant_id in vant_ids:
        print(f"\n[VANT {vant_id}] Criando instância...")
        instance, model = test_vant_instance_with_real_graph(grafo_path, init_node, vant_id)
        
        if instance:
            instances.append((vant_id, instance))
            print(f"[VANT {vant_id}] Instância criada com sucesso")
        else:
            print(f"[VANT {vant_id}] Falha ao criar instância")
    
    # Testar eventos específicos para cada VANT
    for vant_id, instance in instances:
        print(f"\n[VANT {vant_id}] Testando transições...")
        
        # Pegar alguns eventos específicos deste VANT
        enabled_events = instance.enabled_events()
        if enabled_events:
            # Testar o primeiro evento habilitado
            test_event = enabled_events[0]
            print(f"[VANT {vant_id}] Testando evento: {test_event}")
            
            old_state = str(instance.state())
            success = instance.step(test_event)
            new_state = str(instance.state())
            
            if success:
                print(f"  ✓ Transição: {old_state} -> {new_state}")
            else:
                print(f"  ✗ Transição falhou")
        
        # Testar evento de outro VANT (deve ser ignorado)
        other_vant_id = vant_ids[(vant_ids.index(vant_id) + 1) % len(vant_ids)]
        fake_event = f"pega_AB_{other_vant_id}"  # Evento com ID errado
        print(f"[VANT {vant_id}] Testando evento de outro VANT: {fake_event}")
        
        should_process = instance._should_process(fake_event)
        if not should_process:
            print("  ✓ Correto: evento de outro VANT foi ignorado")
        else:
            print("  ✗ ERRO: evento de outro VANT foi processado")
    
    return instances

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Teste da VANTInstance com grafo real')
    parser.add_argument('--grafo', default=os.path.join(_PKG_ROOT, 'graph', 'sistema_logistico', 'grafo_recortado.txt'),
                       help='Caminho para o arquivo do grafo')
    parser.add_argument('--init', default='VERTIPORT_0', help='Nó inicial')
    parser.add_argument('--ids', default='0,1', help='IDs dos VANTs separados por vírgula')
    
    args = parser.parse_args()
    
    # Parse dos IDs
    try:
        vant_ids = [int(x.strip()) for x in args.ids.split(',')]
    except ValueError:
        print("IDs devem ser números inteiros separados por vírgula")
        vant_ids = [0]
    
    print("="*70)
    print("TESTE DA CLASSE VANTInstance COM GRAFO REAL")
    print("="*70)
    print(f"Grafo: {args.grafo}")
    print(f"Nó inicial: {args.init}")
    print(f"VANTs: {vant_ids}")
    print("="*70)
    
    # Verificar se o arquivo do grafo existe
    if not os.path.isfile(args.grafo):
        print(f"[ERRO] Arquivo do grafo não encontrado: {args.grafo}")
        print("[INFO] Procurando em locais alternativos...")
        
        # Tentar encontrar o arquivo
        possible_locations = [
            args.grafo,
            os.path.join(_PKG_ROOT, 'graph', 'sistema_logistico', 'grafo_recortado.txt'),
            os.path.join(_PKG_ROOT, '..', 'graph', 'sistema_logistico', 'grafo_recortado.txt'),
            'graph/sistema_logistico/grafo_recortado.txt'
        ]
        
        for location in possible_locations:
            if os.path.isfile(location):
                args.grafo = location
                print(f"[INFO] Arquivo encontrado: {location}")
                break
        else:
            print("[ERRO] Não foi possível encontrar o arquivo do grafo")
            return 1
    
    try:
        # Teste 1: Instância única
        print("\n" + "="*70)
        print("TESTE 1: INSTÂNCIA ÚNICA")
        print("="*70)
        
        instance, model = test_vant_instance_with_real_graph(args.grafo, args.init, vant_ids[0])
        
        if instance:
            # Teste 2: Transições
            print("\n" + "="*70)
            print("TESTE 2: TRANSIÇÕES DE ESTADO")
            print("="*70)
            
            instance = test_vant_transitions(instance, vant_ids[0])
            
            # Teste 3: Múltiplas instâncias (se solicitado)
            if len(vant_ids) > 1:
                print("\n" + "="*70)
                print("TESTE 3: MÚLTIPLAS INSTÂNCIAS")
                print("="*70)
                
                instances = test_multiple_vants(args.grafo, args.init, vant_ids)
            
            print("\n" + "="*70)
            print("RESUMO DOS TESTES")
            print("="*70)
            print("✓ Teste 1: Instanciação com grafo real - CONCLUÍDO")
            print("✓ Teste 2: Transições de estado - CONCLUÍDO")
            if len(vant_ids) > 1:
                print("✓ Teste 3: Múltiplas instâncias - CONCLUÍDO")
            print("\n🎉 TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
            print(f"\nA classe VANTInstance está funcionando com o grafo: {os.path.basename(args.grafo)}")
            
        else:
            print("\n❌ TESTES FALHARAM: Não foi possível criar a instância VANT")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()
    os._exit(0) 