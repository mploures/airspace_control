#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
des_testbench.py
----------------
Testbench completo para a classe ControladorVANT:

- Carrega grafo_recortado.txt
- Instancia ControladorVANT (conversão p/ MultiDiGraph é interna)
- Valida:
    * contagens de eventos (pega_/libera_/trabalho/carga)
    * controlabilidade dos eventos
    * mapeamento pega_* -> (x,y)
    * número e estrutura dos DFAs criados
    * specs (mapa, bat_mov)
    * determinismo e nº de transições (robusto a IEnumerable da UltraDES)
- (Opcional) Computa supervisor monolítico e/ou supervisores modulares locais

Uso:
  python3 src/airspace_core/des_testbench.py \
      --grafo graph/sistema_logistico/grafo_recortado.txt \
      --id 0 --init VERTIPORT_0 --supervisores both
"""

from typing import Dict, Tuple, Any, List
import argparse
import os
import sys
import re
import networkx as nx

# Importa o controlador local (mesmo diretório deste script)
from controlador_vant import ControladorVANT

# --- Caminho para importar graph/gerar_grafo.py ---
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.append(_PKG_ROOT)

from graph.gerar_grafo import carregar_grafo_txt  # retorna (G, pos)

# UltraDES (apenas o que usamos explicitamente)
from pythonnet import load
load("coreclr")

from ultrades.automata import *

# ----------------------------- Util: carregar posições -----------------------------
_COORD_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")

def carregar_posicoes(caminho_arquivo: str) -> Dict[str, Tuple[float, float]]:
    posicoes: Dict[str, Tuple[float, float]] = {}
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        _ = f.readline()  # cabeçalho
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split(",", 3)
            if len(partes) < 3:
                continue
            label = partes[1].strip()
            pos_raw = partes[2].strip()
            m = _COORD_RE.match(pos_raw) or _COORD_RE.search(linha)
            if not m:
                raise ValueError(f"Não foi possível extrair coordenadas de: {linha}")
            x = float(m.group(1)); y = float(m.group(2))
            posicoes[label] = (x, y)
    return posicoes

def _tipo_norm(s: Any) -> str:
    return str(s).strip().upper()

# ========================= Helpers robustos p/ UltraDES (IEnumerable) =========================
def _to_list(x):
    """Converte qualquer IEnumerable/iterável da UltraDES para list."""
    try:
        return list(x)
    except TypeError:
        try:
            out = []
            for it in x:
                out.append(it)
            return out
        except Exception:
            return []

def _count_trans(Gdfa) -> int:
    """
    Conta transições de forma robusta.
    - dict-like: {q: {e: q'}}
    - sequência de triplas: [(q, e, q'), ...]
    - IEnumerable opaco: itera e conta
    """
    td = transitions(Gdfa)

    # dict-like?
    if hasattr(td, "items"):
        total = 0
        for _, evmap in td.items():
            if hasattr(evmap, "items"):
                total += len(_to_list(evmap.items()))
            else:
                total += len(_to_list(evmap))
        return total

    # sequência comum (lista/tupla/conjunto) com __len__
    try:
        return len(td)
    except Exception:
        pass

    # fallback: contador por iteração
    try:
        c = 0
        for _ in td:
            c += 1
        return c
    except Exception:
        return 0

def _is_deterministic(Gdfa) -> bool:
    """
    Checa determinismo:
    - dict-like: para cada estado, não deve haver duas saídas com o mesmo evento
    - sequência/iterável de triplas: não pode ter (q,e) indo a destinos diferentes
    """
    td = transitions(Gdfa)

    if hasattr(td, "items"):
        for _q, evmap in td.items():
            seen = set()
            if hasattr(evmap, "items"):
                for e, _dest in evmap.items():
                    if e in seen:
                        return False
                    seen.add(e)
            else:
                try:
                    for e, _dest in evmap:
                        if e in seen:
                            return False
                        seen.add(e)
                except Exception:
                    continue
        return True

    try:
        seen = {}
        for t in td:
            if not (isinstance(t, tuple) and len(t) == 3):
                continue
            q, e, q2 = t
            k = (q, e)
            if k in seen and seen[k] != q2:
                return False
            seen[k] = q2
        return True
    except Exception:
        return True

def _summarize(name: str, Gdfa):
    qs = _to_list(states(Gdfa))
    ms = _to_list(marked_states(Gdfa))
    evs = _to_list(events(Gdfa))
    ntr = _count_trans(Gdfa)
    det = _is_deterministic(Gdfa)
    try:
        q0 = initial_state(Gdfa)
    except Exception:
        q0 = None
    print(f"    · {name:18s} | Q={len(qs):3d}  Qm={len(ms):3d}  Σ={len(evs):3d}  δ={ntr:4d}  det={det}  q0={q0}")

# ================================== Supervisores ==================================
def _save_dfa(G, path: str):
    """Salva um DFA no formato indicado pela extensão do arquivo."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        write_xml(G, path)
    elif ext == ".fsm":
        write_fsm(G, path)
    elif ext == ".ads":
        write_ads(G, path)
    elif ext == ".bin":
        write_bin(G, path)
    else:
        # default seguro
        write_xml(G, path)

def _build_supervisors(ctrl, modo, out_dir="out_sup", fmt="xml"):
    """
    Gera o(s) supervisor(es) e, se 'out_dir' for fornecido, salva em disco.
      - modo: "modular", "mono", "both" ou "none"
      - out_dir: diretório para salvar (ex.: "out_sup")
      - fmt: "xml" | "fsm" | "ads" | "bin" | "fm"
    """
    plants = tuple(ctrl.plantas)   # força array .NET no pythonnet
    specs  = tuple(ctrl.specs)

    # prepara saída
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # sempre salvo também o problema (plantas+specs) para reprodutibilidade
        write_wmod(plants, specs, os.path.join(out_dir, "plants_specs.wmod"))

    sup_list = []
    tag = None

    if modo in ("modular", "both"):
        print("[SUP] Computando supervisores modulares locais...")
        try:
            sup_list = list(local_modular_supervisors(plants, specs))
            tag = "lm"
            print(f"[SUP] local_modular_supervisors: {len(sup_list)} módulo(s).")
        except Exception as ex:
            print(f"[SUP][WARN] local_modular_supervisors falhou: {ex}")
            print("[SUP] Tentando localized_supervisors...")
            try:
                sup_list = list(localized_supervisors(plants, specs))
                tag = "loc"
                print(f"[SUP] localized_supervisors: {len(sup_list)} módulo(s).")
            except Exception as ex2:
                print(f"[SUP][WARN] localized_supervisors falhou: {ex2}")
                print("[SUP] Recuando para monolithic_supervisor...")
                S_mono = monolithic_supervisor(plants, specs)
                sup_list = [S_mono]
                tag = "mono"
                print("[SUP] monolithic_supervisor OK (fallback).")

        if out_dir and sup_list:
            for i, G in enumerate(sup_list):
                path = os.path.join(out_dir, f"sup_{tag}_{i:02d}.{fmt}")
                _save_dfa(G, path)
            print(f"[SUP] Supervisores salvos em: {out_dir}")

        # se 'both', continua e também salva o monolítico;
        # se apenas 'modular', retorna aqui
        if modo == "modular":
            return sup_list

    if modo in ("mono", "both"):
        print("[SUP] Computando monolithic_supervisor...")
        S_mono = monolithic_supervisor(plants, specs)
        print("[SUP] monolithic_supervisor OK.")
        if out_dir:
            path = os.path.join(out_dir, f"sup_mono.{fmt}")
            _save_dfa(S_mono, path)
            print(f"[SUP] Supervisor monolítico salvo em: {path}")
        sup_list.append(S_mono)
        return sup_list

    print("[SUP] Modo de supervisão 'none'.")
    return sup_list


# ================================== Test Runner ==================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafo", default=os.path.join(_PKG_ROOT, "graph", "sistema_logistico", "grafo_recortado.txt"))
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--init", default="VERTIPORT_0")
    ap.add_argument("--amostra", type=int, default=12, help="quantos eventos 'pega_' mostrar")
    ap.add_argument("--supervisores", choices=["none", "mono", "modular", "both"], default="none",
                    help="calcular supervisor monolítico/modular")
    args = ap.parse_args()

    # Carrega grafo bruto e mostra estatísticas por tipo
    G_raw, pos_raw = carregar_grafo_txt(args.grafo)
    tipos: Dict[str, int] = {}
    for _, d in G_raw.nodes(data=True):
        t = _tipo_norm(d.get("tipo", ""))
        tipos[t] = tipos.get(t, 0) + 1

    print("[INFO] Estatísticas do grafo:")
    print(f"  Nós: {G_raw.number_of_nodes()}  Arestas (não-dirigidas): {G_raw.number_of_edges()}")
    print("  Por tipo:", ", ".join(f"{k}={v}" for k, v in sorted(tipos.items())))
    tipos_esperados = {"LOGICO", "CLIENTE", "FORNECEDOR", "ESTACAO", "VERTIPORT"}
    if not tipos.keys() <= tipos_esperados:
        print(f"[WARN] Foram encontrados tipos fora do conjunto esperado {tipos_esperados}: "
              f"{set(tipos.keys()) - tipos_esperados}")

    # Instancia controlador (gera alfabeto e mapeamentos)
    ctrl = ControladorVANT(id_vant=args.id, obj_vant=None, grafo_txt=args.grafo, init_node=args.init)

    # Coleta eventos por categoria
    ev_keys = list(ctrl.eventos.keys())
    pega_events   = sorted([k for k in ev_keys if k.startswith("pega_")])
    libera_events = sorted([k for k in ev_keys if k.startswith("libera_")])
    trab_ini      = sorted([k for k in ev_keys if k.startswith("comeca_trabalho_")])
    trab_fim      = sorted([k for k in ev_keys if k.startswith("fim_trabalho_")])
    carrega_ini   = sorted([k for k in ev_keys if k.startswith("carregar_")])
    carrega_fim   = sorted([k for k in ev_keys if k.startswith("fim_carregar_")])

    # Expectativas básicas
    E_und = G_raw.number_of_edges()
    exp_pega = 2 * E_und
    exp_lib  = 2 * E_und

    print(f"\n[INFO] Eventos gerados (total={len(ev_keys)}):")
    print(f"  pega_*      = {len(pega_events)}  (esperado {exp_pega})")
    print(f"  libera_*    = {len(libera_events)} (esperado {exp_lib})")
    print(f"  começa trab = {len(trab_ini)}")
    print(f"  fim trab    = {len(trab_fim)}")
    print(f"  carregar    = {len(carrega_ini)}")
    print(f"  fim_carga   = {len(carrega_fim)}")

    # Validação: contagem de pega/libera
    if len(pega_events) != exp_pega:
        print(f"[WARN] Número de 'pega_*' difere do esperado ({len(pega_events)} != {exp_pega}).")
    if len(libera_events) != exp_lib:
        print(f"[WARN] Número de 'libera_*' difere do esperado ({len(libera_events)} != {exp_lib}).")

    # Validação: controlabilidade
    pega_bad = [e for e in pega_events if not is_controllable(ctrl.eventos[e])]
    libera_bad = [e for e in libera_events if is_controllable(ctrl.eventos[e])]
    if pega_bad:
        print(f"[ERRO] Há 'pega_*' não-controláveis: {pega_bad[:min(5,len(pega_bad))]} ...")
    else:
        print("[OK] Todos os 'pega_*' são controláveis.")
    if libera_bad:
        print(f"[ERRO] Há 'libera_*' controláveis: {libera_bad[:min(5,len(libera_bad))]} ...")
    else:
        print("[OK] Todos os 'libera_*' são não-controláveis.")

    # Validação: eventos de trabalho e carga por tipos de nó (apenas 5 tipos)
    fornecedores = [n for n, d in G_raw.nodes(data=True) if _tipo_norm(d.get("tipo","")) == "FORNECEDOR"]
    clientes     = [n for n, d in G_raw.nodes(data=True) if _tipo_norm(d.get("tipo","")) == "CLIENTE"]
    estacoes     = [n for n, d in G_raw.nodes(data=True) if _tipo_norm(d.get("tipo","")) == "ESTACAO"]
    vertiports   = [n for n, d in G_raw.nodes(data=True) if _tipo_norm(d.get("tipo","")) == "VERTIPORT"]

    exp_trab  = len(fornecedores) + len(clientes)
    exp_carga = len(estacoes) + len(vertiports)

    if len(trab_ini) != exp_trab or len(trab_fim) != exp_trab:
        print(f"[WARN] Eventos de TRABALHO esperados={exp_trab} "
              f"(ini={len(trab_ini)}, fim={len(trab_fim)}).")
    else:
        print("[OK] Eventos de trabalho batem com nós FORNECEDOR/CLIENTE.")

    if len(carrega_ini) != exp_carga or len(carrega_fim) != exp_carga:
        print(f"[WARN] Eventos de CARGA esperados={exp_carga} "
              f"(ini={len(carrega_ini)}, fim={len(carrega_fim)}).")
    else:
        print("[OK] Eventos de carga batem com nós ESTACAO/VERTIPORT.")

    # Validação: mapeamento pega_* -> posição (x,y)
    faltando = [e for e in pega_events if e not in ctrl.posicao_evento]
    if faltando:
        print(f"[WARN] {len(faltando)} eventos 'pega_' sem posição mapeada (nó destino pode não ter coordenada).")
        for e in faltando[:min(5, len(faltando))]:
            print("   -", e)
    else:
        print("[OK] Todos os 'pega_' estão mapeados para posições (x,y).")

    # Amostra do mapeamento: pega_* -> (x,y)
    print("\n[AMOSTRA] pega_* → posição (x,y) do nó destino:")
    for e in pega_events[:max(0, args.amostra)]:
        ev_xy = ctrl.posicao_evento.get(e, None)
        xy = ev_xy[1] if ev_xy else None
        print(f"  {e:35s} -> {xy}")

    # Checagem simples do nó inicial
    if args.init not in G_raw.nodes:
        print(f"\n[WARN] Nó inicial '{args.init}' não existe no grafo. Escolha um label válido (ex.: 'VERTIPORT_0').")
    else:
        xy0 = carregar_posicoes(args.grafo).get(args.init, None)
        print(f"\n[INFO] Nó inicial: {args.init}  pos={xy0}")

    # ===================== Testes da "biblioteca" (ControladorVANT) =====================
    print("\n[LIB] Validação do ControladorVANT/DFAs")

    # 1) dict_aresta_eventos deve ter 1 entrada por aresta não-dirigida
    undirected_edges = G_raw.number_of_edges()
    print(f"  - Arestas não-dirigidas no grafo: {undirected_edges}")
    print(f"  - dict_aresta_eventos: {len(ctrl.dict_aresta_eventos)} entradas")
    if len(ctrl.dict_aresta_eventos) != undirected_edges:
        print("  [WARN] dict_aresta_eventos deveria ter 1 por aresta não-dirigida.")

    # 2) Quantidade de autômatos criados
    expected_plants = 1 + (2 * undirected_edges) + 1 + 3  # movimento + 2*E + modos + (com, vivo, bat)
    print(f"  - Plantas criadas (len(ctrl.plantas)): {len(ctrl.plantas)}  (esperado ~ {expected_plants})")
    if len(ctrl.plantas) != expected_plants:
        print("  [WARN] Número de plantas difere do esperado. Verifique arestas duplicadas/chaves.")

    # 3) Specs esperadas (mapa, bat_mov)
    print(f"  - Especificações (len(ctrl.specs)): {len(ctrl.specs)}  (esperado = 2)")
    has_mapa = "mapa" in ctrl.Dicionario_Automatos
    has_bm   = "bat_mov" in ctrl.Dicionario_Automatos
    print(f"    · contém 'mapa':    {has_mapa}")
    print(f"    · contém 'bat_mov': {has_bm}")
    if not (has_mapa and has_bm):
        print("  [WARN] Especificações ausentes no Dicionario_Automatos.")

    # 4) Sumário por autômato principal
    print("  - Resumo dos principais DFAs:")
    for k in ["movimento", "modos", "comunicacao", "vivacidade", "bateria", "mapa", "bat_mov"]:
        if k in ctrl.Dicionario_Automatos:
            _summarize(k, ctrl.Dicionario_Automatos[k])
        else:
            print(f"    · {k:18s} | [AUSENTE]")

    # 5) Amostra de DFAs aresta_*
    amostra_arestas = [k for k in ctrl.Dicionario_Automatos.keys() if k.startswith("aresta_")]
    amostra_arestas.sort()
    if not amostra_arestas:
        print("  [WARN] Nenhum DFA aresta_* encontrado.")
    else:
        print("  - Amostra DFAs aresta_*:")
        for k in amostra_arestas[:min(6, len(amostra_arestas))]:
            _summarize(k, ctrl.Dicionario_Automatos[k])

    # 6) Checagem leve do automato de movimento
    mov = ctrl.Dicionario_Automatos.get("movimento")
    if mov:
        evs_mov = {str(e) for e in _to_list(events(mov))}
        has_pega = any(s.startswith("pega_") for s in evs_mov)
        has_lib  = any(s.startswith("libera_") for s in evs_mov)
        if not has_pega or not has_lib:
            print("  [ERRO] 'movimento' não contém eventos pega_/libera_.")
        else:
            print("  - 'movimento' contém eventos pega_/libera_. OK")
    else:
        print("  [ERRO] DFA 'movimento' não encontrado.")

    # 7) Supervisor
    if args.supervisores != "none":
        sup = _build_supervisors(ctrl, args.supervisores)
        if not sup:
            print("[SUP][ERRO] Nenhum supervisor retornado.")
        else:
            supervisor = sup[0]
            plants = tuple(ctrl.plantas)   
            specs  = tuple(ctrl.specs)

            print(f"plants:{len(plants)}")
            print(f"specs:{len(specs)}")
            # Versão segura para coleções IEnumerable da UltraDES:
            qs  = _to_list(states(supervisor))
            evs = _to_list(events(supervisor))
            ntr = _count_trans(supervisor)
            print(f"Estados:{len(qs)} Eventos:{len(evs)} Trans:{ntr}")


        print("\n[PRONTO] Testbench finalizado.")

if __name__ == "__main__":
    main()
