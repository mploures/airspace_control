import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def salvar_coordenadas(pontos, arquivo_saida, cabecalho=""):
    """
    Salva as coordenadas dos pontos em um arquivo de texto.
    Formato: uma coordenada por linha, no formato 'x,y'
    """
    with open(arquivo_saida, 'w') as f:
        if cabecalho:
            f.write(f"{cabecalho}\n")
        for x, y in pontos:
            f.write(f"{x},{y}\n")
    print(f"Coordenadas salvas em: {arquivo_saida}")


def concatenar_pontos_proximos(pontos, distancia_max=10):
    """
    Concatena pontos que estão a menos de 'distancia_max' pixels de distância,
    substituindo-os pelo ponto médio do grupo.
    """
    if len(pontos) == 0:
        return pontos

    # Converter para array numpy
    pontos_array = np.array(pontos)

    # Usar DBSCAN para agrupar pontos próximos
    clustering = DBSCAN(eps=distancia_max, min_samples=1).fit(pontos_array)
    labels = clustering.labels_

    pontos_concatenados = []
    # Para cada cluster, calcular o ponto médio
    for label in set(labels):
        pontos_cluster = pontos_array[labels == label]
        ponto_medio = np.mean(pontos_cluster, axis=0).astype(int)
        pontos_concatenados.append(tuple(ponto_medio))

    print(f"Pontos antes da concatenação: {len(pontos)}")
    print(f"Pontos após concatenação: {len(pontos_concatenados)}")
    return pontos_concatenados


def filtrar_pontos_sobre_vias(pontos, img_rgb, tolerancia=20):
    """
    Filtra pontos, mantendo apenas aqueles que estão sobre vias das cores
    #4E5154 ou #956C3E.
    """
    # Definir cores das vias (RGB)
    cor_via1 = np.array([0x95, 0x6C, 0x3E], dtype=np.int16)  # 95 6C 3E
    cor_via2 = np.array([0x4E, 0x51, 0x54], dtype=np.int16)  # 4E 51 54

    # Criar faixas com tolerância e fazer clip para [0,255]
    lower1 = np.clip(cor_via1 - tolerancia, 0, 255).astype(np.uint8)
    upper1 = np.clip(cor_via1 + tolerancia, 0, 255).astype(np.uint8)
    lower2 = np.clip(cor_via2 - tolerancia, 0, 255).astype(np.uint8)
    upper2 = np.clip(cor_via2 + tolerancia, 0, 255).astype(np.uint8)

    # Criar máscaras para vias
    mask_via1 = cv2.inRange(img_rgb, lower1, upper1)
    mask_via2 = cv2.inRange(img_rgb, lower2, upper2)
    mask_vias = cv2.bitwise_or(mask_via1, mask_via2)

    pontos_filtrados = []
    pontos_removidos = 0
    h, w = mask_vias.shape[:2]

    for x, y in pontos:
        # Verificar se o ponto está sobre uma via
        if (0 <= y < h) and (0 <= x < w) and (mask_vias[y, x] > 0):
            pontos_filtrados.append((x, y))
        else:
            pontos_removidos += 1

    print(f"Pontos removidos (não estão sobre vias): {pontos_removidos}")
    print(f"Pontos restantes após filtro de vias: {len(pontos_filtrados)}")
    return pontos_filtrados, mask_vias


def calcular_centros_retangulos(contornos, area_minima=0):
    """
    Calcula os centros de todos os retângulos/contornos detectados.
    """
    centros = []
    for i, contorno in enumerate(contornos):
        # Filtrar por área mínima se especificado
        area = cv2.contourArea(contorno)
        if area < area_minima:
            continue

        # Calcular o momento do contorno para encontrar o centro
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            centro_x = int(M["m10"] / M["m00"])
            centro_y = int(M["m01"] / M["m00"])
            centros.append((centro_x, centro_y))
        else:
            # Se não conseguir calcular pelos momentos, usar o retângulo delimitador
            x, y, w, h = cv2.boundingRect(contorno)
            centro_x = x + w // 2
            centro_y = y + h // 2
            centros.append((centro_x, centro_y))

    print(f"Centros de construções calculados: {len(centros)}")
    return centros


def detectar_retangulos_pretos(
    imagem_path,
    output_path,
    arquivo_cruzamentos=None,
    arquivo_construcoes=None,
    distancia_concatenacao=10
):
    """
    Detecta retângulos pretos (construções) e marca:
    - Pontos verdes nos vértices (cruzamentos)
    - Pontos azuis nos centros (construções)
    """
    # Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: Não foi possível carregar a imagem")
        return

    # Converter para RGB para exibição
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Converter para escala de cinza
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Criar máscara para áreas pretas (construções)
    _, mask_pretas = cv2.threshold(img_cinza, 50, 255, cv2.THRESH_BINARY_INV)

    # Operações morfológicas para limpar a máscara
    kernel = np.ones((3, 3), np.uint8)
    mask_pretas = cv2.morphologyEx(mask_pretas, cv2.MORPH_OPEN, kernel)
    mask_pretas = cv2.morphologyEx(mask_pretas, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mask_pretas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contornos encontrados: {len(contornos)}")

    # ETAPA 1: Encontrar vértices dos retângulos (cruzamentos)
    pontos_cruzamento = []

    # Processar cada contorno para extrair vértices
    for i, contorno in enumerate(contornos):
        # Aproximar o contorno para um polígono
        epsilon = 0.02 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)

        # Se tiver 4 vértices, é um retângulo/quadrilátero
        if len(approx) == 4:
            vertices = approx.reshape(4, 2)
            for vertice in vertices:
                x, y = vertice
                pontos_cruzamento.append((int(x), int(y)))
        else:
            # Usar retângulo delimitador
            x, y, w, h = cv2.boundingRect(contorno)
            pontos_cruzamento.extend(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            )

    # Processar vértices (cruzamentos)
    pontos_concatenados = concatenar_pontos_proximos(pontos_cruzamento, distancia_concatenacao)
    pontos_cruzamentos_filtrados, mask_vias = filtrar_pontos_sobre_vias(pontos_concatenados, img_rgb)

    # ETAPA 2: Calcular centros dos retângulos (construções)
    pontos_construcoes = calcular_centros_retangulos(contornos)

    print("RESUMO:")
    print(f"  Cruzamentos: {len(pontos_cruzamentos_filtrados)} pontos")
    print(f"  Construções: {len(pontos_construcoes)} pontos")

    # Criar imagem de resultado - APENAS COM OS PONTOS, SEM CONTORNOS
    img_resultado = img_rgb.copy()

    # Marcar pontos de CRUZAMENTO nos vértices (VERDE) - SEM BORDA BRANCA
    for x, y in pontos_cruzamentos_filtrados:
        cv2.circle(img_resultado, (x, y), 6, (0, 255, 0), -1)  # Verde sólido

    # Marcar pontos de CONSTRUÇÃO nos centros (AZUL) - SEM BORDA BRANCA
    for x, y in pontos_construcoes:
        cv2.circle(img_resultado, (x, y), 6, (255, 0, 0), -1)  # Azul sólido

    # Salvar imagem resultante
    img_resultado_bgr = cv2.cvtColor(img_resultado, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_resultado_bgr)

    # Salvar coordenadas em arquivos separados
    if arquivo_cruzamentos:
        salvar_coordenadas(pontos_cruzamentos_filtrados, arquivo_cruzamentos, "Cruzamentos (x,y)")
    if arquivo_construcoes:
        salvar_coordenadas(pontos_construcoes, arquivo_construcoes, "Construções (x,y)")

    # Mostrar imagens para análise
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_pretas, cmap='gray')
    plt.title('Máscara - Áreas Pretas')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_resultado)
    plt.title(f'Cruzamentos: {len(pontos_cruzamentos_filtrados)} (Verde)\nConstruções: {len(pontos_construcoes)} (Azul)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return img_resultado, pontos_cruzamentos_filtrados, pontos_construcoes


def detectar_retangulos_pretos_melhorado(
    imagem_path,
    output_path,
    arquivo_cruzamentos=None,
    arquivo_construcoes=None,
    area_minima=100,
    distancia_concatenacao=10
):
    """
    Versão melhorada para detectar retângulos pretos com filtros adicionais.
    """
    # Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: Não foi possível carregar a imagem")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Método mais agressivo para detectar pretos
    mask_pretas = cv2.adaptiveThreshold(
        img_cinza, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Operações morfológicas mais robustas
    kernel_abertura = np.ones((2, 2), np.uint8)
    kernel_fechamento = np.ones((5, 5), np.uint8)
    mask_pretas = cv2.morphologyEx(mask_pretas, cv2.MORPH_OPEN, kernel_abertura)
    mask_pretas = cv2.morphologyEx(mask_pretas, cv2.MORPH_CLOSE, kernel_fechamento)

    # Encontrar contornos
    contornos, _ = cv2.findContours(mask_pretas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ETAPA 1: Encontrar vértices (cruzamentos)
    pontos_cruzamento = []
    retangulos_detectados = 0

    for contorno in contornos:
        # Filtrar por área mínima
        area = cv2.contourArea(contorno)
        if area < area_minima:
            continue

        # Tentar aproximar como quadrilátero
        epsilon = 0.03 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)

        # Se for quadrilátero (4 lados)
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                vertices = approx.reshape(4, 2)
                for vertice in vertices:
                    x, y = vertice
                    pontos_cruzamento.append((int(x), int(y)))
                retangulos_detectados += 1
        else:
            # Usar retângulo delimitador
            x, y, w, h = cv2.boundingRect(contorno)
            proporcao = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if proporcao < 3:
                pontos_cruzamento.extend(
                    [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                )
                retangulos_detectados += 1

    # Processar vértices (cruzamentos)
    pontos_concatenados = concatenar_pontos_proximos(pontos_cruzamento, distancia_concatenacao)
    pontos_cruzamentos_filtrados, mask_vias = filtrar_pontos_sobre_vias(pontos_concatenados, img_rgb)

    # ETAPA 2: Calcular centros (construções)
    pontos_construcoes = calcular_centros_retangulos(contornos, area_minima)

    print("RESUMO - Método Melhorado:")
    print(f"  Retângulos detectados: {retangulos_detectados}")
    print(f"  Cruzamentos: {len(pontos_cruzamentos_filtrados)} pontos")
    print(f"  Construções: {len(pontos_construcoes)} pontos")

    # Criar imagem de resultado - APENAS COM OS PONTOS
    img_resultado = img_rgb.copy()

    # Marcar pontos - SEM BORDAS BRANCAS
    for x, y in pontos_cruzamentos_filtrados:  # Cruzamentos - VERDE
        cv2.circle(img_resultado, (x, y), 5, (0, 255, 0), -1)
    for x, y in pontos_construcoes:            # Construções - AZUL
        cv2.circle(img_resultado, (x, y), 5, (255, 0, 0), -1)

    # Salvar
    cv2.imwrite(output_path, cv2.cvtColor(img_resultado, cv2.COLOR_RGB2BGR))

    # Salvar arquivos
    if arquivo_cruzamentos:
        salvar_coordenadas(pontos_cruzamentos_filtrados, arquivo_cruzamentos, "Cruzamentos (x,y)")
    if arquivo_construcoes:
        salvar_coordenadas(pontos_construcoes, arquivo_construcoes, "Construções (x,y)")

    return img_resultado, pontos_cruzamentos_filtrados, pontos_construcoes


# Função principal
def processar_retangulos_pretos(
    imagem_path,
    output_path,
    metodo='simples',
    arquivo_cruzamentos=None,
    arquivo_construcoes=None,
    area_minima=100,
    distancia_concatenacao=10
):
    """
    Processa o mapa e marca:
    - Pontos verdes nos vértices dos retângulos (cruzamentos)
    - Pontos azuis nos centros dos retângulos (construções)

    Args:
        imagem_path: Caminho para a imagem do mapa
        output_path: Caminho para salvar a imagem resultante
        metodo: 'simples' ou 'melhorado'
        arquivo_cruzamentos: Arquivo para salvar coordenadas dos cruzamentos
        arquivo_construcoes: Arquivo para salvar coordenadas das construções
        area_minima: Área mínima para considerar um contorno
        distancia_concatenacao: Distância máxima para concatenar pontos
    """
    if metodo == 'melhorado':
        return detectar_retangulos_pretos_melhorado(
            imagem_path, output_path,
            arquivo_cruzamentos, arquivo_construcoes,
            area_minima, distancia_concatenacao
        )
    else:
        return detectar_retangulos_pretos(
            imagem_path, output_path,
            arquivo_cruzamentos, arquivo_construcoes,
            distancia_concatenacao
        )


# Usar a função
if __name__ == "__main__":
    imagem_original = 'mapafinal.png'
    imagem_resultado = 'mapa_completo.png'
    arquivo_cruzamentos = 'pontos_cruzamentos.txt'
    arquivo_construcoes = 'pontos_construcoes.txt'

    # Processar retângulos pretos
    resultado, cruzamentos, construcoes = processar_retangulos_pretos(
        imagem_original,
        imagem_resultado,
        metodo='simples',  # 'simples' ou 'melhorado'
        arquivo_cruzamentos=arquivo_cruzamentos,
        arquivo_construcoes=arquivo_construcoes,
        distancia_concatenacao=50
    )

    print("\n=== RESULTADOS FINAIS ===")
    print(f"Imagem resultante salva como: {imagem_resultado}")
    print(f"Cruzamentos salvos em: {arquivo_cruzamentos} ({len(cruzamentos)} pontos)")
    print(f"Construções salvas em: {arquivo_construcoes} ({len(construcoes)} pontos)")

    # Exibir algumas coordenadas de exemplo
    if cruzamentos:
        print("\nPrimeiros 5 CRUZAMENTOS:")
        for i, (x, y) in enumerate(cruzamentos[:5]):
            print(f"  {i+1}: ({x}, {y})")
    if construcoes:
        print("\nPrimeiras 5 CONSTRUÇÕES:")
        for i, (x, y) in enumerate(construcoes[:5]):
            print(f"  {i+1}: ({x}, {y})")
