import numpy as np

def fekine2d(nnel: int, dhdx: np.ndarray, dhdy: np.ndarray) -> np.ndarray:
    """
    Determina a matriz cinemática que relaciona deformações e deslocamentos
    para sólidos bidimensionais.
    
    Parâmetros:
    -----------
    nnel : int
        Número de nós por elemento
    dhdx : np.ndarray
        Derivadas das funções de forma em relação a x
    dhdy :  np.ndarray
        Derivadas das funções de forma em relação a y
    
    Retorna:
    --------
    kinmtx2 :  np.ndarray
        Matriz cinemática [B] de dimensão (3, 2*nnel)
    """
    
    # Pré-alocação da matriz (3 linhas, 2*nnel colunas)
    kinmtx2 = np.zeros((3, 2 * nnel))
    
    for i in range(nnel):
        # Índices das colunas (ajustados para indexação base-0)
        i1 = 2 * i      # Em MATLAB seria (i-1)*2+1, aqui é 2*i
        i2 = i1 + 1     # Coluna seguinte
        
        # Preenchimento da matriz B
        # Linha 1: ∂u/∂x (deformação εx)
        kinmtx2[0, i1] = dhdx[i]
        
        # Linha 2: ∂v/∂y (deformação εy)
        kinmtx2[1, i2] = dhdy[i]
        
        # Linha 3: ∂u/∂y + ∂v/∂x (deformação γxy)
        kinmtx2[2, i1] = dhdy[i]
        kinmtx2[2, i2] = dhdx[i]
    
    return kinmtx2