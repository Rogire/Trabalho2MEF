import numpy as np

def feaplyc2(kk: np.ndarray, ff: np.ndarray, 
             bcdof: np. ndarray, bcval: np. ndarray) -> tuple:
    """
    Aplica condições de contorno (deslocamentos prescritos) ao sistema [K]{u}={F}.
    
    Utiliza o método de eliminação:  zera a linha correspondente ao DOF restringido,
    coloca 1 na diagonal e o valor prescrito no vetor de forças.
    
    Parâmetros:
    -----------
    kk : np.ndarray
        Matriz de rigidez do sistema (será modificada)
    ff : np.ndarray
        Vetor de forças do sistema (será modificado)
    bcdof : np.ndarray
        Vetor contendo os DOFs com deslocamento prescrito (base-1)
    bcval : np.ndarray
        Vetor contendo os valores prescritos de deslocamento
    
    Retorna: 
    --------
    tuple :  (kk, ff)
        Matriz e vetor modificados com condições de contorno aplicadas
    
    Exemplo:
    --------
    Para prescrever u=0 nos DOFs 1 e 2, e u=0. 5 no DOF 5:
    >>> bcdof = np.array([1, 2, 5])
    >>> bcval = np.array([0.0, 0.0, 0.5])
    >>> kk, ff = feaplyc2(kk, ff, bcdof, bcval)
    """
    
    n = len(bcdof)        # Número de condições de contorno
    sdof = kk.shape[0]    # Número total de DOFs (dimensão da matriz)
    
    for i in range(n):
        c = bcdof[i] - 1  # DOF restringido (convertido para base-0)
        
        # Zera toda a linha c
        kk[c, : ] = 0
        
        # Coloca 1 na diagonal
        kk[c, c] = 1
        
        # Define o valor prescrito no vetor de forças
        ff[c] = bcval[i]
    
    return kk, ff


def feaplyc2_symmetric(kk: np.ndarray, ff: np.ndarray,
                       bcdof: np.ndarray, bcval: np.ndarray) -> tuple:
    """
    Aplica condições de contorno preservando a simetria da matriz.
    
    Este método é preferível quando se usa solvers que exploram simetria.
    Além de zerar a linha, também zera a coluna e ajusta o vetor de forças. 
    
    Parâmetros:
    -----------
    kk : np.ndarray
        Matriz de rigidez simétrica do sistema
    ff : np.ndarray
        Vetor de forças do sistema
    bcdof : np.ndarray
        Vetor de DOFs restringidos (base-1)
    bcval : np.ndarray
        Vetor de valores prescritos
    
    Retorna: 
    --------
    tuple : (kk, ff)
        Sistema modificado mantendo simetria
    """
    
    n = len(bcdof)
    sdof = kk.shape[0]
    
    for i in range(n):
        c = bcdof[i] - 1  # Conversão para base-0
        prescribed_value = bcval[i]
        
        # Ajusta o vetor de forças para manter simetria
        # ff = ff - kk[: , c] * prescribed_value
        for j in range(sdof):
            if j != c:
                ff[j] = ff[j] - kk[j, c] * prescribed_value
        
        # Zera linha e coluna
        kk[c, :] = 0
        kk[:, c] = 0
        
        # Coloca 1 na diagonal
        kk[c, c] = 1
        
        # Define valor prescrito
        ff[c] = prescribed_value
    
    return kk, ff


def feaplyc2_penalty(kk: np.ndarray, ff: np.ndarray,
                     bcdof: np.ndarray, bcval: np.ndarray,
                     penalty: float = 1e20) -> tuple:
    """
    Aplica condições de contorno pelo método da penalidade.
    
    Adiciona um valor muito grande (penalidade) à diagonal e ao vetor de forças,
    forçando o deslocamento a assumir o valor prescrito.
    
    Parâmetros:
    -----------
    kk : np. ndarray
        Matriz de rigidez do sistema
    ff :  np.ndarray
        Vetor de forças do sistema
    bcdof : np.ndarray
        Vetor de DOFs restringidos (base-1)
    bcval : np.ndarray
        Vetor de valores prescritos
    penalty : float
        Fator de penalidade (default:  1e20)
    
    Retorna:
    --------
    tuple : (kk, ff)
        Sistema modificado pelo método da penalidade
    
    Vantagem:
    ---------
    Mantém a estrutura da matriz original e é mais fácil de implementar
    em códigos com matrizes esparsas.
    """
    
    n = len(bcdof)
    
    for i in range(n):
        c = bcdof[i] - 1  # Conversão para base-0
        
        # Adiciona penalidade à diagonal
        kk[c, c] = kk[c, c] + penalty
        
        # Ajusta vetor de forças
        ff[c] = penalty * bcval[i]
    
    return kk, ff