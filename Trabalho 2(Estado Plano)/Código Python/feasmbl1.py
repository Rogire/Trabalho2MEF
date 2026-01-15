import numpy as np

def feasmbl1(kk: np.ndarray, k: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Montagem (assembly) das matrizes de rigidez elementares na matriz global.
    
    Parâmetros:
    -----------
    kk : np.ndarray
        Matriz de rigidez global do sistema (será modificada)
    k : np.ndarray
        Matriz de rigidez do elemento
    index : np.ndarray
        Vetor de DOFs globais associados ao elemento
    
    Retorna:
    --------
    kk : np.ndarray
        Matriz de rigidez global atualizada
    
    Notas:
    ------
    Esta função assume que 'index' contém índices em base-1 (convenção MATLAB).
    Se usar índices base-0, remova o '-1' nas linhas de ii e jj.
    """
    
    edof = len(index)
    
    for i in range(edof):
        ii = index[i] - 1  # Conversão base-1 → base-0
        for j in range(edof):
            jj = index[j] - 1  # Conversão base-1 → base-0
            kk[ii, jj] = kk[ii, jj] + k[i, j]
    
    return kk


def feasmbl1_vectorized(kk: np.ndarray, k: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Versão vetorizada (mais eficiente) da montagem da matriz global.
    
    Parâmetros:
    -----------
    kk : np.ndarray
        Matriz de rigidez global do sistema
    k : np. ndarray
        Matriz de rigidez do elemento
    index :  np.ndarray
        Vetor de DOFs globais (base-0)
    
    Retorna:
    --------
    kk : np.ndarray
        Matriz de rigidez global atualizada
    """
    
    # Cria grade de índices para atribuição vetorizada
    # np.ix_ cria arrays de índices para indexação avançada
    idx = np.ix_(index, index)
    
    # Soma vetorizada - muito mais rápida para grandes sistemas
    kk[idx] += k
    
    return kk


def feasmbl1_scipy(kk, k:  np.ndarray, index: np.ndarray, sdof: int):
    """
    Versão usando scipy.sparse para sistemas muito grandes.
    
    Parâmetros:
    -----------
    kk : scipy.sparse matrix ou None
        Matriz esparsa global (None para criar nova)
    k : np.ndarray
        Matriz de rigidez do elemento
    index : np.ndarray
        Vetor de DOFs globais (base-0)
    sdof : int
        Número total de DOFs do sistema
    
    Retorna:
    --------
    kk : scipy.sparse. lil_matrix
        Matriz de rigidez global esparsa
    """
    from scipy import sparse
    
    if kk is None:
        kk = sparse.lil_matrix((sdof, sdof))
    
    edof = len(index)
    for i in range(edof):
        for j in range(edof):
            kk[index[i], index[j]] += k[i, j]
    
    return kk