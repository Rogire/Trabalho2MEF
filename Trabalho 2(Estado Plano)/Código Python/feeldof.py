import numpy as np

def feeldof(nd: np.ndarray, nnel: int, ndof: int) -> np.ndarray:
    """
    Calcula os graus de liberdade do sistema associados a cada elemento. 
    
    Parâmetros:
    -----------
    nd : np.ndarray
        Vetor com os números dos nós do elemento (conectividade)
    nnel : int
        Número de nós por elemento
    ndof :  int
        Número de graus de liberdade por nó
    
    Retorna: 
    --------
    index : np.ndarray
        Vetor de DOFs globais associados ao elemento
    
    Exemplo:
    --------
    Para um elemento triangular (nnel=3) com nós [1, 3, 2] e ndof=2:
    >>> nd = np.array([1, 3, 2])
    >>> index = feeldof(nd, 3, 2)
    >>> print(index)  # [1, 2, 5, 6, 3, 4] (em notação base-1)
    """
    
    # Número total de DOFs do elemento
    edof = nnel * ndof
    
    # Pré-alocação do vetor de índices
    index = np.zeros(edof, dtype=int)
    
    k = 0  # Contador para posição no vetor index
    
    for i in range(nnel):
        # Calcula o DOF inicial para o nó nd[i]
        # nd[i] - 1 porque a numeração de nós começa em 1
        start = (nd[i] - 1) * ndof
        
        for j in range(ndof):
            # Em Python, index[k] já está em base-0
            # start + j + 1 para manter compatibilidade com numeração base-1 dos DOFs
            index[k] = start + j + 1
            k += 1
    
    return index


def feeldof_pythonic(nd: np.ndarray, nnel: int, ndof:  int) -> np.ndarray:
    """
    Versão alternativa mais Pythonica usando list comprehension.
    Retorna índices em base-0 (convenção Python).
    
    Parâmetros: 
    -----------
    nd : np. ndarray
        Vetor com os números dos nós do elemento (base-1)
    nnel : int
        Número de nós por elemento
    ndof : int
        Número de graus de liberdade por nó
    
    Retorna:
    --------
    index : np.ndarray
        Vetor de DOFs globais em base-0 (convenção Python)
    """
    
    # Usando list comprehension para construção mais elegante
    index = np.array([
        (nd[i] - 1) * ndof + j
        for i in range(nnel)
        for j in range(ndof)
    ], dtype=int)
    
    return index